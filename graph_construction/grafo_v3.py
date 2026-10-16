#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds the V3 heterogeneous graph (PEA, ERB, CPS, CTC, CTNC) as a PyTorch Geometric HeteroData object."""

import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import gc
import warnings

import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

print(f"Using PyTorch version: {torch.__version__}")
print(f"Using device: CPU (configured)")

from tqdm.auto import tqdm

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ARPIAGrafoBuilderV3:
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        self.graph = HeteroData()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': self.config['project']['version'],
            'config_file': str(self.config_path),
            'statistics': {},
            'force_model_params': self.config['force_model']
        }
        
        self._node_id_mappings = {}
        self._reverse_id_mappings = {}
        
        self._node_coordinates = {}
        
        self.force_params = self.config['force_model']
        
        self.id_issues = {
            'missing_source_ids': defaultdict(set),
            'missing_target_ids': defaultdict(set),
            'fixed_edges': defaultdict(int),
            'removed_edges': defaultdict(int),
            'summary': {}
        }

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_device(self) -> torch.device:
        print('Using the CPU for processing (configured in the YAML).')
        return torch.device('cpu')

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ARPIA_Grafo_V3')
        logger.setLevel(self.config.get('logging', {}).get('level', 'INFO').upper())
        
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        output_dir = Path(self.config['paths']['output']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / self.config.get('logging', {}).get('log_filename', 'grafo_v3.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _validate_ids_consistency(self):
        self.logger.info('Validating the consistency of the V3 IDs...')
        
        output_dir = Path(self.config['paths']['output']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        inconsistency_stats = {}
        
        for edge_type, edge_config in self.config['edge_processing']['edge_types_config'].items():
            input_file = self._get_edge_input_file(edge_type)
            if not input_file.exists():
                continue
                
            df = pd.read_parquet(input_file)
            if df.empty:
                continue
                
            if not edge_config.get('dynamic_typing', False):
                source_type = edge_config['source_type']
                target_type = edge_config['target_type']
                source_id_col = edge_config['source_id_col']
                target_id_col = edge_config['target_id_col']
                
                if source_type not in self._node_id_mappings or target_type not in self._node_id_mappings:
                    self.logger.warning(f"Cannot validate '{edge_type}' - node mappings not available")
                    continue
                    
                source_mapping = self._node_id_mappings[source_type]
                target_mapping = self._node_id_mappings[target_type]
                
                df[source_id_col] = df[source_id_col].astype(str)
                df[target_id_col] = df[target_id_col].astype(str)
                
                source_ids = set(df[source_id_col].unique())
                target_ids = set(df[target_id_col].unique())
                
                source_valid = set(source_mapping.keys())
                target_valid = set(target_mapping.keys())
                
                source_missing = source_ids - source_valid
                target_missing = target_ids - target_valid
                
                edge_stats = {
                    'total_edges': len(df),
                    'source_type': source_type,
                    'target_type': target_type,
                    'source_ids': {
                        'total': len(source_ids),
                        'valid': len(source_ids.intersection(source_valid)),
                        'missing': len(source_missing),
                        'missing_percentage': round(len(source_missing) / len(source_ids) * 100, 2) if source_ids else 0
                    },
                    'target_ids': {
                        'total': len(target_ids),
                        'valid': len(target_ids.intersection(target_valid)),
                        'missing': len(target_missing),
                        'missing_percentage': round(len(target_missing) / len(target_ids) * 100, 2) if target_ids else 0
                    }
                }
                
                inconsistency_stats[edge_type] = edge_stats
                
                self.logger.info(f"Validação de '{edge_type}':")
                self.logger.info(f"  • {source_type}: {len(source_valid):,} IDs available, {len(source_ids):,} IDs in the edge")
                self.logger.info(f"  • {target_type}: {len(target_valid):,} IDs available, {len(target_ids):,} IDs in the edge")
                
                if source_missing:
                    self.logger.warning(f"  • {len(source_missing):,} source IDs not found ({round(len(source_missing) / len(source_ids) * 100, 2)}%)")
                    self.id_issues['missing_source_ids'][edge_type].update(source_missing)
                    
                if target_missing:
                    self.logger.warning(f"  • {len(target_missing):,} target IDs not found ({round(len(target_missing) / len(target_ids) * 100, 2)}%)")
                    self.id_issues['missing_target_ids'][edge_type].update(target_missing)
        
        self.id_issues['summary'] = inconsistency_stats
        inconsistency_file = output_dir / "id_inconsistency_analysis_v3.json"
        with open(inconsistency_file, 'w', encoding='utf-8') as f:
            serializable_issues = {
                'missing_source_ids': {k: list(v) for k, v in self.id_issues['missing_source_ids'].items()},
                'missing_target_ids': {k: list(v) for k, v in self.id_issues['missing_target_ids'].items()},
                'fixed_edges': dict(self.id_issues['fixed_edges']),
                'removed_edges': dict(self.id_issues['removed_edges']),
                'summary': self.id_issues['summary']
            }
            json.dump(serializable_issues, f, indent=2)
        
        self.logger.info(f"Analysis of V3 ID inconsistencies saved to: {inconsistency_file}")
        self.logger.info('Validation of the V3 IDs finished')

    def build(self):
        self.logger.info(f"Starting the construction of the ARPIA V3 graph v{self.config['project']['version']}...")
        
        self.logger.info("Processando nós V3...")
        self._process_all_nodes()
        
        
        self.logger.info('Processing V3 edges (NO filters - data already validated)...')
        self._process_all_edges()
        
        if self.force_params.get('enabled', False):
            self.logger.info('Applying the force model for initial embeddings...')
            self._apply_force_model()
        else:
            self.logger.info('Force model DISABLED - keeping the original embeddings for training')
        
        if self.config.get('add_relative_features', False):
            self.logger.info('Adding relative localization features...')
            self._add_relative_location_features()
        else:
            self.logger.info('Relative localization features DISABLED - let the GNN learn')
        
        self.logger.info("Validando grafo V3...")
        self._validate_graph()
        
        self.logger.info("Salvando artefatos V3...")
        self._save_artifacts()
        
        self.logger.info('Construction of the V3 graph finished successfully!')
        return self.graph

    def _process_all_nodes(self):
        node_id_col = self.config['node_processing']['node_id_column']
        
        for node_type in self.config['graph_schema']['node_types']:
            self.logger.info(f"Processing nodes of type '{node_type}'...")
            
            input_file = self._get_node_input_file(node_type)
            if not input_file.exists():
                self.logger.warning(f"File for nodes '{node_type}' not found: {input_file}")
                self.graph[node_type].num_nodes = 0
                self._node_id_mappings[node_type] = {}
                self._reverse_id_mappings[node_type] = {}
                continue
            
            df = pd.read_parquet(input_file)
            if df.empty:
                self.logger.warning(f"No data found for nodes '{node_type}'")
                self.graph[node_type].num_nodes = 0
                self._node_id_mappings[node_type] = {}
                self._reverse_id_mappings[node_type] = {}
                continue
            
            null_count = df[node_id_col].isna().sum()
            if null_count > 0:
                self.logger.warning(f"  • {null_count:,} null values found in the ID for '{node_type}'. Removing...")
                df = df.dropna(subset=[node_id_col])
            
            df[node_id_col] = df[node_id_col].astype(str)
            
            duplicates = df[node_id_col].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"  • {duplicates:,} duplicate IDs found for '{node_type}'. Removing...")
                df = df.drop_duplicates(subset=[node_id_col])
            
            ids = df[node_id_col].unique()
            id_to_idx = {id_val: idx for idx, id_val in enumerate(ids)}
            self._node_id_mappings[node_type] = id_to_idx
            self._reverse_id_mappings[node_type] = {idx: id_val for id_val, idx in id_to_idx.items()}
            
            self.logger.info(f"  • {node_type}: {len(ids):,} unique IDs processed")
            
            self.graph[node_type].num_nodes = len(ids)
            
            features_df = df.set_index(node_id_col).loc[ids].reset_index()
            
            features, coords = self._extract_node_features_by_type(features_df, node_type)
            
            self.graph[node_type].x = features.to(self.device)
            self.logger.info(f"  • Features extracted for {node_type}: {features.shape[1]} dimensions")
            
            if coords is not None:
                self._node_coordinates[node_type] = coords
            
            if 'operadora' in features_df.columns:
                operadoras = features_df['operadora'].fillna('UNKNOWN')
                operadora_mapping = {op: i for i, op in enumerate(operadoras.unique())}
                operadora_indices = torch.tensor([operadora_mapping[op] for op in operadoras], dtype=torch.long)
                self.graph[node_type].operadora = operadora_indices.to(self.device)
            
            self.logger.info(f"  OK {len(ids):,} nodes of type '{node_type}' processed")

    def _get_node_input_file(self, node_type: str) -> Path:
        base_dir = Path(self.config['paths']['input']['base_dir'])
        node_type_lower = node_type.lower()
        
        file_path = self.config['paths']['input']['nodes'].get(node_type_lower)
        if not file_path:
            file_path = f"nodes_{node_type_lower}_v3.parquet"
        
        return base_dir / file_path

    def _extract_node_features(self, df: pd.DataFrame, feature_cols: List[str]) -> torch.Tensor:
        existing_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Feature columns not found: {missing_cols}")
            self.logger.info(f"Colunas disponíveis: {df.columns.tolist()}")
        
        if not existing_cols:
            self.logger.warning('No feature column found. Using basic features.')
            return torch.ones((len(df), 1), dtype=torch.float32)
        
        features = df[existing_cols].fillna(0).values.astype(np.float32)
        
        if self.config.get('feature_normalization', {}).get('enabled', True):
            features = self._normalize_features(features)
        
        return torch.from_numpy(features)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std

    def _process_all_edges(self):
        for edge_type, edge_config in self.config['edge_processing']['edge_types_config'].items():
            self.logger.info(f"Processing edges of type '{edge_type}'...")
            
            input_file = self._get_edge_input_file(edge_type)
            if not input_file.exists():
                self.logger.warning(f"File for edges '{edge_type}' not found: {input_file}")
                continue
            
            df = pd.read_parquet(input_file)
            if df.empty:
                self.logger.warning(f"No data found for edges '{edge_type}'")
                continue
            
            if edge_config.get('dynamic_typing', False):
                self._process_dynamic_edges(df, edge_type, edge_config)
            else:
                self._process_static_edges(df, edge_type, edge_config, handle_inconsistencies=False)

    def _get_edge_input_file(self, edge_type: str) -> Path:
        base_dir = Path(self.config['paths']['input']['base_dir'])
        edge_type_lower = edge_type.lower()
        
        file_path = self.config['paths']['input']['edges'].get(edge_type_lower)
        if not file_path:
            file_path = f"edges_{edge_type_lower}_v3.parquet"
        
        return base_dir / file_path

    def _normalize_node_id(self, node_id: str, expected_type: str = None) -> str:
        if not isinstance(node_id, str):
            node_id = str(node_id)
        
        if node_id.startswith('PEA_PEA_'):
            return node_id.replace('PEA_PEA_', 'PEA_')
        elif node_id.startswith('ERB_ERB_'):
            return node_id.replace('ERB_ERB_', 'ERB_')
        elif node_id.startswith('CPS_CPS_'):
            return node_id.replace('CPS_CPS_', 'CPS_')
        elif node_id.startswith('CTC_CTC_'):
            return node_id.replace('CTC_CTC_', 'CTC_')
        elif node_id.startswith('CTNC_CTNC_'):
            return node_id.replace('CTNC_CTNC_', 'CTNC_')
        
        if expected_type and not node_id.startswith(f"{expected_type}_"):
            return f"{expected_type}_{node_id}"
        
        return node_id

    def _infer_node_type_from_id(self, node_id: str) -> str:
        if not isinstance(node_id, str):
            node_id = str(node_id)
        
        node_id = self._normalize_node_id(node_id)
        
        if node_id.startswith('CTC_'):
            return 'CTC'
        elif node_id.startswith('CTNC_'):
            return 'CTNC'
        elif node_id.startswith('PEA_'):
            return 'PEA'
        elif node_id.startswith('ERB_'):
            return 'ERB'
        elif node_id.startswith('CPS_'):
            return 'CPS'
        else:
            parts = node_id.split('_')
            if len(parts) > 0:
                return parts[0].upper()
            return 'UNKNOWN'

    def _process_static_edges(self, df: pd.DataFrame, edge_type: str, edge_config: Dict, handle_inconsistencies: bool = False):
        source_type = edge_config['source_type']
        target_type = edge_config['target_type']
        canonical_name = edge_config['canonical_name']
        source_id_col = edge_config['source_id_col']
        target_id_col = edge_config['target_id_col']
        
        source_id_mapping = self._node_id_mappings.get(source_type, {})
        target_id_mapping = self._node_id_mappings.get(target_type, {})
        
        if not source_id_mapping or not target_id_mapping:
            self.logger.error(f"ID mapping not found for '{source_type}' or '{target_type}'")
            return
        
        df[source_id_col] = df[source_id_col].astype(str)
        df[target_id_col] = df[target_id_col].astype(str)
        
        df[source_id_col] = df[source_id_col].apply(lambda x: self._normalize_node_id(x, source_type))
        df[target_id_col] = df[target_id_col].apply(lambda x: self._normalize_node_id(x, target_type))
        
        original_source_count = len(df[source_id_col].unique())
        original_target_count = len(df[target_id_col].unique())
        
        original_df = df.copy()
        
        if handle_inconsistencies:
            duplicates = df.duplicated(subset=[source_id_col, target_id_col], keep='first').sum()
            if duplicates > 0:
                self.logger.warning(f"  • {duplicates:,} duplicate edges found in '{edge_type}'. Removing...")
                df = df.drop_duplicates(subset=[source_id_col, target_id_col], keep='first')
            
            df_valid = df[
                df[source_id_col].isin(source_id_mapping.keys()) & 
                df[target_id_col].isin(target_id_mapping.keys())
            ]
            
            filtered_count = len(df) - len(df_valid)
            if filtered_count > 0:
                self.logger.warning(f"  • {filtered_count:,} edges with invalid IDs were filtered from '{edge_type}' ({filtered_count/len(df)*100:.1f}%)")
                
                missing_source = set(df[source_id_col]) - set(source_id_mapping.keys())
                missing_target = set(df[target_id_col]) - set(target_id_mapping.keys())
                
                if missing_source:
                    self.logger.info(f"  • Source IDs not found in {source_type}: {list(missing_source)[:10]}...")
                if missing_target:
                    self.logger.info(f"  • Target IDs not found in {target_type}: {list(missing_target)[:10]}...")
                
                self.id_issues['removed_edges'][edge_type] = filtered_count
                df = df_valid
        
        valid_edges = []
        valid_indices = []
        
        df_reset = df.reset_index(drop=True)
        
        all_source_ids = set(df_reset[source_id_col].unique())
        all_target_ids = set(df_reset[target_id_col].unique())
        
        for source_id in all_source_ids:
            if source_id not in source_id_mapping:
                new_idx = len(source_id_mapping)
                source_id_mapping[source_id] = new_idx
                self._reverse_id_mappings[source_type][new_idx] = source_id
        
        for target_id in all_target_ids:
            if target_id not in target_id_mapping:
                new_idx = len(target_id_mapping)
                target_id_mapping[target_id] = new_idx
                self._reverse_id_mappings[target_type][new_idx] = target_id
        
        if len(source_id_mapping) > self.graph[source_type].num_nodes:
            self.graph[source_type].num_nodes = len(source_id_mapping)
            self.logger.info(f"  • Atualizado num_nodes para {source_type}: {len(source_id_mapping)}")
        
        if len(target_id_mapping) > self.graph[target_type].num_nodes:
            self.graph[target_type].num_nodes = len(target_id_mapping)
            self.logger.info(f"  • Atualizado num_nodes para {target_type}: {len(target_id_mapping)}")
        
        for idx, row in df_reset.iterrows():
            source_id = row[source_id_col]
            target_id = row[target_id_col]
            
            source_idx = source_id_mapping[source_id]
            target_idx = target_id_mapping[target_id]
            valid_edges.append((source_idx, target_idx))
            valid_indices.append(idx)
        
        total_edges = len(original_df)
        valid_count = len(valid_edges)
        invalid_count = 0
        
        self.logger.info(f"  • Arestas '{edge_type}': {valid_count:,} processadas, {invalid_count:,} descartadas")
        self.logger.info(f"  • IDs mapeados: {len(source_id_mapping):,} sources, {len(target_id_mapping):,} targets")
        
        if not valid_edges:
            self.logger.warning(f"No edge found for '{edge_type}'")
            return
        
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        canonical_edge_type = (source_type, canonical_name, target_type)
        
        max_source_idx = edge_index[0].max().item() if edge_index.shape[1] > 0 else -1
        max_target_idx = edge_index[1].max().item() if edge_index.shape[1] > 0 else -1
        
        if max_source_idx >= len(source_id_mapping):
            self.logger.error(f"Invalid source index {max_source_idx} >= {len(source_id_mapping)} for {source_type}")
            raise ValueError(f"Invalid source index for {canonical_edge_type}")
            
        if max_target_idx >= len(target_id_mapping):
            self.logger.error(f"Invalid target index {max_target_idx} >= {len(target_id_mapping)} for {target_type}")
            raise ValueError(f"Invalid target index for {canonical_edge_type}")
        
        self.graph[canonical_edge_type].edge_index = edge_index.to(self.device)
        
        self._process_edge_features(df_reset.iloc[valid_indices], canonical_edge_type, edge_config)
        
        self.logger.info(f"  OK {len(valid_edges):,} edges of type '{canonical_edge_type}' processed")

    def _process_dynamic_edges(self, df: pd.DataFrame, edge_type: str, edge_config: Dict):
        source_id_col = edge_config['source_id_col']
        target_id_col = edge_config['target_id_col']
        source_type_col = edge_config['source_type_col']
        target_type_col = edge_config['target_type_col']
        canonical_name = edge_config['canonical_name']
        
        df[source_id_col] = df[source_id_col].astype(str)
        df[target_id_col] = df[target_id_col].astype(str)
        
        if source_type_col not in df.columns or target_type_col not in df.columns:
            self.logger.warning(f"Type columns not found for '{edge_type}'. Inferring types from the IDs...")
            
            df[source_type_col] = df[source_id_col].apply(self._infer_node_type_from_id)
            df[target_type_col] = df[target_id_col].apply(self._infer_node_type_from_id)
            
            self.logger.info(f"Tipos inferidos para '{edge_type}': {df[source_type_col].value_counts().to_dict()}")
        
        df[source_id_col] = df[source_id_col].apply(self._normalize_node_id)
        df[target_id_col] = df[target_id_col].apply(self._normalize_node_id)
        
        grouped_edges = defaultdict(list)
        grouped_indices = defaultdict(list)
        invalid_counts = defaultdict(int)
        
        for idx, row in df.iterrows():
            source_id = row[source_id_col]
            target_id = row[target_id_col]
            source_type = row[source_type_col]
            target_type = row[target_type_col]
            
            edge_type_key = (source_type, canonical_name, target_type)
            
            source_mapping = self._node_id_mappings.setdefault(source_type, {})
            target_mapping = self._node_id_mappings.setdefault(target_type, {})
            
            if source_id not in source_mapping:
                new_idx = len(source_mapping)
                source_mapping[source_id] = new_idx
                self._reverse_id_mappings.setdefault(source_type, {})[new_idx] = source_id
                if hasattr(self.graph[source_type], 'num_nodes') and len(source_mapping) > self.graph[source_type].num_nodes:
                    self.graph[source_type].num_nodes = len(source_mapping)
            
            if target_id not in target_mapping:
                new_idx = len(target_mapping)
                target_mapping[target_id] = new_idx
                self._reverse_id_mappings.setdefault(target_type, {})[new_idx] = target_id
                if hasattr(self.graph[target_type], 'num_nodes') and len(target_mapping) > self.graph[target_type].num_nodes:
                    self.graph[target_type].num_nodes = len(target_mapping)
            
            source_idx = source_mapping[source_id]
            target_idx = target_mapping[target_id]
            grouped_edges[edge_type_key].append((source_idx, target_idx))
            grouped_indices[edge_type_key].append(idx)
        
        for type_key, edges in grouped_edges.items():
            if not edges:
                continue
            
            if type_key in invalid_counts and invalid_counts[type_key] > 0:
                self.logger.warning(f"  • {invalid_counts[type_key]:,} invalid edges ignored for '{type_key}'")
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.graph[type_key].edge_index = edge_index.to(self.device)
            
            if type_key in grouped_indices:
                subset_df = df.iloc[grouped_indices[type_key]]
                self._process_edge_features(subset_df, type_key, edge_config)
            
            self.logger.info(f"  OK {len(edges):,} edges of type '{type_key}' processed")

    def _process_edge_features(self, df: pd.DataFrame, edge_type: Tuple, edge_config: Dict):
        feature_cols = edge_config.get('features', [])
        if not feature_cols:
            return
        
        existing_cols = [col for col in feature_cols if col in df.columns]
        if not existing_cols:
            return
        
        features = df[existing_cols].fillna(0).values.astype(np.float32)
        
        if self.config.get('feature_normalization', {}).get('enabled', True):
            features = self._normalize_features(features)
        
        self.graph[edge_type].edge_attr = torch.from_numpy(features).to(self.device)

    def _apply_force_model(self):
        self.logger.info('Force model DISABLED - preserving the embeddings for GNN training')
        return

    def _add_relative_location_features(self):
        self.logger.info('Relative localization features DISABLED - let the GNN learn')
        return

    def _validate_graph(self):
        self.metadata['statistics'] = {
            'total_nodes': self.graph.num_nodes,
            'total_edges': self.graph.num_edges,
            'node_types': {nt: {'count': data.num_nodes} for nt, data in self.graph.node_items()},
            'edge_types': {str(et): {'count': data.num_edges} for et, data in self.graph.edge_items()}
        }
        
        self.metadata['id_consistency'] = {
            'total_removed_edges': sum(self.id_issues['removed_edges'].values()),
            'total_fixed_edges': sum(self.id_issues['fixed_edges'].values()),
            'details_by_edge_type': {
                edge_type: {
                    'removed': self.id_issues['removed_edges'].get(edge_type, 0),
                    'fixed': self.id_issues['fixed_edges'].get(edge_type, 0)
                }
                for edge_type in set(list(self.id_issues['removed_edges'].keys()) + list(self.id_issues['fixed_edges'].keys()))
            }
        }
        
        for node_type, data in self.graph.node_items():
            if hasattr(data, 'x') and data.x is not None:
                self.metadata['statistics']['node_types'][node_type]['feature_dim'] = data.x.shape[1]
        
        for edge_type, data in self.graph.edge_items():
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                self.metadata['statistics']['edge_types'][str(edge_type)]['feature_dim'] = data.edge_attr.shape[1]
        
        self.logger.info(f"Statistics of the V3 graph:")
        self.logger.info(f"  • Total number of nodes: {self.graph.num_nodes:,}")
        self.logger.info(f"  • Total number of edges: {self.graph.num_edges:,}")
        
        for node_type, stats in self.metadata['statistics']['node_types'].items():
            self.logger.info(f"  • Nodes {node_type}: {stats['count']:,} (features: {stats.get('feature_dim', 0)})")
        
        for edge_type, stats in self.metadata['statistics']['edge_types'].items():
            self.logger.info(f"  • Arestas {edge_type}: {stats['count']:,} (features: {stats.get('feature_dim', 0)})")
        
        try:
            self.graph.validate(raise_on_error=True)
            self.logger.info('Validation of the V3 graph: OK')
            self.metadata['validation'] = {'status': 'success'}
        except Exception as e:
            self.logger.error(f"Error in the validation of the V3 graph: {str(e)}")
            self.metadata['validation'] = {'status': 'error', 'message': str(e)}

    def _save_artifacts(self):
        output_dir = Path(self.config['paths']['output']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graph_file = output_dir / self.config['paths']['output']['graph_file']
        torch.save(self.graph, graph_file)
        self.logger.info(f"V3 graph saved to: {graph_file}")
        
        metadata_file = output_dir / self.config['paths']['output']['metadata_file']
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        self.logger.info(f"V3 metadata saved to: {metadata_file}")
        
        id_mapping_file = output_dir / self.config['paths']['output']['id_mapping_file']
        serializable_mapping = {
            'node_mappings': {
                node_type: {str(id_val): int(idx) for id_val, idx in mapping.items()}
                for node_type, mapping in self._node_id_mappings.items()
            },
            'reverse_mappings': {
                node_type: {str(idx): str(id_val) for idx, id_val in mapping.items()}
                for node_type, mapping in self._reverse_id_mappings.items()
            }
        }
        with open(id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_mapping, f, indent=2)
        self.logger.info(f"V3 ID mapping saved to: {id_mapping_file}")
        
        if self._node_coordinates:
            coords_file = output_dir / 'node_coordinates_v3.json'
            coords_serializable = {
                node_type: coords.tolist() for node_type, coords in self._node_coordinates.items()
            }
            with open(coords_file, 'w', encoding='utf-8') as f:
                json.dump(coords_serializable, f, indent=2)
            self.logger.info(f"V3 coordinates saved to: {coords_file}")

    def _extract_node_features_by_type(self, df: pd.DataFrame, node_type: str) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        coords = None
        
        coord_mappings = {
            'PEA': ['pea_x_norm', 'pea_y_norm', 'pea_lat', 'pea_lon'],
            'ERB': ['erb_x_norm', 'erb_y_norm', 'lat_erb', 'lon_erb'], 
            'CPS': ['cps_x_norm', 'cps_y_norm', 'lat_centroide', 'lon_centroide'],
            'CTC': ['ctc_x_norm', 'ctc_y_norm', 'centroid_lat', 'centroid_lon'],
            'CTNC': ['ctnc_x_norm', 'ctnc_y_norm', 'centroid_lat', 'centroid_lon']
        }
        
        available_cols = coord_mappings.get(node_type, [])
        coord_cols = [col for col in available_cols if col in df.columns]
        
        feature_list = []
        
        x_col, y_col = None, None
        
        if node_type == 'PEA':
            for x_candidate, y_candidate in [('pea_x_norm', 'pea_y_norm'), ('pea_lat', 'pea_lon')]:
                if x_candidate in df.columns and y_candidate in df.columns:
                    x_col, y_col = x_candidate, y_candidate
                    break
        elif node_type == 'ERB':
            for x_candidate, y_candidate in [('erb_x_norm', 'erb_y_norm'), ('lat_erb', 'lon_erb')]:
                if x_candidate in df.columns and y_candidate in df.columns:
                    x_col, y_col = x_candidate, y_candidate
                    break
        elif node_type == 'CPS':
            for x_candidate, y_candidate in [('cps_x_norm', 'cps_y_norm'), ('lat_centroide', 'lon_centroide')]:
                if x_candidate in df.columns and y_candidate in df.columns:
                    x_col, y_col = x_candidate, y_candidate
                    break
        elif node_type in ['CTC', 'CTNC']:
            for x_candidate, y_candidate in [('ctc_x_norm', 'ctc_y_norm'), ('ctnc_x_norm', 'ctnc_y_norm'), ('centroid_lat', 'centroid_lon')]:
                if x_candidate in df.columns and y_candidate in df.columns:
                    x_col, y_col = x_candidate, y_candidate
                    break
        
        if x_col and y_col:
            x_values = df[x_col].fillna(0).values.astype(np.float32)
            y_values = df[y_col].fillna(0).values.astype(np.float32)
            
            if 'lat' in x_col.lower() or 'lon' in x_col.lower():
                x_values = self._normalize_coordinates(x_values)
                y_values = self._normalize_coordinates(y_values)
                self.logger.info(f"  • GEOGRAPHIC coordinates for {node_type}: {x_col}, {y_col} (normalized)")
            else:
                self.logger.info(f"  • Coordenadas NORMALIZADAS para {node_type}: {x_col}, {y_col}")
            
            feature_list.extend([x_values, y_values])
            coords = np.column_stack([x_values, y_values])
            
        else:
            self.logger.warning(f"  • No coordinate found for {node_type}")
            placeholder = np.zeros(len(df), dtype=np.float32)
            feature_list.extend([placeholder, placeholder])
        
        if node_type == 'PEA':
            if 'Duração' in df.columns:
                duracao = df['Duração'].fillna(0).values.astype(np.float32)
                duracao_norm = self._normalize_features_simple(duracao)
                feature_list.append(duracao_norm)
                
        elif node_type == 'ERB':
            if 'raio_cobertura_km' in df.columns:
                raio = df['raio_cobertura_km'].fillna(0).values.astype(np.float32)
                raio_norm = self._normalize_features_simple(raio)
                feature_list.append(raio_norm)
                
        elif node_type in ['CTC', 'CTNC']:
            if 'fator_punitivo' in df.columns:
                fator = df['fator_punitivo'].fillna(0).values.astype(np.float32)
                feature_list.append(fator)
        
        if not feature_list:
            feature_list = [np.ones(len(df), dtype=np.float32)]
        
        if len(feature_list) == 1:
            features = feature_list[0].reshape(-1, 1)
        else:
            features = np.column_stack(feature_list)
        
        return torch.from_numpy(features), coords

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        if len(coords) == 0:
            return coords
        
        mean = coords.mean()
        std = coords.std() + 1e-8
        return (coords - mean) / std

    def _normalize_features_simple(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return features
        
        min_val = features.min()
        max_val = features.max()
        
        if max_val - min_val < 1e-8:
            return np.zeros_like(features)
        
        return (features - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(description='ARPIA - V3 Graph Construction for the GNN')
    parser.add_argument("config_path", help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    try:
        builder = ARPIAGrafoBuilderV3(args.config_path)
        builder.build()
        print('\nSUCCESS: V3 graph built successfully!')
    except Exception as e:
        print(f"\nERROR: Error during the construction of the V3 graph: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
