#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 normalization of the node and edge tables (coordinates, IDs and features)."""

import os
import sys
import yaml
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import pyproj
from pyproj import Transformer
from tqdm import tqdm
import gc
import time

try:
    import cupy as cp
    import cudf
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from numba import jit, vectorize, cuda
    NUMBA_AVAILABLE = True
    
    @vectorize(['float64(float64)'])
    def _sin_transform(angle_rad):
        return np.sin(angle_rad)
    
    @vectorize(['float64(float64)'])
    def _cos_transform(angle_rad):
        return np.cos(angle_rad)
        
except ImportError:
    NUMBA_AVAILABLE = False
    
    def _sin_transform(angle_rad):
        return np.sin(angle_rad)
    
    def _cos_transform(angle_rad):
        return np.cos(angle_rad)


class DocumentacaoNormalizacao:
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.sections = []
        self.statistics = {}
        self.transformations = {}
        
    def add_section(self, title: str, content: str):
        self.sections.append({
            'title': title,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_transformation(self, data_type: str, transformation: Dict[str, Any]):
        if data_type not in self.transformations:
            self.transformations[data_type] = []
        self.transformations[data_type].append({
            **transformation,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_statistics(self, key: str, stats: Dict[str, Any]):
        self.statistics[key] = {
            **stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_markdown(self) -> str:
        doc = []
        
        doc.append('# Normalization Documentation - ARPIA V3')
        doc.append(f"**Date of generation:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.append("")
        doc.append("## Índice")
        
        for i, section in enumerate(self.sections, 1):
            doc.append(f"{i}. [{section['title']}](#{section['title'].lower().replace(' ', '-')})")
        doc.append("")
        
        for section in self.sections:
            doc.append(f"## {section['title']}")
            doc.append(section['content'])
            doc.append("")
        
        if self.transformations:
            doc.append("## Transformações Aplicadas")
            for data_type, transforms in self.transformations.items():
                doc.append(f"### {data_type}")
                for transform in transforms:
                    doc.append(f"- **{transform.get('name', 'N/A')}**: {transform.get('description', 'N/A')}")
                    if 'parameters' in transform:
                        doc.append(f"  - Parâmetros: {transform['parameters']}")
                doc.append("")
        
        if self.statistics:
            doc.append('## Statistics of the Data')
            for key, stats in self.statistics.items():
                doc.append(f"### {key}")
                for stat_key, stat_value in stats.items():
                    if stat_key != 'timestamp':
                        doc.append(f"- **{stat_key}**: {stat_value}")
                doc.append("")
        
        return "\n".join(doc)
    
    def save(self):
        content = self.generate_markdown()
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(content)


class NormalizadorGrafoV3:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        self._setup_logging()
        
        self.system_info = self._get_system_info()
        self.logger.info(f"Sistema: {self.system_info}")
        
        self._configure_parallel_processing()
        
        self._setup_utm_transformer()
        
        doc_path = Path(self.config['paths']['output_dir']) / self.config['output']['files']['documentation']
        self.documentation = DocumentacaoNormalizacao(doc_path)
        
        self.scalers = {}
        self.id_mapping = {}
        self.feature_stats = {}
        
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        
        self.memory_threshold = self.config['system'].get('memory_limit_gb', 16) * 1024 * 1024 * 1024
        
        self.logger.info('V3 normalizer initialized successfully')
        self.documentation.add_section(
            "Inicialização",
            f"V3 normalizer initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nConfiguration loaded from: {config_path}\nMode: process ALL data (no limitations)"
        )
    
    def _setup_logging(self):
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('normalizacao_v3')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        log_file = log_dir / 'normalizacao_v3.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_system_info(self) -> Dict[str, Any]:
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': CUDA_AVAILABLE,
            'numba_available': NUMBA_AVAILABLE,
            'python_version': sys.version,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__
        }
    
    def _configure_parallel_processing(self):
        self.max_workers = min(
            self.config['system']['max_workers'],
            psutil.cpu_count()
        )
        self.chunk_size = self.config['system']['chunk_size']
        
        if CUDA_AVAILABLE and self.config['system']['use_gpu']:
            self.logger.info('CUDA available! Using GPU acceleration')
        else:
            self.logger.info("Usando processamento CPU")
    
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['paths', 'input', 'output', 'normalization']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Section '{section}' missing from the configuration")
            
            return config
        except Exception as e:
            raise ValueError(f"Error while loading configuration: {str(e)}")
    
    def _setup_utm_transformer(self):
        source_crs = self.config['coordinates']['source_crs']
        target_crs = self.config['coordinates']['target_crs']
        
        self.utm_transformer = Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
        
        self.logger.info(f"Transformador UTM configurado: {source_crs} -> {target_crs}")
    
    def check_memory_usage(self) -> Dict[str, float]:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def normalizar_grafo_completo(self) -> bool:
        try:
            self.logger.info("=" * 80)
            self.logger.info('STARTING FULL ARPIA V3 NORMALIZATION')
            self.logger.info("=" * 80)
            
            start_time = time.time()
            
            self.logger.info('1. Processing ALL the nodes...')
            if not self._processar_todos_nos():
                return False
            
            self.logger.info("2. Inicializando scalers globais...")
            self._inicializar_scalers_globais()
            
            self.logger.info('3. Applying coordinate transformations...')
            self._aplicar_transformacoes_coordenadas()
            
            self.logger.info('4. Processing ALL the edges...')
            if not self._processar_todas_arestas():
                return False
            
            self.logger.info("5. Validando grafo completo...")
            if not self._validar_grafo_completo():
                return False
            
            self.logger.info("6. Salvando resultados...")
            if not self._salvar_resultados_completos():
                return False
            
            self.logger.info("7. Gerando documentação...")
            self._gerar_documentacao_final()
            
            total_time = time.time() - start_time
            self.logger.info(f"✓ V3 normalization finished in {total_time/60:.2f} minutes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during the normalization: {str(e)}", exc_info=True)
            return False
    
    def _processar_todos_nos(self) -> bool:
        try:
            nodes_config = self.config['input']['nodes']
            input_dir = Path(self.config['paths']['input_dir'])
            
            all_nodes = []
            
            for node_group, config in nodes_config.items():
                self.logger.info(f"Processing node group {node_group.upper()}...")
                
                if Path(config['file']).is_absolute():
                    filepath = Path(config['file'])
                elif config['file'].startswith('../../'):
                    config_dir = Path(self.config_path).parent
                    filepath = config_dir / config['file']
                else:
                    filepath = input_dir / config['file']
                
                if not filepath.exists():
                    if config['required']:
                        raise FileNotFoundError(f"Required file not found: {filepath}")
                    else:
                        self.logger.warning(f"Optional file not found: {filepath}")
                        continue
                
                df = self._load_large_dataframe(filepath)
                
                if node_group == 'consolidated':
                    processed_nodes = self._processar_arquivo_consolidado(df, config)
                    all_nodes.extend(processed_nodes)
                else:
                    if node_group.lower() == 'pea':
                        processed_df = self._processar_nos_pea(df, config)
                    else:
                        processed_df = self._processar_tipo_no(df, node_group.upper(), config)
                    
                    if not processed_df.empty:
                        all_nodes.append(processed_df)
                        self.logger.info(f"✓ {node_group.upper()}: {len(processed_df):,} nodes processed")
            
            if all_nodes:
                self.nodes_df = pd.concat(all_nodes, ignore_index=True)
                self.logger.info(f"Total combined nodes: {len(self.nodes_df):,}")
                
                self._verificar_duplicatas_nos()
                
                return True
            else:
                self.logger.error('No node was processed!')
                return False
                
        except Exception as e:
            self.logger.error(f"Error while processing nodes: {str(e)}", exc_info=True)
            return False
    
    def _processar_arquivo_consolidado(self, df: pd.DataFrame, config: dict) -> List[pd.DataFrame]:
        processed_nodes = []
        
        for node_type, type_config in config['types'].items():
            try:
                self.logger.info(f"Extracting {node_type.upper()} nodes from the consolidated file...")
                
                type_df = df.copy()
                if 'filter_column' in type_config and 'filter_value' in type_config:
                    filter_col = type_config['filter_column']
                    filter_val = type_config['filter_value']
                    type_df = type_df[type_df[filter_col] == filter_val]
                
                id_col = type_config['id_column']
                if id_col in type_df.columns:
                    valid_ids = type_df[id_col].notna()
                    type_df = type_df[valid_ids]
                    
                    if not type_df.empty:
                        processed_df = self._processar_tipo_no_consolidado(type_df, node_type.upper(), type_config)
                        
                        if not processed_df.empty:
                            processed_nodes.append(processed_df)
                            self.logger.info(f"✓ {node_type.upper()}: {len(processed_df):,} nodes processed")
                            
                            self.documentation.add_transformation(
                                f"Nós {node_type.upper()}",
                                {
                                    'name': 'Processing of the Consolidated Nodes',
                                    'description': f"Processing of {len(processed_df):,} nodes of type {node_type.upper()} from a consolidated file",
                                    'parameters': {
                                        'tipo': node_type,
                                        'total_registros': len(processed_df),
                                        'arquivo_origem': config['file']
                                    }
                                }
                            )
                    else:
                        self.logger.warning(f"No valid data found for {node_type.upper()}")
                else:
                    self.logger.warning(f"Column {id_col} not found for {node_type.upper()}")
                    
            except Exception as e:
                self.logger.error(f"Error while processing {node_type}: {str(e)}")
        
        return processed_nodes
    
    def _processar_tipo_no_consolidado(self, df: pd.DataFrame, node_type: str, config: dict) -> pd.DataFrame:
        if df.empty:
            return df
        
        try:
            df = df.copy()
            df['node_type'] = node_type
            
            if 'pea_id' in df.columns:
                df['pea_id'] = df['pea_id'].astype(str).str.replace('PEA_PEA_', 'PEA_', regex=False)
            if 'erb_id' in df.columns:
                df['erb_id'] = df['erb_id'].astype(str).str.replace('ERB_ERB_', 'ERB_', regex=False)
            if 'cps_id' in df.columns:
                df['cps_id'] = df['cps_id'].astype(str).str.replace('CPS_CPS_', 'CPS_', regex=False)
            
            lat_col = config['lat_column']
            lon_col = config['lon_column']
            
            if lat_col in df.columns and lon_col in df.columns:
                df = self._converter_coordenadas_utm(df, lat_col, lon_col, node_type.lower())
            
            id_col = config['id_column']
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str)
                if node_type.lower() == 'ctc' or node_type.lower() == 'ctnc':
                    df[id_col] = df[id_col].str.replace('CTC_CTC_', 'CTC_', regex=False)
                    df[id_col] = df[id_col].str.replace('CTNC_CTNC_', 'CTNC_', regex=False)
                
                df = df.rename(columns={id_col: 'node_id'})
                df = self._normalizar_ids_nos(df, node_type)
            
            df = self._otimizar_memoria_dataframe(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error while processing nodes {node_type}: {str(e)}")
            return pd.DataFrame()
    
    def _load_large_dataframe(self, filepath: Path) -> pd.DataFrame:
        try:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 1000:
                self.logger.info(f"Large file detected ({file_size_mb:.1f}MB), using optimized loading...")
                
                chunks = []
                chunksize = min(self.chunk_size, 100000)
                
                for chunk in pd.read_csv(filepath, chunksize=chunksize):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(filepath)
            
            self.logger.info(f"Carregado: {len(df):,} registros de {filepath.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error while loading {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def _processar_nos_pea(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        try:
            pea_df = df.copy()
            pea_df['node_type'] = 'PEA'
            
            if config['id_column'] in pea_df.columns:
                pea_df[config['id_column']] = pea_df[config['id_column']].astype(str).str.replace('PEA_PEA_', 'PEA_', regex=False)
            
            id_col = config.get('id_column', 'node_id')
            if id_col != 'node_id' and id_col in pea_df.columns:
                pea_df = pea_df.rename(columns={id_col: 'node_id'})
            
            lat_col = config['lat_column']
            lon_col = config['lon_column']
            
            if lat_col in pea_df.columns and lon_col in pea_df.columns:
                pea_df = self._converter_coordenadas_utm(pea_df, lat_col, lon_col, 'pea')
            
            if 'erb_operadora' in pea_df.columns:
                pea_df['operadora'] = pea_df['erb_operadora']
            
            pea_df = self._normalizar_ids_nos(pea_df, 'PEA')
            
            pea_df = self._otimizar_memoria_dataframe(pea_df)
            
            return pea_df
            
        except Exception as e:
            self.logger.error(f"Error while processing PEA nodes: {str(e)}")
            return pd.DataFrame()
    
    def _processar_tipo_no(self, df: pd.DataFrame, node_type: str, config: dict) -> pd.DataFrame:
        if df.empty:
            return df
        
        try:
            df = df.copy()
            df['node_type'] = node_type
            
            id_col = config.get('id_column', 'node_id')
            if id_col != 'node_id' and id_col in df.columns:
                df = df.rename(columns={id_col: 'node_id'})
                self.logger.info(f"✓ {node_type}: Renomeada coluna '{id_col}' para 'node_id'")
            
            if config['lat_column'] in df.columns and config['lon_column'] in df.columns:
                df = self._converter_coordenadas_utm(df, config['lat_column'], config['lon_column'], node_type.lower())
            
            if 'node_id' in df.columns:
                df = self._normalizar_ids_nos(df, node_type)
            
            df = self._otimizar_memoria_dataframe(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error while processing nodes {node_type}: {str(e)}")
            return pd.DataFrame()
    
    def _converter_coordenadas_utm(self, df: pd.DataFrame, lat_col: str, lon_col: str, prefix: str) -> pd.DataFrame:
        try:
            valid_coords = df[lat_col].notna() & df[lon_col].notna()
            
            if valid_coords.sum() == 0:
                self.logger.warning(f"No valid coordinate found for {prefix}")
                return df
            
            x_utm, y_utm = self.utm_transformer.transform(
                df.loc[valid_coords, lon_col].values,
                df.loc[valid_coords, lat_col].values
            )
            
            df[f'{prefix}_x_utm'] = np.nan
            df[f'{prefix}_y_utm'] = np.nan
            df.loc[valid_coords, f'{prefix}_x_utm'] = x_utm
            df.loc[valid_coords, f'{prefix}_y_utm'] = y_utm
            
            self.logger.info(f"✓ {prefix}: {valid_coords.sum():,} coordenadas convertidas para UTM")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in the UTM conversion for {prefix}: {str(e)}")
            return df
    
    def _normalizar_ids_nos(self, df: pd.DataFrame, node_type: str) -> pd.DataFrame:
        try:
            original_ids = df['node_id'].unique()
            
            id_mapping = {}
            for orig_id in original_ids:
                id_mapping[orig_id] = orig_id
            
            self.id_mapping[node_type] = id_mapping
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in the normalization of IDs {node_type}: {str(e)}")
            return df
    
    def _otimizar_memoria_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    if unique_count / total_count < 0.1:
                        df[col] = df[col].astype('category')
                
                elif df[col].dtype == 'float64':
                    if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                        df[col] = df[col].astype('float32')
                
                elif df[col].dtype == 'int64':
                    if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype('int32')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error in the memory optimization: {str(e)}")
            return df
    
    def _verificar_duplicatas_nos(self):
        initial_count = len(self.nodes_df)
        
        if 'node_id' in self.nodes_df.columns:
            duplicates = self.nodes_df.duplicated(subset=['node_id'], keep='first')
            
            if duplicates.sum() > 0:
                self.logger.warning(f"Found {duplicates.sum():,} duplicate nodes")
                self.nodes_df = self.nodes_df[~duplicates]
                
                self.documentation.add_transformation(
                    "Remoção de Duplicatas",
                    {
                        'name': 'Removal of Duplicate Nodes',
                        'description': f"Removed {duplicates.sum():,} duplicate nodes",
                        'parameters': {
                            'registros_iniciais': initial_count,
                            'registros_finais': len(self.nodes_df),
                            'duplicatas_removidas': duplicates.sum()
                        }
                    }
                )
        
        final_count = len(self.nodes_df)
        self.logger.info(f"Duplicate check: {initial_count:,} -> {final_count:,} nodes")
    
    def _inicializar_scalers_globais(self):
        try:
            all_coords = []
            
            for node_type in ['pea', 'erb', 'centroide_cobertura', 'ctc', 'ctnc']:
                x_col = f'{node_type}_x_utm'
                y_col = f'{node_type}_y_utm'
                
                if x_col in self.nodes_df.columns and y_col in self.nodes_df.columns:
                    coords = self.nodes_df[[x_col, y_col]].dropna()
                    if not coords.empty:
                        all_coords.append(coords.values)
            
            if all_coords:
                combined_coords = np.vstack(all_coords)
                
                self.scalers['coordinates'] = StandardScaler()
                self.scalers['coordinates'].fit(combined_coords)
                
                self.logger.info(f"Scalers globais inicializados com {len(combined_coords):,} pontos")
                
                self.documentation.add_transformation(
                    "Scalers Globais",
                    {
                        'name': 'Initialization of the Global Scalers',
                        'description': f'Scalers treinados com {len(combined_coords):,} coordenadas',
                        'parameters': {
                            'scaler_type': 'StandardScaler',
                            'total_pontos': len(combined_coords),
                            'mean_x': float(self.scalers['coordinates'].mean_[0]),
                            'mean_y': float(self.scalers['coordinates'].mean_[1]),
                            'std_x': float(self.scalers['coordinates'].scale_[0]),
                            'std_y': float(self.scalers['coordinates'].scale_[1])
                        }
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Error while initializing scalers: {str(e)}")
    
    def _aplicar_transformacoes_coordenadas(self):
        try:
            if 'coordinates' not in self.scalers:
                self.logger.warning('Coordinate scaler is not available')
                return
            
            total_normalized = 0
            
            for node_type in ['pea', 'erb', 'centroide_cobertura', 'ctc', 'ctnc']:
                x_col = f'{node_type}_x_utm'
                y_col = f'{node_type}_y_utm'
                x_norm_col = f'{node_type}_x_norm'
                y_norm_col = f'{node_type}_y_norm'
                
                if x_col in self.nodes_df.columns and y_col in self.nodes_df.columns:
                    valid_mask = self.nodes_df[x_col].notna() & self.nodes_df[y_col].notna()
                    
                    if valid_mask.sum() > 0:
                        coords = self.nodes_df.loc[valid_mask, [x_col, y_col]].values
                        normalized_coords = self.scalers['coordinates'].transform(coords)
                        
                        self.nodes_df[x_norm_col] = np.nan
                        self.nodes_df[y_norm_col] = np.nan
                        self.nodes_df.loc[valid_mask, x_norm_col] = normalized_coords[:, 0]
                        self.nodes_df.loc[valid_mask, y_norm_col] = normalized_coords[:, 1]
                        
                        total_normalized += valid_mask.sum()
                        self.logger.info(f"✓ {valid_mask.sum():,} coordenadas {node_type.upper()} normalizadas")
            
            self.logger.info(f"Total normalized coordinates: {total_normalized:,}")
            
            self.documentation.add_transformation(
                "Normalização de Coordenadas",
                {
                    'name': 'Application of the Global Normalization',
                    'description': f'Normalizadas {total_normalized:,} coordenadas usando scaler global',
                    'parameters': {
                        'total_coordenadas': total_normalized,
                        'metodo': 'StandardScaler',
                        'escala_global': True
                    }
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error in the coordinate transformation: {str(e)}")
    
    def _processar_todas_arestas(self) -> bool:
        try:
            edges_config = self.config['input']['edges']
            input_dir = Path(self.config['paths']['input_dir'])
            
            all_edges = []
            
            for edge_type, config in edges_config.items():
                self.logger.info(f"Processando arestas {edge_type.upper()}...")
                
                filepath = input_dir / config['file']
                if not filepath.exists():
                    if config['required']:
                        raise FileNotFoundError(f"Required file not found: {filepath}")
                    else:
                        self.logger.warning(f"Optional file not found: {filepath}")
                        continue
                
                df = self._load_large_dataframe(filepath)
                
                if not df.empty:
                    processed_df = self._processar_arestas_arquivo(filepath, edge_type.upper())
                    
                    if not processed_df.empty:
                        all_edges.append(processed_df)
                        self.logger.info(f"✓ {edge_type.upper()}: {len(processed_df):,} arestas processadas")
                        
                        self.documentation.add_transformation(
                            f"Arestas {edge_type.upper()}",
                            {
                                'name': 'Processamento de Arestas',
                                'description': f"Full processing of {len(processed_df):,} edges of type {edge_type.upper()}",
                                'parameters': {
                                    'arquivo': config['file'],
                                    'total_registros': len(processed_df),
                                    'sem_limitacoes': True
                                }
                            }
                        )
            
            if all_edges:
                self.edges_df = pd.concat(all_edges, ignore_index=True)
                self.logger.info(f"Total combined edges: {len(self.edges_df):,}")
                
                return True
            else:
                self.logger.error("Nenhuma aresta foi processada!")
                return False
                
        except Exception as e:
            self.logger.error(f"Error while processing edges: {str(e)}", exc_info=True)
            return False
    
    def _processar_arestas_arquivo(self, arquivo_path: Path, edge_type: str) -> pd.DataFrame:
        try:
            self.logger.info(f"Processing edges {edge_type} of {arquivo_path}")
            
            df = pd.read_csv(arquivo_path, low_memory=False)
            
            if df.empty:
                self.logger.warning(f"Edge file {edge_type} is empty")
                return pd.DataFrame()
            
            if edge_type == 'erb_erb':
                df.columns = ['tipo_aresta', 'valor1', 'valor2', 'source_id', 'target_id']
                df = df[['source_id', 'target_id']].copy()
                df['tipo_aresta'] = 'erb_erb'
                
            elif edge_type == 'intra_triangulo_conexao':
                df['source_type'] = df['source_id'].apply(self._inferir_tipo_do_id)
                df['target_type'] = df['target_id'].apply(self._inferir_tipo_do_id)
                
            elif edge_type == 'intra_triangulo_nao_conexao':
                if 'source_type' not in df.columns:
                    df['source_type'] = df['source_id'].apply(self._inferir_tipo_do_id)
                if 'target_type' not in df.columns:
                    df['target_type'] = df['target_id'].apply(self._inferir_tipo_do_id)
            
            for col in ['source_id', 'target_id']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.replace('PEA_PEA_', 'PEA_', regex=False)
                    df[col] = df[col].str.replace('ERB_ERB_', 'ERB_', regex=False)
                    df[col] = df[col].str.replace('CPS_CPS_', 'CPS_', regex=False)
                    df[col] = df[col].str.replace('CTC_CTC_', 'CTC_', regex=False)
                    df[col] = df[col].str.replace('CTNC_CTNC_', 'CTNC_', regex=False)
            
            if 'edge_type' not in df.columns:
                df['edge_type'] = edge_type
            
            df = self._normalizar_features_arestas(df, edge_type)
            
            self.logger.info(f"✓ {edge_type}: {len(df):,} arestas processadas")
            return df
            
        except Exception as e:
            self.logger.error(f"Error while processing edges {edge_type}: {str(e)}")
            return pd.DataFrame()
    
    def _inferir_tipo_do_id(self, node_id: str) -> str:
        if not isinstance(node_id, str):
            node_id = str(node_id)
        
        if node_id.startswith('PEA_'):
            return 'PEA'
        elif node_id.startswith('ERB_'):
            return 'ERB'
        elif node_id.startswith('CPS_'):
            return 'CPS'
        elif node_id.startswith('CTC_'):
            return 'CTC'
        elif node_id.startswith('CTNC_'):
            return 'CTNC'
        else:
            return 'UNKNOWN'
    
    def _normalizar_features_arestas(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        try:
            if 'distancia_km' in df.columns:
                df['distancia_norm'] = df['distancia_km'] / df['distancia_km'].max()
            
            if 'mesma_operadora' in df.columns:
                df['mesma_operadora_norm'] = df['mesma_operadora'].astype(float)
            
            if 'fator_punitivo' in df.columns:
                df['fator_punitivo_norm'] = df['fator_punitivo']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in the feature normalization for {edge_type}: {str(e)}")
            return df
    
    def _validar_grafo_completo(self) -> bool:
        try:
            total_nodes = len(self.nodes_df)
            total_edges = len(self.edges_df)
            
            self.logger.info(f"Validation: {total_nodes:,} nodes, {total_edges:,} edges")
            
            if 'node_type' in self.nodes_df.columns:
                node_types = self.nodes_df['node_type'].value_counts()
                self.logger.info(f"Node types in use: {dict(node_types)}")
            
            if 'edge_type' in self.edges_df.columns:
                edge_types = self.edges_df['edge_type'].value_counts()
                self.logger.info(f"Edge types in use: {dict(edge_types)}")
            
            if not self.edges_df.empty and 'node_id' in self.nodes_df.columns:
                all_node_ids = set(self.nodes_df['node_id'].astype(str))
                
                connected_ids = set()
                for col in ['source_node_id', 'target_node_id']:
                    if col in self.edges_df.columns:
                        connected_ids.update(self.edges_df[col].astype(str).dropna())
                
                isolated_count = len(all_node_ids - connected_ids)
                isolated_pct = isolated_count / len(all_node_ids) * 100 if all_node_ids else 0
                
                self.logger.info(f"Isolated nodes: {isolated_count:,} ({isolated_pct:.2f}%)")
                
                self.documentation.add_statistics(
                    "Conectividade",
                    {
                        'total_nos': total_nodes,
                        'nos_conectados': len(connected_ids),
                        'nos_isolados': isolated_count,
                        'percentual_isolados': isolated_pct,
                        'observacao': 'V3 keeps all nodes to preserve full integrity'
                    }
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in the validation: {str(e)}")
            return False
    
    def _salvar_resultados_completos(self) -> bool:
        try:
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config['output']['separate_by_type']:
                self._salvar_por_tipo(output_dir)
            
            self._salvar_arquivos_combinados(output_dir)
            
            self._salvar_metadados(output_dir)
            
            self._salvar_scalers(output_dir)
            
            self._salvar_mapeamento_ids(output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error while saving results: {str(e)}")
            return False
    
    def _salvar_por_tipo(self, output_dir: Path):
        if 'node_type' in self.nodes_df.columns:
            for node_type in self.nodes_df['node_type'].unique():
                type_df = self.nodes_df[self.nodes_df['node_type'] == node_type]
                
                parquet_path = output_dir / f"nodes_{node_type.lower()}_v3.parquet"
                type_df.to_parquet(parquet_path, index=False)
                
                csv_path = output_dir / f"nodes_{node_type.lower()}_v3.csv"
                type_df.to_csv(csv_path, index=False)
                
                self.logger.info(f"✓ Saved {node_type} nodes: {len(type_df):,} records")
        
        if 'edge_type' in self.edges_df.columns:
            for edge_type in self.edges_df['edge_type'].unique():
                type_df = self.edges_df[self.edges_df['edge_type'] == edge_type]
                
                parquet_path = output_dir / f"edges_{edge_type.lower()}_v3.parquet"
                type_df.to_parquet(parquet_path, index=False)
                
                csv_path = output_dir / f"edges_{edge_type.lower()}_v3.csv"
                type_df.to_csv(csv_path, index=False)
                
                self.logger.info(f"✓ Saved {edge_type} edges: {len(type_df):,} records")
    
    def _salvar_arquivos_combinados(self, output_dir: Path):
        nodes_parquet = output_dir / self.config['output']['files']['nodes']
        self.nodes_df.to_parquet(nodes_parquet, index=False)
        
        nodes_csv = output_dir / "nodes_normalized_v3.csv"
        self.nodes_df.to_csv(nodes_csv, index=False)
        
        edges_parquet = output_dir / self.config['output']['files']['edges']
        self.edges_df.to_parquet(edges_parquet, index=False)
        
        edges_csv = output_dir / "edges_normalized_v3.csv"
        self.edges_df.to_csv(edges_csv, index=False)
        
        self.logger.info(f"✓ Combined files saved: {len(self.nodes_df):,} nodes, {len(self.edges_df):,} edges")
    
    def _salvar_metadados(self, output_dir: Path):
        metadata = {
            'version': '3.0',
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'system_info': self.system_info,
            'statistics': {
                'total_nodes': len(self.nodes_df),
                'total_edges': len(self.edges_df),
                'node_types': self.nodes_df['node_type'].value_counts().to_dict() if 'node_type' in self.nodes_df.columns else {},
                'edge_types': self.edges_df['edge_type'].value_counts().to_dict() if 'edge_type' in self.edges_df.columns else {},
                'memory_usage': self.check_memory_usage()
            },
            'features': {
                'node_columns': list(self.nodes_df.columns),
                'edge_columns': list(self.edges_df.columns)
            }
        }
        
        metadata_path = output_dir / self.config['output']['files']['metadata']
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"✓ Metadata saved to {metadata_path}")
    
    def _salvar_scalers(self, output_dir: Path):
        scalers_path = output_dir / self.config['output']['files']['scalers']
        
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        self.logger.info(f"✓ Scalers saved to {scalers_path}")
    
    def _salvar_mapeamento_ids(self, output_dir: Path):
        mapping_path = output_dir / self.config['output']['files']['id_mapping']
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_mapping, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ ID mapping saved to {mapping_path}")
    
    def _gerar_documentacao_final(self):
        self.documentation.add_section(
            'Summary of the V3 Normalization',
            f"\nThis normalization processed ALL available data without limitations:\n\n- **Total Nodes**: {len(self.nodes_df):,}\n- **Total Edges**: {len(self.edges_df):,}\n- **Node Types**: {len(self.nodes_df['node_type'].unique()) if 'node_type' in self.nodes_df.columns else 0}\n- **Edge Types**: {len(self.edges_df['edge_type'].unique()) if 'edge_type' in self.edges_df.columns else 0}\n\n**V3 Features**:\n- Full processing without sampling\n- Preservation of the full integrity of the data\n- Consistent global normalization\n- Automatic documentation of transformations\n\n**Memory Use**: {self.check_memory_usage()['used_gb']:.1f}GB\n            "
        )
        
        self.documentation.save()
        self.logger.info(f"✓ Documentation saved to {self.documentation.output_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "normalizacao_v3.yaml"
    
    try:
        normalizador = NormalizadorGrafoV3(config_path)
        
        success = normalizador.normalizar_grafo_completo()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        sys.exit(1) 
