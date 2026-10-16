#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mapping of node IDs across the V3 node and edge tables."""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import yaml


class MapeadorIDsV3:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        self.node_mappings = {}
        self.edge_mappings = {}
        self.missing_ids = {}
        self.statistics = {}
        
        self.input_dir = Path(self.config['paths']['input_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Mapeador IDs V3 inicializado")
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('id_mapeamento_v3')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(
            log_dir / 'id_mapeamento_v3.log',
            encoding='utf-8'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def processar_mapeamentos_completos(self) -> bool:
        try:
            self.logger.info("Processando mapeamentos completos V3...")
            
            self._mapear_todos_nos()
            
            self._mapear_todas_arestas()
            
            self._validar_consistencia()
            
            self._gerar_estatisticas()
            
            self._salvar_mapeamentos()
            
            self.logger.info("✓ Mapeamentos completos processados")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during the processing: {str(e)}", exc_info=True)
            return False
    
    def _mapear_todos_nos(self):
        self.logger.info('Mapping all the nodes...')
        
        for node_type, config in self.config['input']['nodes'].items():
            filepath = self.input_dir / config['file']
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                id_col = config['id_column']
                
                if id_col in df.columns:
                    unique_ids = df[id_col].dropna().unique()
                    
                    mapping = {str(id_val): str(id_val) for id_val in unique_ids}
                    
                    self.node_mappings[node_type] = {
                        'original_to_normalized': mapping,
                        'normalized_to_original': {v: k for k, v in mapping.items()},
                        'total_ids': len(unique_ids),
                        'missing_ids': df[id_col].isnull().sum()
                    }
                    
                    self.logger.info(f"✓ {node_type}: {len(unique_ids):,} unique IDs")
                
                else:
                    self.logger.warning(f"Column {id_col} not found in {node_type}")
            
            else:
                self.logger.warning(f"The file was not found: {filepath}")
    
    def _mapear_todas_arestas(self):
        self.logger.info('Mapping all the edges...')
        
        for edge_type, config in self.config['input']['edges'].items():
            filepath = self.input_dir / config['file']
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                source_col = config['source_column']
                target_col = config['target_column']
                
                if source_col in df.columns and target_col in df.columns:
                    source_ids = df[source_col].dropna().unique()
                    target_ids = df[target_col].dropna().unique()
                    all_edge_ids = set(source_ids) | set(target_ids)
                    
                    mapping = {str(id_val): str(id_val) for id_val in all_edge_ids}
                    
                    self.edge_mappings[edge_type] = {
                        'original_to_normalized': mapping,
                        'normalized_to_original': {v: k for k, v in mapping.items()},
                        'source_ids': len(source_ids),
                        'target_ids': len(target_ids),
                        'unique_ids': len(all_edge_ids),
                        'missing_source': df[source_col].isnull().sum(),
                        'missing_target': df[target_col].isnull().sum()
                    }
                    
                    self.logger.info(f"✓ {edge_type}: {len(all_edge_ids):,} unique IDs")
                
                else:
                    self.logger.warning(f"Columns {source_col}/{target_col} not found in {edge_type}")
            
            else:
                self.logger.warning(f"The file was not found: {filepath}")
    
    def _validar_consistencia(self):
        self.logger.info("Validando consistência...")
        
        all_node_ids = set()
        for node_type, mapping in self.node_mappings.items():
            all_node_ids.update(mapping['original_to_normalized'].keys())
        
        all_edge_ids = set()
        for edge_type, mapping in self.edge_mappings.items():
            all_edge_ids.update(mapping['original_to_normalized'].keys())
        
        missing_in_nodes = all_edge_ids - all_node_ids
        missing_in_edges = all_node_ids - all_edge_ids
        
        self.missing_ids = {
            'ids_em_arestas_nao_em_nos': list(missing_in_nodes),
            'ids_em_nos_nao_em_arestas': list(missing_in_edges),
            'total_missing_nodes': len(missing_in_nodes),
            'total_missing_edges': len(missing_in_edges)
        }
        
        if missing_in_nodes:
            self.logger.warning(f"{len(missing_in_nodes):,} IDs in edges not found in nodes")
        
        if missing_in_edges:
            self.logger.info(f"{len(missing_in_edges):,} IDs in nodes not connected by edges")
        
        self.logger.info('V3: keeping all IDs to preserve full integrity')
    
    def _gerar_estatisticas(self):
        self.logger.info("Gerando estatísticas...")
        
        node_stats = {}
        total_nodes = 0
        
        for node_type, mapping in self.node_mappings.items():
            stats = {
                'total_ids': mapping['total_ids'],
                'missing_ids': mapping['missing_ids'],
                'coverage_pct': (mapping['total_ids'] / (mapping['total_ids'] + mapping['missing_ids'])) * 100 if (mapping['total_ids'] + mapping['missing_ids']) > 0 else 0
            }
            node_stats[node_type] = stats
            total_nodes += mapping['total_ids']
        
        edge_stats = {}
        total_edges = 0
        
        for edge_type, mapping in self.edge_mappings.items():
            stats = {
                'unique_ids': mapping['unique_ids'],
                'source_ids': mapping['source_ids'],
                'target_ids': mapping['target_ids'],
                'missing_source': mapping['missing_source'],
                'missing_target': mapping['missing_target']
            }
            edge_stats[edge_type] = stats
            total_edges += mapping['unique_ids']
        
        self.statistics = {
            'timestamp': datetime.now().isoformat(),
            'version': '3.0',
            'nodes': {
                'by_type': node_stats,
                'total_unique_ids': total_nodes,
                'total_types': len(node_stats)
            },
            'edges': {
                'by_type': edge_stats,
                'total_unique_ids': total_edges,
                'total_types': len(edge_stats)
            },
            'consistency': {
                'missing_in_nodes': self.missing_ids['total_missing_nodes'],
                'missing_in_edges': self.missing_ids['total_missing_edges'],
                'integrity_preserved': True
            }
        }
        
        self.logger.info(f"✓ Statistics: {total_nodes:,} nodes, {total_edges:,} edges")
    
    def _salvar_mapeamentos(self):
        self.logger.info("Salvando mapeamentos...")
        
        full_mappings = {
            'version': '3.0',
            'timestamp': datetime.now().isoformat(),
            'nodes': self.node_mappings,
            'edges': self.edge_mappings,
            'missing_ids': self.missing_ids,
            'statistics': self.statistics
        }
        
        mapping_path = self.output_dir / self.config['output']['files']['id_mapping']
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(full_mappings, f, indent=2, ensure_ascii=False)
        
        mappings_dir = self.output_dir / 'mappings'
        mappings_dir.mkdir(exist_ok=True)
        
        for node_type, mapping in self.node_mappings.items():
            node_path = mappings_dir / f'nodes_{node_type}_mapping.json'
            with open(node_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        for edge_type, mapping in self.edge_mappings.items():
            edge_path = mappings_dir / f'edges_{edge_type}_mapping.json'
            with open(edge_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        inconsist_path = mappings_dir / 'inconsistencies_report.json'
        with open(inconsist_path, 'w', encoding='utf-8') as f:
            json.dump(self.missing_ids, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ Mappings saved to {mapping_path}")
    
    def obter_id_normalizado(self, original_id: str, node_type: str) -> Optional[str]:
        if node_type in self.node_mappings:
            return self.node_mappings[node_type]['original_to_normalized'].get(str(original_id))
        return None
    
    def obter_id_original(self, normalized_id: str, node_type: str) -> Optional[str]:
        if node_type in self.node_mappings:
            return self.node_mappings[node_type]['normalized_to_original'].get(str(normalized_id))
        return None
    
    def obter_estatisticas(self) -> Dict[str, Any]:
        return self.statistics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "normalizacao_v3.yaml"
    
    try:
        mapeador = MapeadorIDsV3(config_path)
        success = mapeador.processar_mapeamentos_completos()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        sys.exit(1) 
