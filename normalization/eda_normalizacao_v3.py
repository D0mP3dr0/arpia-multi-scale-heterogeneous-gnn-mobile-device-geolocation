#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory analysis of the V3 normalized tables."""

import os
import sys
import yaml
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


class EDANormalizacaoV3:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        self.input_dir = Path(self.config['paths']['input_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.eda_dir = self.output_dir / self.config['eda']['output_dir']
        self.eda_dir.mkdir(parents=True, exist_ok=True)
        
        self.nodes_data = {}
        self.edges_data = {}
        self.statistics = {}
        
        self.logger.info('V3 EDA initialized - full processing without limitations')
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('eda_normalizacao_v3')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(
            log_dir / 'eda_normalizacao_v3.log',
            encoding='utf-8'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def executar_eda_completa(self) -> bool:
        try:
            self.logger.info("Iniciando EDA completa V3...")
            
            if not self._carregar_todos_dados():
                return False
            
            self._analisar_nos()
            
            self._analisar_arestas()
            
            self._analisar_espacial()
            
            self._analisar_conectividade()
            
            self._gerar_relatorios()
            
            self.logger.info('✓ Full EDA is finished')
            return True
            
        except Exception as e:
            self.logger.error(f"Error in the EDA: {str(e)}", exc_info=True)
            return False
    
    def _carregar_todos_dados(self) -> bool:
        try:
            for node_group, config in self.config['input']['nodes'].items():
                if Path(config['file']).is_absolute():
                    filepath = Path(config['file'])
                elif config['file'].startswith('../../'):
                    config_dir = Path(self.config_path).parent
                    filepath = config_dir / config['file']
                else:
                    filepath = self.input_dir / config['file']
                
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    
                    if node_group == 'consolidated':
                        self.logger.info(f"Loading the consolidated file: {len(df):,} records")
                        
                        if 'types' in config:
                            for node_type, type_config in config['types'].items():
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
                                        self.nodes_data[node_type] = type_df
                                        self.logger.info(f"✓ Extracted {node_type}: {len(type_df):,} records")
                        else:
                            self.nodes_data[node_group] = df
                            self.logger.info(f"✓ Carregado {node_group}: {len(df):,} registros")
                    else:
                        self.nodes_data[node_group] = df
                        self.logger.info(f"✓ Carregado {node_group}: {len(df):,} registros")
                else:
                    self.logger.warning(f"The file was not found: {filepath}")
            
            for edge_type, config in self.config['input']['edges'].items():
                filepath = self.input_dir / config['file']
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    self.edges_data[edge_type] = df
                    self.logger.info(f"✓ Carregado {edge_type}: {len(df):,} registros")
                else:
                    self.logger.warning(f"The file was not found: {filepath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error while loading data: {str(e)}")
            return False
    
    def _analisar_nos(self):
        self.logger.info("Analisando nós...")
        
        node_stats = {}
        
        for node_type, df in self.nodes_data.items():
            stats = {
                'total_registros': len(df),
                'colunas': list(df.columns),
                'memoria_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'tipos_dados': df.dtypes.to_dict(),
                'valores_nulos': df.isnull().sum().to_dict(),
                'valores_unicos': df.nunique().to_dict()
            }
            
            lat_col = None
            lon_col = None
            
            consolidated_config = self.config['input']['nodes'].get('consolidated', {})
            if 'types' in consolidated_config and node_type in consolidated_config['types']:
                type_config = consolidated_config['types'][node_type]
                lat_col = type_config.get('lat_column')
                lon_col = type_config.get('lon_column')
            else:
                node_config = self.config['input']['nodes'].get(node_type, {})
                lat_col = node_config.get('lat_column')
                lon_col = node_config.get('lon_column')
            
            if lat_col in df.columns and lon_col in df.columns:
                valid_coords = df[lat_col].notna() & df[lon_col].notna()
                stats['coordenadas_validas'] = valid_coords.sum()
                stats['coordenadas_invalidas'] = len(df) - valid_coords.sum()
                
                if valid_coords.sum() > 0:
                    stats['lat_range'] = [df[lat_col].min(), df[lat_col].max()]
                    stats['lon_range'] = [df[lon_col].min(), df[lon_col].max()]
            
            node_stats[node_type] = stats
            self.logger.info(f"✓ Analisado {node_type}: {stats['total_registros']:,} registros")
        
        self.statistics['nodes'] = node_stats
    
    def _analisar_arestas(self):
        self.logger.info("Analisando arestas...")
        
        edge_stats = {}
        
        for edge_type, df in self.edges_data.items():
            stats = {
                'total_registros': len(df),
                'colunas': list(df.columns),
                'memoria_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'tipos_dados': df.dtypes.to_dict(),
                'valores_nulos': df.isnull().sum().to_dict(),
                'valores_unicos': df.nunique().to_dict()
            }
            
            source_col = self.config['input']['edges'][edge_type]['source_column']
            target_col = self.config['input']['edges'][edge_type]['target_column']
            
            if source_col in df.columns and target_col in df.columns:
                valid_edges = df[source_col].notna() & df[target_col].notna()
                stats['arestas_validas'] = valid_edges.sum()
                stats['arestas_invalidas'] = len(df) - valid_edges.sum()
                
                if valid_edges.sum() > 0:
                    stats['nos_origem_unicos'] = df[source_col].nunique()
                    stats['nos_destino_unicos'] = df[target_col].nunique()
                    stats['nos_conectados_unicos'] = len(set(df[source_col].dropna()) | set(df[target_col].dropna()))
            
            edge_stats[edge_type] = stats
            self.logger.info(f"✓ Analisado {edge_type}: {stats['total_registros']:,} registros")
        
        self.statistics['edges'] = edge_stats
    
    def _analisar_espacial(self):
        self.logger.info("Analisando distribuição espacial...")
        
        spatial_stats = {}
        
        for node_type, df in self.nodes_data.items():
            lat_col = None
            lon_col = None
            
            consolidated_config = self.config['input']['nodes'].get('consolidated', {})
            if 'types' in consolidated_config and node_type in consolidated_config['types']:
                type_config = consolidated_config['types'][node_type]
                lat_col = type_config.get('lat_column')
                lon_col = type_config.get('lon_column')
            else:
                node_config = self.config['input']['nodes'].get(node_type, {})
                lat_col = node_config.get('lat_column')
                lon_col = node_config.get('lon_column')
            
            if not lat_col or not lon_col:
                self.logger.warning(f"Coordinate configuration not found for {node_type}")
                continue
            
            if lat_col in df.columns and lon_col in df.columns:
                valid_coords = df[lat_col].notna() & df[lon_col].notna()
                
                if valid_coords.sum() > 0:
                    coords = df[valid_coords]
                    
                    stats = {
                        'total_pontos': len(coords),
                        'centroide_lat': coords[lat_col].mean(),
                        'centroide_lon': coords[lon_col].mean(),
                        'bbox': {
                            'min_lat': coords[lat_col].min(),
                            'max_lat': coords[lat_col].max(),
                            'min_lon': coords[lon_col].min(),
                            'max_lon': coords[lon_col].max()
                        },
                        'densidade_km2': self._calcular_densidade(coords, lat_col, lon_col)
                    }
                    
                    spatial_stats[node_type] = stats
        
        self.statistics['spatial'] = spatial_stats
    
    def _calcular_densidade(self, coords: pd.DataFrame, lat_col: str, lon_col: str) -> float:
        try:
            lat_diff = coords[lat_col].max() - coords[lat_col].min()
            lon_diff = coords[lon_col].max() - coords[lon_col].min()
            
            area_km2 = (lat_diff * 111) * (lon_diff * 111 * np.cos(np.radians(coords[lat_col].mean())))
            
            if area_km2 > 0:
                return len(coords) / area_km2
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analisar_conectividade(self):
        self.logger.info("Analisando conectividade...")
        
        connectivity_stats = {}
        
        all_node_ids = set()
        for node_type, df in self.nodes_data.items():
            id_col = None
            
            consolidated_config = self.config['input']['nodes'].get('consolidated', {})
            if 'types' in consolidated_config and node_type in consolidated_config['types']:
                id_col = consolidated_config['types'][node_type].get('id_column')
            else:
                node_config = self.config['input']['nodes'].get(node_type, {})
                id_col = node_config.get('id_column')
            
            if id_col and id_col in df.columns:
                all_node_ids.update(df[id_col].astype(str).dropna())
        
        connected_ids = set()
        for edge_type, df in self.edges_data.items():
            source_col = self.config['input']['edges'][edge_type]['source_column']
            target_col = self.config['input']['edges'][edge_type]['target_column']
            
            if source_col in df.columns and target_col in df.columns:
                connected_ids.update(df[source_col].astype(str).dropna())
                connected_ids.update(df[target_col].astype(str).dropna())
        
        isolated_ids = all_node_ids - connected_ids
        
        connectivity_stats = {
            'total_nos': len(all_node_ids),
            'nos_conectados': len(connected_ids),
            'nos_isolados': len(isolated_ids),
            'percentual_conectados': len(connected_ids) / len(all_node_ids) * 100 if all_node_ids else 0,
            'percentual_isolados': len(isolated_ids) / len(all_node_ids) * 100 if all_node_ids else 0
        }
        
        self.statistics['connectivity'] = connectivity_stats
        self.logger.info(f"✓ Conectividade: {connectivity_stats['percentual_conectados']:.1f}% conectados")
    
    def _gerar_relatorios(self):
        self.logger.info("Gerando relatórios...")
        
        json_path = self.eda_dir / 'eda_report_v3.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2, ensure_ascii=False, default=str)
        
        self._gerar_relatorio_html()
        
        self._gerar_relatorio_markdown()
        
        self.logger.info(f"✓ Reports saved to {self.eda_dir}")
    
    def _gerar_relatorio_html(self):
        html_content = f"\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>V3 Normalization EDA - ARPIA</title>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}\n                .section {{ margin: 20px 0; }}\n                .stats {{ background-color: #f9f9f9; padding: 15px; }}\n                table {{ width: 100%; border-collapse: collapse; }}\n                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}\n                th {{ background-color: #f2f2f2; }}\n            </style>\n        </head>\n        <body>\n            <div class=\"header\">\n                <h1>V3 Normalization EDA - ARPIA</h1>\n                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Overall Summary</h2>\n                <div class=\"stats\">\n                    <p><strong>Total Node Types:</strong> {len(self.statistics.get('nodes', {}))}</p>\n                    <p><strong>Total Edge Types:</strong> {len(self.statistics.get('edges', {}))}</p>\n                    <p><strong>Processing:</strong> All data (no limitations)</p>\n                </div>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Node Statistics</h2>\n                <table>\n                    <tr><th>Type</th><th>Records</th><th>Valid Coordinates</th><th>Memory (MB)</th></tr>\n        "
        
        for node_type, stats in self.statistics.get('nodes', {}).items():
            html_content += f"""
                    <tr>
                        <td>{node_type}</td>
                        <td>{stats['total_registros']:,}</td>
                        <td>{stats.get('coordenadas_validas', 'N/A')}</td>
                        <td>{stats['memoria_mb']:.1f}</td>
                    </tr>
            """
        
        html_content += '\n                </table>\n            </div>\n            \n            <div class="section">\n                <h2>Edge Statistics</h2>\n                <table>\n                    <tr><th>Type</th><th>Records</th><th>Valid Edges</th><th>Memory (MB)</th></tr>\n        '
        
        for edge_type, stats in self.statistics.get('edges', {}).items():
            html_content += f"""
                    <tr>
                        <td>{edge_type}</td>
                        <td>{stats['total_registros']:,}</td>
                        <td>{stats.get('arestas_validas', 'N/A')}</td>
                        <td>{stats['memoria_mb']:.1f}</td>
                    </tr>
            """
        
        connectivity = self.statistics.get('connectivity', {})
        html_content += f"\n                </table>\n            </div>\n            \n            <div class=\"section\">\n                <h2>Graph Connectivity</h2>\n                <div class=\"stats\">\n                    <p><strong>Total Nodes:</strong> {connectivity.get('total_nos', 0):,}</p>\n                    <p><strong>Connected Nodes:</strong> {connectivity.get('nos_conectados', 0):,} ({connectivity.get('percentual_conectados', 0):.1f}%)</p>\n                    <p><strong>Isolated Nodes:</strong> {connectivity.get('nos_isolados', 0):,} ({connectivity.get('percentual_isolados', 0):.1f}%)</p>\n                </div>\n            </div>\n            \n        </body>\n        </html>\n        "
        
        html_path = self.eda_dir / 'eda_report_v3.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _gerar_relatorio_markdown(self):
        md_content = f"# V3 Normalization EDA - ARPIA\n\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Overall Summary\n\n- **Version:** 3.0\n- **Processing:** All data (no limitations)\n- **Node Types:** {len(self.statistics.get('nodes', {}))}\n- **Edge Types:** {len(self.statistics.get('edges', {}))}\n\n## Node Statistics\n\n| Type | Records | Valid Coordinates | Memory (MB) |\n|------|-----------|-------------------|--------------|\n"
        
        for node_type, stats in self.statistics.get('nodes', {}).items():
            md_content += f"| {node_type} | {stats['total_registros']:,} | {stats.get('coordenadas_validas', 'N/A')} | {stats['memoria_mb']:.1f} |\n"
        
        md_content += '\n## Edge Statistics\n\n| Type | Records | Valid Edges | Memory (MB) |\n|------|-----------|---------------|--------------|\n'
        
        for edge_type, stats in self.statistics.get('edges', {}).items():
            md_content += f"| {edge_type} | {stats['total_registros']:,} | {stats.get('arestas_validas', 'N/A')} | {stats['memoria_mb']:.1f} |\n"
        
        connectivity = self.statistics.get('connectivity', {})
        md_content += f"\n## Graph Connectivity\n\n- **Total Nodes:** {connectivity.get('total_nos', 0):,}\n- **Connected Nodes:** {connectivity.get('nos_conectados', 0):,} ({connectivity.get('percentual_conectados', 0):.1f}%)\n- **Isolated Nodes:** {connectivity.get('nos_isolados', 0):,} ({connectivity.get('percentual_isolados', 0):.1f}%)\n\n## V3 Notes\n\nThis version processes ALL available data without applying limitations or restrictive filters, \npreserving the full integrity of the dataset to maximize the quality of GNN model training.\n"
        
        md_path = self.eda_dir / 'eda_report_v3.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content) 
