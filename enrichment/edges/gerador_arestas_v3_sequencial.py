# -*- coding: utf-8 -*-
"""Sequential V3 edge generator: connection, non-connection, same-type proximity and intra-triangle edges."""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import json
import warnings
import gc
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
import hashlib
warnings.filterwarnings('ignore')

NUMBA_AVAILABLE = False
print(f"🔄 Numba DISABLED to avoid conflicts - using plain NumPy")

def haversine_vectorized_seq(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    lat1_rad = lat1_arr * 0.017453292519943295
    lat2_rad = lat2_arr * 0.017453292519943295
    dlat = (lat2_arr - lat1_arr) * 0.017453292519943295
    dlon = (lon2_arr - lon1_arr) * 0.017453292519943295
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371.0 * c
    
    return distances.astype(np.float32)

def bearing_vectorized_seq(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    lat1_rad = lat1_arr * 0.017453292519943295
    lon1_rad = lon1_arr * 0.017453292519943295
    lat2_rad = lat2_arr * 0.017453292519943295
    lon2_rad = lon2_arr * 0.017453292519943295
    
    dlon = lon2_rad - lon1_rad
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing_rad = np.arctan2(x, y)
    bearing_deg = bearing_rad * 57.29577951308232
    bearings = (bearing_deg + 360.0) % 360.0
    
    return bearings.astype(np.float32)

def haversine_single(lat1, lon1, lat2, lon2):
    lat1_rad = lat1 * 0.017453292519943295
    lat2_rad = lat2 * 0.017453292519943295
    dlat = (lat2 - lat1) * 0.017453292519943295
    dlon = (lon2 - lon1) * 0.017453292519943295
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

class GeradorArestasV3Sequencial:
    
    def __init__(self, config_path: str = "arestas_config_sequencial.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.stats = {
            'edges_created': {},
            'processing_time': {},
            'data_loaded': {},
            'quality_metrics': {},
            'filters_applied': {},
            'alerts_generated': [],
            'hardware_info': {}
        }
        
        self.use_numba = False
        self.chunk_size = self.config.get('hardware', {}).get('memory_chunk_size', 25000)
        
        self.quality_filters = self.config.get('processing', {}).get('quality_filters', {})
        self.quality_monitoring = self.config.get('quality_monitoring', {})
        
        self.logger.info(f"🔄 Edge Generator V3 SEQUENTIAL OPTIMIZED initialized")
        self.logger.info(f"   Versão: {self.config.get('metadata', {}).get('version', '3.1')}")
        self.logger.info(f"   Melhorias: Filtros de qualidade, clustering, monitoramento")
        self.logger.info(f"   Chunk size: {self.chunk_size:,}")
        
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error while loading configuration: {e}")
            raise
            
    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        log_file = log_config.get('log_file', 'arestas_pipeline_sequencial.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('log_format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def _apply_quality_filters(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        self.logger.info(f"⚡ Skipping filters for {edge_type} - data already processed in the triangulation")
        return df
        
        original_count = len(df)
        
        if 'distancia_m' in df.columns:
            min_dist = self.quality_filters.get('min_distance_threshold', 0.001) * 1000
            df = df[df['distancia_m'] >= min_dist]
            filtered_count = original_count - len(df)
            self.stats['filters_applied'][f'{edge_type}_zero_distance'] = filtered_count
            self.logger.info(f"   🚫 Removed {filtered_count:,} edges with distance < {min_dist}m")
        
        if 'distancia_km' in df.columns:
            min_dist_km = self.quality_filters.get('min_distance_threshold', 0.001)
            df = df[df['distancia_km'] >= min_dist_km]
            filtered_count = original_count - len(df)
            self.stats['filters_applied'][f'{edge_type}_zero_distance_km'] = filtered_count
            self.logger.info(f"   🚫 Removed {filtered_count:,} edges with distance < {min_dist_km}km")
        
        if self.quality_filters.get('remove_outliers', True):
            df = self._remove_outliers(df, edge_type)
        
        if self.quality_filters.get('remove_duplicates', True):
            df = self._remove_duplicates(df, edge_type)
        
        final_count = len(df)
        reduction_percent = (original_count - final_count) / original_count * 100
        self.logger.info(f"   ✅ Filtros aplicados: {original_count:,} → {final_count:,} (-{reduction_percent:.1f}%)")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        if 'distancia_m' in df.columns:
            distance_col = 'distancia_m'
        elif 'distancia_km' in df.columns:
            distance_col = 'distancia_km'
        else:
            return df
            
        std_threshold = self.quality_filters.get('outlier_std_threshold', 3.0)
        mean_dist = df[distance_col].mean()
        std_dist = df[distance_col].std()
        
        lower_bound = mean_dist - std_threshold * std_dist
        upper_bound = mean_dist + std_threshold * std_dist
        
        original_count = len(df)
        df = df[(df[distance_col] >= lower_bound) & (df[distance_col] <= upper_bound)]
        outliers_removed = original_count - len(df)
        
        self.stats['filters_applied'][f'{edge_type}_outliers'] = outliers_removed
        self.logger.info(f"   🎯 Removidos {outliers_removed:,} outliers ({std_threshold}σ)")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        if len(df) == 0:
            return df
            
        original_count = len(df)
        
        df = df.drop_duplicates()
        exact_duplicates = original_count - len(df)
        
        if edge_type not in ['conexao_pea_erb', 'nao_conexao']:
            if 'distancia_m' in df.columns or 'distancia_km' in df.columns:
                tolerance = self.quality_filters.get('duplicate_tolerance', 0.01)
                df = self._remove_geographic_duplicates(df, tolerance)
        
        if 'source_id' in df.columns and 'target_id' in df.columns:
            df = df.drop_duplicates(subset=['source_id', 'target_id'])
        
        final_count = len(df)
        total_duplicates = original_count - final_count
        
        self.stats['filters_applied'][f'{edge_type}_duplicates'] = total_duplicates
        self.logger.info(f"   🔄 Removidas {total_duplicates:,} duplicatas ({exact_duplicates:,} exatas)")
        
        return df
    
    def _remove_geographic_duplicates(self, df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
        if len(df) < 2:
            return df
            
        if 'source_id' in df.columns and 'target_id' in df.columns:
            if 'distancia_m' in df.columns:
                df['geo_hash'] = (df['source_id'].astype(str) + '_' + 
                                 df['target_id'].astype(str) + '_' + 
                                 df['distancia_m'].round(0).astype(str))
            elif 'distancia_km' in df.columns:
                df['geo_hash'] = (df['source_id'].astype(str) + '_' + 
                                 df['target_id'].astype(str) + '_' + 
                                 df['distancia_km'].round(3).astype(str))
            else:
                return df
                
            df = df.drop_duplicates(subset=['geo_hash'])
            df = df.drop(columns=['geo_hash'])
        
        return df
    
    def _apply_connection_limits(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        if edge_type not in ['nao_conexao', 'pea_pea']:
            return df
            
        max_connections = self.quality_filters.get('max_connections_per_pea', 50)
        if max_connections is None:
            return df
            
        self.logger.info(f"🔗 Applying a limit of {max_connections} connections per PEA")
        
        original_count = len(df)
        
        source_col = None
        if 'source_id' in df.columns:
            source_col = 'source_id'
        elif 'id_pea' in df.columns:
            source_col = 'id_pea'
        
        if source_col is None:
            return df
        
        limited_dfs = []
        for pea_id in df[source_col].unique():
            pea_connections = df[df[source_col] == pea_id]
            if len(pea_connections) > max_connections:
                if 'distancia_m' in pea_connections.columns:
                    pea_connections = pea_connections.nsmallest(max_connections, 'distancia_m')
                elif 'distancia_km' in pea_connections.columns:
                    pea_connections = pea_connections.nsmallest(max_connections, 'distancia_km')
                else:
                    pea_connections = pea_connections.head(max_connections)
            
            limited_dfs.append(pea_connections)
        
        df = pd.concat(limited_dfs, ignore_index=True)
        
        final_count = len(df)
        reduction = original_count - final_count
        
        self.stats['filters_applied'][f'{edge_type}_connection_limit'] = reduction
        self.logger.info(f"   ✂️ Connections limited: {original_count:,} → {final_count:,} (-{reduction:,})")
        
        return df
    
    def _apply_geographic_clustering(self, df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
        if edge_type != 'pea_pea':
            return df
            
        clustering_config = self.config.get('processing', {}).get('geographic_clustering', {})
        if not clustering_config.get('enable_clustering', False):
            return df
        
        self.logger.info(f"🌍 Applying geographic clustering for PEA-PEA")
        
        threshold = clustering_config.get('cluster_distance_threshold', 0.1)
        
        clustered_dfs = []
        processed_pairs = set()
        
        for _, row in df.iterrows():
            pair_key = f"{row['source_id']}-{row['target_id']}"
            if pair_key in processed_pairs:
                continue
                
            similar_mask = (df['distancia_km'] >= row['distancia_km'] - threshold) & \
                          (df['distancia_km'] <= row['distancia_km'] + threshold)
            
            similar_edges = df[similar_mask]
            
            if len(similar_edges) > 1:
                representative = similar_edges.iloc[0:1]
                clustered_dfs.append(representative)
                
                for _, similar_row in similar_edges.iterrows():
                    processed_pairs.add(f"{similar_row['source_id']}-{similar_row['target_id']}")
            else:
                clustered_dfs.append(similar_edges)
                processed_pairs.add(pair_key)
        
        if clustered_dfs:
            df = pd.concat(clustered_dfs, ignore_index=True)
            
        self.logger.info(f"   🎯 Clustering aplicado: {len(clustered_dfs)} clusters formados")
        
        return df
    
    def _generate_quality_metrics(self, edges_dict: Dict[str, pd.DataFrame]):
        self.logger.info('⚡ Skipping quality metrics - only creating edges')
        return
        
        for edge_type, df in edges_dict.items():
            if len(df) == 0:
                continue
                
            metrics = {}
            
            if 'distancia_m' in df.columns:
                metrics['distance_stats'] = {
                    'mean': float(df['distancia_m'].mean()),
                    'std': float(df['distancia_m'].std()),
                    'min': float(df['distancia_m'].min()),
                    'max': float(df['distancia_m'].max()),
                    'zero_count': int((df['distancia_m'] == 0).sum()),
                    'zero_percentage': float((df['distancia_m'] == 0).mean() * 100)
                }
            
            if 'distancia_km' in df.columns:
                metrics['distance_km_stats'] = {
                    'mean': float(df['distancia_km'].mean()),
                    'std': float(df['distancia_km'].std()),
                    'min': float(df['distancia_km'].min()),
                    'max': float(df['distancia_km'].max()),
                    'zero_count': int((df['distancia_km'] == 0).sum()),
                    'zero_percentage': float((df['distancia_km'] == 0).mean() * 100)
                }
            
            if 'source_id' in df.columns and 'target_id' in df.columns:
                source_target_pairs = df[['source_id', 'target_id']].drop_duplicates()
                duplicate_rate = 1 - (len(source_target_pairs) / len(df))
                metrics['duplicate_rate'] = float(duplicate_rate)
            
            if 'source_id' in df.columns:
                source_counts = df['source_id'].value_counts()
                metrics['connectivity_stats'] = {
                    'unique_sources': len(source_counts),
                    'avg_connections_per_source': float(source_counts.mean()),
                    'max_connections_per_source': int(source_counts.max()),
                    'min_connections_per_source': int(source_counts.min())
                }
            
            self.stats['quality_metrics'][edge_type] = metrics
    
    def _check_quality_alerts(self, edges_dict: Dict[str, pd.DataFrame]):
        self.logger.info('⚡ Skipping quality alerts - only creating edges')
        return
        
        alerts = []
        metrics_config = self.quality_monitoring.get('metrics', {})
        
        for edge_type, df in edges_dict.items():
            if len(df) == 0:
                continue
                
            edge_metrics = self.stats['quality_metrics'].get(edge_type, {})
            
            duplicate_threshold = metrics_config.get('duplicate_rate_threshold', 0.1)
            duplicate_rate = edge_metrics.get('duplicate_rate', 0)
            if duplicate_rate > duplicate_threshold:
                alerts.append({
                    'type': 'HIGH_DUPLICATE_RATE',
                    'edge_type': edge_type,
                    'value': duplicate_rate,
                    'threshold': duplicate_threshold,
                    'message': f"High duplicate rate in {edge_type}: {duplicate_rate:.1%}"
                })
            
            zero_threshold = metrics_config.get('zero_distance_threshold', 0.05)
            distance_stats = edge_metrics.get('distance_stats', {})
            zero_percentage = distance_stats.get('zero_percentage', 0) / 100
            if zero_percentage > zero_threshold:
                alerts.append({
                    'type': 'HIGH_ZERO_DISTANCE',
                    'edge_type': edge_type,
                    'value': zero_percentage,
                    'threshold': zero_threshold,
                    'message': f"Too many zero distances in {edge_type}: {zero_percentage:.1%}"
                })
        
        self.stats['alerts_generated'] = alerts
        
        if alerts:
            self.logger.warning(f"🚨 {len(alerts)} alertas de qualidade gerados")
            for alert in alerts:
                self.logger.warning(f"   ⚠️ {alert['message']}")
        else:
            self.logger.info("✅ Nenhum alerta de qualidade")
    
    def _save_quality_reports(self):
        if not self.quality_monitoring.get('enable_monitoring', False):
            return
            
        alerts_config = self.quality_monitoring.get('alerts', {})
        if alerts_config.get('enable_alerts', False):
            alert_file = alerts_config.get('alert_file', 'quality_alerts.json')
            try:
                os.makedirs(os.path.dirname(alert_file), exist_ok=True)
                with open(alert_file, 'w', encoding='utf-8') as f:
                    json.dump(self.stats['alerts_generated'], f, indent=2, ensure_ascii=False)
                self.logger.info(f"📋 Alerts saved to: {alert_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error while saving alerts: {e}")
        
        reports_config = self.quality_monitoring.get('quality_reports', {})
        if reports_config.get('enable_reports', False):
            report_file = reports_config.get('report_file', 'quality_report.json')
            try:
                os.makedirs(os.path.dirname(report_file), exist_ok=True)
                quality_report = {
                    'timestamp': datetime.now().isoformat(),
                    'metrics': self.stats['quality_metrics'],
                    'filters_applied': self.stats['filters_applied'],
                    'alerts': self.stats['alerts_generated'],
                    'summary': {
                        'total_edges': sum(len(df) for df in self.stats['edges_created'].values()),
                        'total_alerts': len(self.stats['alerts_generated']),
                        'filters_applied': len(self.stats['filters_applied'])
                    }
                }
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(quality_report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"📊 Quality report saved to: {report_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error while saving report: {e}")
    
    def _haversine_distance_fast(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
        
    def _corrigir_ids_malformados(self, df: pd.DataFrame, colunas_id: List[str]) -> pd.DataFrame:
        df_corrigido = df.copy()
        total_corrigidos = 0
        
        for coluna in colunas_id:
            if coluna in df_corrigido.columns:
                malformados_antes = df_corrigido[coluna].astype(str).str.contains('_PEA_|_ERB_|_CPS_|_CTC_|_CTNC_', regex=True, na=False).sum()
                
                df_corrigido[coluna] = df_corrigido[coluna].astype(str).apply(self._corrigir_id_individual)
                
                if malformados_antes > 0:
                    total_corrigidos += malformados_antes
                    self.logger.info(f"   🔧 {coluna}: {malformados_antes:,} IDs malformados corrigidos")
        
        if total_corrigidos > 0:
            self.logger.info(f"✅ Total corrected IDs: {total_corrigidos:,}")
        
        return df_corrigido
    
    def _corrigir_id_individual(self, node_id: str) -> str:
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
        
        return node_id

    def load_data(self) -> Dict[str, pd.DataFrame]:
        self.logger.info('📂 Loading input data with quality filters...')
        
        data = {}
        input_files = self.config['input_files']
        
        triangulacao_path = input_files['triangulacao']
        if os.path.exists(triangulacao_path):
            self.logger.info("   📊 Carregando triangulação...")
            
            required_cols = [
                'pea_id', 'pea_lat', 'pea_lon', 
                'erb_id', 'lat_erb', 'lon_erb',
                'cps_id', 'lat_centroide', 'lon_centroide',
                'triangulation_id', 'triangulation_type',
                'triangulation_centroid_lat', 'triangulation_centroid_lon',
                'operadora', 'fator_punitivo'
            ]
            
            data['triangulacao'] = pd.read_csv(triangulacao_path, usecols=required_cols, dtype={
                'pea_id': 'string',
                'erb_id': 'string', 
                'cps_id': 'string',
                'triangulation_id': 'string',
                'triangulation_type': 'string',
                'operadora': 'string',
                'pea_lat': 'float32',
                'pea_lon': 'float32',
                'lat_erb': 'float32',
                'lon_erb': 'float32',
                'lat_centroide': 'float32',
                'lon_centroide': 'float32',
                'triangulation_centroid_lat': 'float32',
                'triangulation_centroid_lon': 'float32',
                'fator_punitivo': 'float32'
            })
            
            self.logger.info('   🔧 Mapping columns to the standard format...')
            data['triangulacao'] = data['triangulacao'].rename(columns={
                'lat_erb': 'erb_lat',
                'lon_erb': 'erb_lon',
                'lat_centroide': 'cps_lat',
                'lon_centroide': 'cps_lon',
                'triangulation_centroid_lat': 'centroid_lat',
                'triangulation_centroid_lon': 'centroid_lon'
            })
            
            self.logger.info('   ⚡ Skipping ID correction - already done in the triangulation')
            
            data['triangulacao'] = self._validate_coordinates(data['triangulacao'])
            
            self.logger.info(f"   ✅ Triangulation: {len(data['triangulacao']):,} records")
            self.stats['data_loaded']['triangulacao'] = len(data['triangulacao'])
        
        conexoes_path = input_files['conexoes_pea_erb']
        if os.path.exists(conexoes_path):
            data['conexoes_pea_erb'] = pd.read_csv(conexoes_path, dtype={
                'pea_id': 'string',
                'erb_id': 'string',
                'pea_lat': 'float32',
                'pea_lon': 'float32',
                'Latitude RE': 'float32',
                'Longitude RE': 'float32',
                'erb_operadora': 'string',
                'Duração': 'float32'
            })
            
            data['conexoes_pea_erb'] = data['conexoes_pea_erb'].rename(columns={
                'Latitude RE': 'erb_lat',
                'Longitude RE': 'erb_lon',
                'Duração': 'duracao_conexao_gprs'
            })
            
            self.logger.info('   ⚡ Skipping ID correction in the connections - already done in the triangulation')
            
            data['conexoes_pea_erb'] = self._validate_connections(data['conexoes_pea_erb'])
            
            self.logger.info(f"   ✅ PEA-ERB connections: {len(data['conexoes_pea_erb']):,} records")
            self.stats['data_loaded']['conexoes_pea_erb'] = len(data['conexoes_pea_erb'])
        
        triangulacao_df = data['triangulacao']
        data['ctcs'] = triangulacao_df[triangulacao_df['triangulation_type'] == 'CTC'].copy()
        data['ctncs'] = triangulacao_df[triangulacao_df['triangulation_type'] == 'CTNC'].copy()
        
        max_ctc_limit = self.config.get('processing', {}).get('max_edges', {}).get('ctc_ctc', 10000000)
        max_ctnc_limit = self.config.get('processing', {}).get('max_edges', {}).get('ctnc_ctnc', 20000000)
        
        if len(data['ctcs']) > max_ctc_limit:
            data['ctcs'] = data['ctcs'].sample(n=max_ctc_limit, random_state=42)
            self.logger.warning(f"   ⚠️ CTCs limitados a {max_ctc_limit:,} registros")
        
        if len(data['ctncs']) > max_ctnc_limit:
            data['ctncs'] = data['ctncs'].sample(n=max_ctnc_limit, random_state=42)
            self.logger.warning(f"   ⚠️ CTNCs limitados a {max_ctnc_limit:,} registros")
        
        self.logger.info(f"   ✅ CTCs: {len(data['ctcs']):,} registros")
        self.logger.info(f"   ✅ CTNCs: {len(data['ctncs']):,} registros")
        
        self.stats['data_loaded']['ctcs'] = len(data['ctcs'])
        self.stats['data_loaded']['ctncs'] = len(data['ctncs'])
        
        del triangulacao_df
        gc.collect()
        
        return data
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        original_count = len(df)
        
        df = df.dropna(subset=['pea_lat', 'pea_lon', 'erb_lat', 'erb_lon'])
        
        df = df[
            (df['pea_lat'] >= -35) & (df['pea_lat'] <= 10) &
            (df['pea_lon'] >= -75) & (df['pea_lon'] <= -30) &
            (df['erb_lat'] >= -35) & (df['erb_lat'] <= 10) &
            (df['erb_lon'] >= -75) & (df['erb_lon'] <= -30)
        ]
        
        invalid_coords = original_count - len(df)
        if invalid_coords > 0:
            self.logger.warning(f"   ⚠️ Removed {invalid_coords:,} invalid coordinates")
        
        return df
    
    def _validate_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        original_count = len(df)
        
        df = df.dropna(subset=['pea_id', 'erb_id'])
        df = df[df['pea_id'].str.len() > 0]
        df = df[df['erb_id'].str.len() > 0]
        
        if 'duracao_conexao_gprs' in df.columns:
            df = df[df['duracao_conexao_gprs'] >= 0]
        
        invalid_connections = original_count - len(df)
        if invalid_connections > 0:
            self.logger.warning(f"   ⚠️ Removed {invalid_connections:,} invalid connections")
        
        return df
        
    def create_edges_conexao_pea_erb(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info('🔗 Creating PEA-ERB connection edges...')
        
        start_time = datetime.now()
        conexoes_df = data['conexoes_pea_erb'].copy()
        
        if len(conexoes_df) == 0:
            return pd.DataFrame()
        
        edge_config = self.config.get('edge_config', {}).get('conexao_pea_erb', {})
        min_dist = edge_config.get('min_distance_m', 1.0)
        max_dist = edge_config.get('max_distance_m', 10000.0)
        
        if len(conexoes_df) > 1000:
            self.logger.info('   ⚡ Using NumPy vectorization...')
            
            distances = haversine_vectorized_seq(
                conexoes_df['pea_lat'].values.astype(np.float32),
                conexoes_df['pea_lon'].values.astype(np.float32),
                conexoes_df['erb_lat'].values.astype(np.float32),
                conexoes_df['erb_lon'].values.astype(np.float32)
            ) * 1000
            
            azimutes = bearing_vectorized_seq(
                conexoes_df['pea_lat'].values.astype(np.float32),
                conexoes_df['pea_lon'].values.astype(np.float32),
                conexoes_df['erb_lat'].values.astype(np.float32),
                conexoes_df['erb_lon'].values.astype(np.float32)
            )
            
            df_edges = pd.DataFrame({
                'source_id': conexoes_df['pea_id'],
                'target_id': conexoes_df['erb_id'],
                'id_pea': conexoes_df['pea_id'],
                'id_erb': conexoes_df['erb_id'],
                'distancia_m': distances,
                'azimute_pea_erb': azimutes,
                'operadora_pea': conexoes_df.get('operadora', conexoes_df.get('erb_operadora', '')),
                'operadora_erb': conexoes_df.get('erb_operadora', conexoes_df.get('operadora', '')),
                'tecnologia': conexoes_df.get('tecnologia', ''),
                'duracao_conexao_gprs': conexoes_df.get('duracao_conexao_gprs', 0)
            })
            
            df_edges = df_edges[
                (df_edges['distancia_m'] >= min_dist) & 
                (df_edges['distancia_m'] <= max_dist)
            ]
            
        else:
            self.logger.info('   🔄 Using the manual sequential method...')
            edges_data = []
            
            for _, row in conexoes_df.iterrows():
                distance_m = self._haversine_distance_fast(
                    row['pea_lat'], row['pea_lon'],
                    row['erb_lat'], row['erb_lon']
                ) * 1000
                
                if min_dist <= distance_m <= max_dist:
                    edge = {
                        'source_id': row['pea_id'],
                        'target_id': row['erb_id'],
                        'id_pea': row['pea_id'],
                        'id_erb': row['erb_id'],
                        'distancia_m': distance_m,
                        'azimute_pea_erb': 0.0,
                        'operadora_pea': row.get('operadora', row.get('erb_operadora', '')),
                        'operadora_erb': row.get('erb_operadora', row.get('operadora', '')),
                        'tecnologia': row.get('tecnologia', ''),
                        'duracao_conexao_gprs': row.get('duracao_conexao_gprs', 0)
                    }
                    edges_data.append(edge)
            
            df_edges = pd.DataFrame(edges_data)
        
        df_edges = self._apply_quality_filters(df_edges, 'conexao_pea_erb')
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.stats['edges_created']['conexao_pea_erb'] = len(df_edges)
        self.stats['processing_time']['conexao_pea_erb'] = elapsed_time
        
        self.logger.info(f"   ✅ {len(df_edges):,} edges created in {elapsed_time:.2f}s")
        
        return df_edges
        
    def create_edges_nao_conexao(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info('🚫 Creating OPTIMIZED non-connection edges...')
        
        start_time = datetime.now()
        ctncs_df = data['ctncs'].copy()
        
        if len(ctncs_df) == 0:
            return pd.DataFrame()
        
        edge_config = self.config.get('edge_config', {}).get('nao_conexao', {})
        max_per_pea = edge_config.get('max_per_pea', 100)
        
        self.logger.info(f"   🎯 Limit: {max_per_pea} non-connections per PEA")
        
        chunk_size = self.chunk_size // 2
        total_chunks = len(ctncs_df) // chunk_size + 1
        
        all_edges = []
        
        for i in range(0, len(ctncs_df), chunk_size):
            chunk_num = i // chunk_size + 1
            if chunk_num % 5 == 0:
                self.logger.info(f"      Processando chunk {chunk_num}/{total_chunks}")
            
            chunk = ctncs_df.iloc[i:i+chunk_size]
            
            chunk_edges = []
            for pea_id in chunk['pea_id'].unique():
                pea_data = chunk[chunk['pea_id'] == pea_id]
                
                if len(pea_data) > max_per_pea:
                    pea_data = pea_data.nsmallest(max_per_pea, 'fator_punitivo')
                
                if len(pea_data) > 50:
                    distances = haversine_vectorized_seq(
                        pea_data['pea_lat'].values.astype(np.float32),
                        pea_data['pea_lon'].values.astype(np.float32),
                        pea_data['erb_lat'].values.astype(np.float32),
                        pea_data['erb_lon'].values.astype(np.float32)
                    ) * 1000
                    
                    azimutes = bearing_vectorized_seq(
                        pea_data['pea_lat'].values.astype(np.float32),
                        pea_data['pea_lon'].values.astype(np.float32),
                        pea_data['erb_lat'].values.astype(np.float32),
                        pea_data['erb_lon'].values.astype(np.float32)
                    )
                    
                    operadora_pea = pea_data.get('operadora', '').astype(str).fillna('')
                    operadora_erb = pea_data.get('operadora', '').astype(str).fillna('')
                    mesma_operadora = (operadora_pea == operadora_erb).astype(int)
                    
                    pea_edges = pd.DataFrame({
                        'source_id': pea_data['pea_id'],
                        'target_id': pea_data['erb_id'],
                        'id_pea': pea_data['pea_id'],
                        'id_erb_nc': pea_data['erb_id'],
                        'distancia_m': distances,
                        'azimute_pea_erb_nc': azimutes,
                        'operadora_pea': operadora_pea,
                        'operadora_erb_nc': operadora_erb,
                        'fator_punitivo': pea_data.get('fator_punitivo', 10.0),
                        'mesma_operadora': mesma_operadora
                    })
                    
                    chunk_edges.append(pea_edges)
                else:
                    for _, row in pea_data.iterrows():
                        distance_m = self._haversine_distance_fast(
                            row['pea_lat'], row['pea_lon'],
                            row['erb_lat'], row['erb_lon']
                        ) * 1000
                        
                        edge = {
                            'source_id': row['pea_id'],
                            'target_id': row['erb_id'],
                            'id_pea': row['pea_id'],
                            'id_erb_nc': row['erb_id'],
                            'distancia_m': distance_m,
                            'azimute_pea_erb_nc': 0.0,
                            'operadora_pea': row.get('operadora', ''),
                            'operadora_erb_nc': row.get('operadora', ''),
                            'fator_punitivo': row.get('fator_punitivo', 10.0),
                            'mesma_operadora': 1 if row.get('operadora', '') == row.get('operadora', '') else 0
                        }
                        chunk_edges.append(pd.DataFrame([edge]))
            
            if chunk_edges:
                chunk_df = pd.concat(chunk_edges, ignore_index=True)
                all_edges.append(chunk_df)
            
            if chunk_num % 10 == 0:
                gc.collect()
        
        if all_edges:
            df_edges = pd.concat(all_edges, ignore_index=True)
        else:
            df_edges = pd.DataFrame()
        
        df_edges = self._apply_quality_filters(df_edges, 'nao_conexao')
        df_edges = self._apply_connection_limits(df_edges, 'nao_conexao')
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.stats['edges_created']['nao_conexao'] = len(df_edges)
        self.stats['processing_time']['nao_conexao'] = elapsed_time
        
        self.logger.info(f"   ✅ {len(df_edges):,} OPTIMIZED edges created in {elapsed_time:.2f}s")
        
        return df_edges
        
    def create_same_type_edges_sequencial(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.logger.info('🔄 Creating OPTIMIZED edges between nodes of the same type...')
        
        start_time = datetime.now()
        same_type_edges = {}
        
        triangulacao_df = data['triangulacao']
        
        node_types_data = self._prepare_node_types_otimizado(triangulacao_df, data)
        
        for node_type, (nodes_df, edge_type) in node_types_data.items():
            if len(nodes_df) > 0:
                self.logger.info(f"   🔄 Processing {node_type.upper()}: {len(nodes_df):,} nodes")
                edges_df = self._create_edges_sequencial_otimizado(nodes_df, node_type, edge_type)
                
                edges_df = self._apply_quality_filters(edges_df, edge_type)
                
                if node_type == 'pea':
                    edges_df = self._apply_geographic_clustering(edges_df, 'pea_pea')
                    edges_df = self._apply_connection_limits(edges_df, 'pea_pea')
                
                same_type_edges[edge_type] = edges_df
                self.stats['edges_created'][edge_type] = len(edges_df)
            else:
                self.logger.warning(f"   ⚠️ No {node_type.upper()} node found")
                same_type_edges[edge_type] = pd.DataFrame()
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['same_type_edges'] = elapsed_time
        
        self.logger.info(f"   ⏱️ Tempo total sequencial otimizado: {elapsed_time:.2f}s")
        
        return same_type_edges
    
    def _prepare_node_types_otimizado(self, triangulacao_df: pd.DataFrame, data: Dict) -> Dict:
        
        peas_df = triangulacao_df[['pea_id', 'pea_lat', 'pea_lon', 'operadora']].drop_duplicates(subset=['pea_id'])
        peas_df = peas_df.rename(columns={'pea_id': 'node_id', 'pea_lat': 'lat', 'pea_lon': 'lon'})
        
        erbs_df = triangulacao_df[['erb_id', 'erb_lat', 'erb_lon', 'operadora']].drop_duplicates(subset=['erb_id'])
        erbs_df = erbs_df.rename(columns={'erb_id': 'node_id', 'erb_lat': 'lat', 'erb_lon': 'lon'})
        
        cps_df = triangulacao_df[['cps_id', 'cps_lat', 'cps_lon', 'operadora']].drop_duplicates(subset=['cps_id'])
        cps_df = cps_df.rename(columns={'cps_id': 'node_id', 'cps_lat': 'lat', 'cps_lon': 'lon'})
        
        ctcs_df = data['ctcs'][['triangulation_id', 'centroid_lat', 'centroid_lon', 'operadora']].copy()
        ctcs_df = ctcs_df.rename(columns={'triangulation_id': 'node_id', 'centroid_lat': 'lat', 'centroid_lon': 'lon'})
        
        max_ctc_nodes = self.config.get('processing', {}).get('max_edges', {}).get('ctc_ctc', 10000000)
        expected_ctc_combinations = len(ctcs_df) * (len(ctcs_df) - 1) // 2
        if expected_ctc_combinations > max_ctc_nodes:
            max_nodes = int(np.sqrt(max_ctc_nodes * 2))
            ctcs_df = ctcs_df.sample(n=min(max_nodes, len(ctcs_df)), random_state=42)
            self.logger.warning(f"      ⚠️ CTCs limited to {len(ctcs_df):,} nodes")
        
        ctncs_df = data['ctncs'][['triangulation_id', 'centroid_lat', 'centroid_lon', 'operadora']].copy()
        ctncs_df = ctncs_df.rename(columns={'triangulation_id': 'node_id', 'centroid_lat': 'lat', 'centroid_lon': 'lon'})
        
        max_ctnc_nodes = self.config.get('processing', {}).get('max_edges', {}).get('ctnc_ctnc', 20000000)
        expected_ctnc_combinations = len(ctncs_df) * (len(ctncs_df) - 1) // 2
        if expected_ctnc_combinations > max_ctnc_nodes:
            max_nodes = int(np.sqrt(max_ctnc_nodes * 2))
            ctncs_df = ctncs_df.sample(n=min(max_nodes, len(ctncs_df)), random_state=42)
            self.logger.warning(f"      ⚠️ CTNCs limited to {len(ctncs_df):,} nodes")
        
        self.logger.info(f"      🔄 Preparing OPTIMIZED node types:")
        self.logger.info(f"         PEAs: {len(peas_df):,}")
        self.logger.info(f"         ERBs: {len(erbs_df):,}")
        self.logger.info(f"         CPS: {len(cps_df):,}")
        self.logger.info(f"         CTCs: {len(ctcs_df):,}")
        self.logger.info(f"         CTNCs: {len(ctncs_df):,}")
        
        return {
            'pea': (peas_df, 'edges_pea_pea'),
            'erb': (erbs_df, 'edges_erb_erb'),
            'cps': (cps_df, 'edges_cps_cps'),
            'ctc': (ctcs_df, 'edges_ctc_ctc'),
            'ctnc': (ctncs_df, 'edges_ctnc_ctnc')
        }
        
    def _create_edges_sequencial_otimizado(self, nodes_df: pd.DataFrame, node_type: str, edge_type: str) -> pd.DataFrame:
        max_distance = self.config.get('processing', {}).get('max_distances', {}).get(node_type, 2.0)
        max_edges = self.config.get('processing', {}).get('max_edges', {}).get(edge_type, 5000000)
        
        self.logger.info(f"      🔄 Processando {node_type} sequencialmente...")
        self.logger.info(f"         Distância máxima: {max_distance}km")
        self.logger.info(f"         Limit on the edges: {max_edges:,}")
        
        edges_data = []
        nodes_array = nodes_df[['node_id', 'lat', 'lon', 'operadora']].fillna('').to_numpy()
        
        total_combinations = len(nodes_array) * (len(nodes_array) - 1) // 2
        self.logger.info(f"         Combinações totais: {total_combinations:,}")
        
        if total_combinations > max_edges:
            sample_rate = max_edges / total_combinations
            self.logger.info(f"         Amostragem: {sample_rate:.1%}")
        else:
            sample_rate = 1.0
        
        processed = 0
        edges_created = 0
        
        for i, j in combinations(range(len(nodes_array)), 2):
            if sample_rate < 1.0 and np.random.random() > sample_rate:
                continue
                
            node1, node2 = nodes_array[i], nodes_array[j]
            
            lat1, lon1 = float(node1[1]), float(node1[2])
            lat2, lon2 = float(node2[1]), float(node2[2])
            
            distance_km = self._haversine_distance_fast(lat1, lon1, lat2, lon2)
            
            if distance_km <= max_distance:
                operadora1 = str(node1[3])
                operadora2 = str(node2[3])
                mesma_operadora = 1 if operadora1 == operadora2 else 0
                
                base_edge = {
                    'tipo_aresta': edge_type,
                    'distancia_km': distance_km,
                    'mesma_operadora': mesma_operadora
                }
                
                edge1 = base_edge.copy()
                edge1['source_id'] = str(node1[0])
                edge1['target_id'] = str(node2[0])
                edges_data.append(edge1)
                
                edge2 = base_edge.copy()
                edge2['source_id'] = str(node2[0])
                edge2['target_id'] = str(node1[0])
                edges_data.append(edge2)
                
                edges_created += 2
                
                if edges_created >= max_edges:
                    self.logger.warning(f"         ⚠️ Limit of {max_edges:,} edges reached")
                    break
            
            processed += 1
            
            if processed % 500000 == 0:
                progress = processed / total_combinations * 100
                self.logger.info(f"         Processado: {processed:,}/{total_combinations:,} ({progress:.1f}%)")
                
                if processed % 2000000 == 0:
                    gc.collect()
        
        result_df = pd.DataFrame(edges_data)
        
        self.logger.info(f"      ✅ {len(result_df):,} edges created for {node_type}")
        
        return result_df
    
    def create_edges_intra_triangulo(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info('🔺 Creating intra-triangle edges WITH FILTERS...')
        
        start_time = datetime.now()
        
        edges_conexao = self._create_intra_triangle_edges_otimizado(data['ctcs'], 'conexao')
        
        edges_nao_conexao = self._create_intra_triangle_edges_otimizado(data['ctncs'], 'nao_conexao')
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.stats['edges_created']['intra_triangulo_conexao'] = len(edges_conexao)
        self.stats['edges_created']['intra_triangulo_nao_conexao'] = len(edges_nao_conexao)
        self.stats['processing_time']['intra_triangulo'] = elapsed_time
        
        self.logger.info(f"   ✅ Connection: {len(edges_conexao):,} edges")
        self.logger.info(f"   ✅ Non-connection: {len(edges_nao_conexao):,} edges")
        self.logger.info(f"   ⏱️ Tempo total: {elapsed_time:.2f}s")
        
        return edges_conexao, edges_nao_conexao
        
    def _create_intra_triangle_edges_otimizado(self, triangles_df: pd.DataFrame, triangle_type: str) -> pd.DataFrame:
        if len(triangles_df) == 0:
            return pd.DataFrame()
            
        self.logger.info(f"      Processing {len(triangles_df):,} triangles of type {triangle_type}")
        
        edges_data = []
        edge_type = f"intra_triangulo_{triangle_type}"
        
        chunk_size = self.chunk_size // 2
        total_chunks = len(triangles_df) // chunk_size + 1
        
        for i in range(0, len(triangles_df), chunk_size):
            chunk_num = i // chunk_size + 1
            if chunk_num % 10 == 0:
                self.logger.info(f"         Chunk {chunk_num}/{total_chunks}")
            
            chunk = triangles_df.iloc[i:i+chunk_size]
            
            if self.config.get('edge_config', {}).get('intra_triangulo', {}).get('validate_geometry', False):
                chunk = self._validate_triangle_geometry(chunk)
            
            triangle_ids = chunk['triangulation_id'].values
            pea_ids = chunk['pea_id'].values
            erb_ids = chunk['erb_id'].values
            cps_ids = chunk['cps_id'].values
            fator_punitivo = chunk.get('fator_punitivo', pd.Series([0.0] * len(chunk))).values
            
            for j in range(len(triangle_ids)):
                triangle_id = triangle_ids[j]
                pea_id = pea_ids[j]
                erb_id = erb_ids[j]
                cps_id = cps_ids[j]
                fator = fator_punitivo[j] if triangle_type == 'nao_conexao' else None
                
                edge_base = {
                    'tipo_aresta': edge_type,
                    'id_triangulo': triangle_id
                }
                if fator is not None:
                    edge_base['fator_punitivo'] = fator
                
                connections = [
                    (triangle_id, pea_id), (pea_id, triangle_id),
                    (triangle_id, erb_id), (erb_id, triangle_id),
                    (triangle_id, cps_id), (cps_id, triangle_id)
                ]
                
                for source, target in connections:
                    edge = edge_base.copy()
                    edge['source_id'] = source
                    edge['target_id'] = target
                    edges_data.append(edge)
            
            if chunk_num % 20 == 0:
                gc.collect()
        
        result_df = pd.DataFrame(edges_data)
        
        result_df = self._apply_quality_filters(result_df, edge_type)
        
        return result_df
    
    def _validate_triangle_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        original_count = len(df)
        
        valid_triangles = []
        
        for _, row in df.iterrows():
            pea_lat, pea_lon = row['pea_lat'], row['pea_lon']
            erb_lat, erb_lon = row['erb_lat'], row['erb_lon']
            cps_lat, cps_lon = row['cps_lat'], row['cps_lon']
            
            area = abs((pea_lat * (erb_lon - cps_lon) + 
                       erb_lat * (cps_lon - pea_lon) + 
                       cps_lat * (pea_lon - erb_lon)) / 2)
            
            if area > 1e-10:
                valid_triangles.append(row)
        
        if valid_triangles:
            result_df = pd.DataFrame(valid_triangles)
            invalid_count = original_count - len(result_df)
            if invalid_count > 0:
                self.logger.warning(f"         ⚠️ Removed {invalid_count:,} invalid triangles")
            return result_df
        else:
            return df
    
    def validate_edges(self, edges_dict: Dict[str, pd.DataFrame]) -> bool:
        self.logger.info("✅ Validando arestas geradas...")
        
        validation_config = self.config.get('validation', {})
        min_expected = validation_config.get('min_edges_expected', {})
        
        all_valid = True
        
        for edge_type, df_edges in edges_dict.items():
            if edge_type in min_expected:
                min_count = min_expected[edge_type]
                actual_count = len(df_edges)
                
                if actual_count < min_count:
                    self.logger.warning(f"   ⚠️ {edge_type}: {actual_count:,} < {min_count:,} (esperado)")
                    all_valid = False
                else:
                    self.logger.info(f"   ✅ {edge_type}: {actual_count:,} edges (valid)")
            
            if validation_config.get('check_distance_validity', True):
                self._validate_distances(df_edges, edge_type)
            
            if validation_config.get('check_geometric_consistency', True):
                self._validate_geometric_consistency(df_edges, edge_type)
        
        return all_valid
    
    def _validate_distances(self, df: pd.DataFrame, edge_type: str):
        if len(df) == 0:
            return
            
        distance_columns = [col for col in df.columns if 'distancia' in col.lower()]
        
        for col in distance_columns:
            if col in df.columns:
                negative_distances = (df[col] < 0).sum()
                zero_distances = (df[col] == 0).sum()
                
                if negative_distances > 0:
                    self.logger.warning(f"   ⚠️ {edge_type}: {negative_distances:,} negative distances in {col}")
                
                if zero_distances > len(df) * 0.1:
                    self.logger.warning(f"   ⚠️ {edge_type}: {zero_distances:,} zero distances in {col}")
    
    def _validate_geometric_consistency(self, df: pd.DataFrame, edge_type: str):
        if len(df) == 0:
            return
            
        if 'source_id' in df.columns and 'target_id' in df.columns:
            empty_source = df['source_id'].isna().sum()
            empty_target = df['target_id'].isna().sum()
            
            if empty_source > 0:
                self.logger.warning(f"   ⚠️ {edge_type}: {empty_source:,} source_id vazios")
            
            if empty_target > 0:
                self.logger.warning(f"   ⚠️ {edge_type}: {empty_target:,} target_id vazios")
    
    def run_pipeline(self) -> bool:
        self.logger.info("🚀 Iniciando pipeline SEQUENCIAL OTIMIZADO V3")
        self.logger.info("   ⚡ MELHORIAS: Filtros de qualidade, clustering, monitoramento")
        
        pipeline_start = datetime.now()
        
        try:
            data = self.load_data()
            
            edges_dict = {}
            
            if self.config.get('edge_types', {}).get('basic_connections', True):
                edges_dict['edges_conexao_pea_erb'] = self.create_edges_conexao_pea_erb(data)
            
            if self.config.get('edge_types', {}).get('non_connections', True):
                edges_dict['edges_nao_conexao'] = self.create_edges_nao_conexao(data)
            
            if self.config.get('edge_types', {}).get('intra_triangle', True):
                edges_conexao, edges_nao_conexao = self.create_edges_intra_triangulo(data)
                edges_dict['edges_intra_triangulo_conexao'] = edges_conexao
                edges_dict['edges_intra_triangulo_nao_conexao'] = edges_nao_conexao
            
            if self.config.get('edge_types', {}).get('same_type_nodes', True):
                same_type_edges = self.create_same_type_edges_sequencial(data)
                edges_dict.update(same_type_edges)
            
            self._generate_quality_metrics(edges_dict)
            
            self._check_quality_alerts(edges_dict)
            
            is_valid = self.validate_edges(edges_dict)
            
            saved_files = self.save_edges(edges_dict)
            
            self.save_stats()
            self._save_quality_reports()
            
            total_edges = sum(len(df) for df in edges_dict.values())
            elapsed_time = (datetime.now() - pipeline_start).total_seconds()
            
            self.logger.info('🎉 SEQUENTIAL OPTIMIZED V3 pipeline done!')
            self.logger.info(f"   📊 Total edges created: {total_edges:,}")
            self.logger.info(f"   📁 Arquivos salvos: {len(saved_files)}")
            self.logger.info(f"   ⏱️ Tempo total: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
            self.logger.info(f"   🔄 Performance: {total_edges/elapsed_time:.0f} arestas/segundo")
            self.logger.info(f"   ✅ Validação: {'Aprovada' if is_valid else 'Com avisos'}")
            self.logger.info(f"   🚨 Alertas: {len(self.stats['alerts_generated'])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def save_edges(self, edges_dict: Dict[str, pd.DataFrame]) -> List[str]:
        self.logger.info('💾 Saving edge files...')
        
        output_dir = self.config['output_dir']
        output_files = self.config['output_files']
        saved_files = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for edge_type, df_edges in edges_dict.items():
            if edge_type in output_files and len(df_edges) > 0:
                output_file = os.path.join(output_dir, output_files[edge_type])
                
                df_edges.to_csv(output_file, index=False)
                saved_files.append(output_file)
                
                file_size = os.path.getsize(output_file) / (1024*1024)
                self.logger.info(f"   ✅ {edge_type}: {len(df_edges):,} arestas → {output_file} ({file_size:.1f}MB)")
            elif len(df_edges) == 0:
                self.logger.warning(f"   ⚠️ {edge_type}: No edge to save")
        
        return saved_files
        
    def save_stats(self):
        if self.config.get('logging', {}).get('save_stats', False):
            stats_file = self.config.get('logging', {}).get('stats_file', 'arestas_stats_sequencial.json')
            
            self.stats['hardware_info'] = {
                'numba_available': False,
                'numba_used': False,
                'numpy_vectorization': True,
                'workers': 1,
                'processing_mode': 'SEQUENCIAL_OTIMIZADO_V3',
                'chunk_size': self.chunk_size,
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                os.makedirs(os.path.dirname(stats_file), exist_ok=True)
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, indent=2, ensure_ascii=False)
                self.logger.info(f"📊 Statistics saved to: {stats_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error while saving statistics: {e}")


if __name__ == "__main__":
    gerador = GeradorArestasV3Sequencial()
    success = gerador.run_pipeline()
    sys.exit(0 if success else 1) 
