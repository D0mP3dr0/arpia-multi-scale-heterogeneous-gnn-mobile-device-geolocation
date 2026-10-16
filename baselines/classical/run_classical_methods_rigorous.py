#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical localization baselines (RSSI, TOA, TDOA) with hierarchical clustering to allow trilateration."""

import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy.optimize import least_squares
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('classical_methods_clustering.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    cluster_id: int
    pea_ids: List[str]
    available_erbs: Set[str]
    centroid: np.ndarray


class RigorousClassicalTester:
    
    def __init__(self, 
                 graph_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 scalers_path: Optional[str] = None):
        logger.info('✅ RigorousClassicalTester initialized (with clustering)')
        
        self.base_dir = Path(__file__).parent.parent / "GNN" / "TREINAMENTO_REAL-20250717T215639Z-1-001" / "TREINAMENTO_REAL"
        if graph_path:
            self.graph_path = Path(graph_path)
            self.base_dir = self.graph_path.parent
        else:
            self.graph_path = self.base_dir / "arpia_hetero_graph_v3.pt"
        
        self.output_dir = Path(__file__).parent / "classical_methods_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.erb_latlon: Dict[str, Tuple[float, float]] = {}
        self.pea_initial_latlon: Dict[str, Tuple[float, float]] = {}
        self.erb_positions: Dict[str, np.ndarray] = {}
        self.pea_initial_positions: Dict[str, np.ndarray] = {}
        self.erb_data: Dict[str, Dict] = {}
        self.pea_connections: Dict[str, List[str]] = {}
        
        self.clusters: List[ClusterInfo] = []
        self.pea_to_cluster: Dict[str, int] = {}
        
        self.experiment_log: Dict[str, Any] = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'approach': 'Hierarchical Clustering for Classical Localization Methods',
                'description': 'Localization using hierarchical clustering to enable trilateration'
            },
            'configuration': {},
            'step1_analysis': {},
            'step2_clustering': {},
            'step3_localization': {},
            'step4_validation': {}
        }
        
        self.rssi_params = {
            'n': 2.5,
            'P0': -30.0,
            'sigma_db': 15.0
        }
        self.toa_params = {
            'c': 299792458.0,
            'sigma_ns': 200.0
        }
        self.tdoa_params = {
            'c': 299792458.0,
            'sigma_ns': 100.0
        }
        
        self.nodes_pea_csv_paths = []
        self.nodes_erb_csv_path = None
        self.edges_pea_erb_csv_path = None
        
        logger.info(f"📂 Base directory: {self.base_dir}")
    
    def _save_intermediate_log(self, step_name: str, data: Dict[str, Any]):
        self.experiment_log[f'step{step_name}'] = data
        log_file = self.output_dir / f"step_{step_name}_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        logger.info(f"💾 Intermediate log saved: {log_file}")
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def load_data(self):
        logger.info('📂 Loading ARPIA V3 graph data...')
        
        possible_pea_paths = [
            Path(__file__).parent.parent / "DATA_PROCESSED" / "GRAPH_NODES_EDGES" / "ARESTAS_V2" / "nodes_pea.csv",
            Path(__file__).parent.parent / "DATA_PROCESSED" / "GRAPH_NODES_EDGES" / "nodes_pea.csv",
            Path(__file__).parent.parent / "NORMALIZACAO_OUTPUT" / "nodes_pea_v3.csv",
        ]
        
        for csv_path in possible_pea_paths:
            if csv_path.exists():
                self.nodes_pea_csv_paths.append(csv_path)
                logger.info(f"✅ PEA CSV found: {csv_path}")
        
        possible_erb_paths = [
            Path(__file__).parent.parent / "NORMALIZACAO_OUTPUT" / "nodes_erb_v3.csv",
        ]
        
        for csv_path in possible_erb_paths:
            if csv_path.exists():
                self.nodes_erb_csv_path = csv_path
                logger.info(f"✅ ERB CSV found: {csv_path}")
                break
        
        self.edges_pea_erb_csv_path = Path(__file__).parent.parent / "NORMALIZACAO_OUTPUT" / "edges_conexao_pea_erb_v3.csv"
        if self.edges_pea_erb_csv_path.exists():
            logger.info(f"✅ Connections CSV found: {self.edges_pea_erb_csv_path}")
        
        self.experiment_log['configuration'] = {
            'rssi_params': self.rssi_params,
            'toa_params': self.toa_params,
            'tdoa_params': self.tdoa_params,
            'data_sources': {
                'pea_csvs': [str(p) for p in self.nodes_pea_csv_paths],
                'erb_csv': str(self.nodes_erb_csv_path) if self.nodes_erb_csv_path else None,
                'edges_csv': str(self.edges_pea_erb_csv_path) if self.edges_pea_erb_csv_path else None
            }
        }
        
        self._load_real_coordinates_from_csv()
        
        self._load_connections_from_csv()
        
        logger.info(f"✅ Dados carregados: {len(self.pea_initial_positions)} PEAs, {len(self.erb_positions)} ERBs")
    
    def _load_real_coordinates_from_csv(self):
        logger.info('🔄 Loading real coordinates from the CSVs...')
        
        for csv_path in self.nodes_pea_csv_paths:
            try:
                df_peas = pd.read_csv(csv_path, encoding='utf-8')
                id_col = 'node_id' if 'node_id' in df_peas.columns else 'pea_id'
                lat_col = 'pea_lat' if 'pea_lat' in df_peas.columns else 'lat'
                lon_col = 'pea_lon' if 'pea_lon' in df_peas.columns else 'lon'
                
                for _, row in df_peas.iterrows():
                    pea_id = str(row[id_col])
                    if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        if pea_id not in self.pea_initial_latlon:
                            self.pea_initial_latlon[pea_id] = (lat, lon)
            except Exception as e:
                logger.warning(f"⚠️ Error while loading {csv_path}: {e}")
        
        if self.nodes_erb_csv_path:
            try:
                df_erbs = pd.read_csv(self.nodes_erb_csv_path, encoding='utf-8')
                id_col = 'node_id' if 'node_id' in df_erbs.columns else 'erb_id'
                lat_col = 'lat_erb' if 'lat_erb' in df_erbs.columns else 'lat'
                lon_col = 'lon_erb' if 'lon_erb' in df_erbs.columns else 'lon'
                
                for _, row in df_erbs.iterrows():
                    erb_id = str(row[id_col])
                    if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        self.erb_latlon[erb_id] = (lat, lon)
                
                self._load_erb_additional_data(df_erbs, id_col)
            except Exception as e:
                logger.warning(f"⚠️ Error while loading ERBs: {e}")
        
        if len(self.erb_latlon) > 0:
            all_lats = [lat for lat, _ in self.erb_latlon.values()]
            all_lons = [lon for _, lon in self.erb_latlon.values()]
            self.ref_lat = np.mean(all_lats)
            self.ref_lon = np.mean(all_lons)
        else:
            self.ref_lat = 0.0
            self.ref_lon = 0.0
        
        for erb_id, (lat, lon) in self.erb_latlon.items():
            x_m, y_m = self._latlon_to_local_meters(lat, lon, self.ref_lat, self.ref_lon)
            self.erb_positions[erb_id] = np.array([x_m, y_m], dtype=np.float64)
        
        for pea_id, (lat, lon) in self.pea_initial_latlon.items():
            x_m, y_m = self._latlon_to_local_meters(lat, lon, self.ref_lat, self.ref_lon)
            self.pea_initial_positions[pea_id] = np.array([x_m, y_m], dtype=np.float64)
        
        logger.info(f"   ✅ {len(self.pea_initial_positions)} PEAs and {len(self.erb_positions)} ERBs with coordinates")
    
    def _load_erb_additional_data(self, df_erbs: pd.DataFrame, id_col: str):
        for _, row in df_erbs.iterrows():
            erb_id = str(row[id_col])
            if erb_id not in self.erb_latlon:
                continue
            
            erb_info = {}
            
            potencia_cols = ['PotenciaTransmissorWatts', 'potencia_transmissor_watts']
            ganho_cols = ['GanhoAntena', 'ganho_antena_dbi']
            freq_cols = ['FreqTxMHz', 'freq_tx_mhz']
            altura_cols = ['AlturaAntena', 'altura_antena_m']
            
            for col in potencia_cols:
                if col in df_erbs.columns and pd.notna(row.get(col)):
                    try:
                        erb_info['potencia_transmissor_watts'] = float(row[col])
                        break
                    except:
                        pass
            
            for col in ganho_cols:
                if col in df_erbs.columns and pd.notna(row.get(col)):
                    try:
                        valor = str(row[col]).replace(',', '.')
                        erb_info['ganho_antena_dbi'] = float(valor)
                        break
                    except:
                        pass
            
            for col in freq_cols:
                if col in df_erbs.columns and pd.notna(row.get(col)):
                    try:
                        erb_info['freq_tx_mhz'] = float(row[col])
                        break
                    except:
                        pass
            
            for col in altura_cols:
                if col in df_erbs.columns and pd.notna(row.get(col)):
                    try:
                        erb_info['altura_antena_m'] = float(row[col])
                        break
                    except:
                        pass
            
            if erb_info:
                self.erb_data[erb_id] = erb_info
    
    def _load_connections_from_csv(self):
        logger.info('🔄 Loading connections from the CSV...')
        
        if not self.edges_pea_erb_csv_path or not self.edges_pea_erb_csv_path.exists():
            logger.warning('⚠️ Connections file not found')
            return
        
        try:
            df_edges = pd.read_csv(self.edges_pea_erb_csv_path, encoding='utf-8')
            source_col = 'source_id' if 'source_id' in df_edges.columns else 'id_pea'
            target_col = 'target_id' if 'target_id' in df_edges.columns else 'id_erb'
            
            for _, row in df_edges.iterrows():
                pea_id = str(row[source_col])
                erb_id = str(row[target_col])
                
                if erb_id not in self.erb_latlon:
                    continue
                
                if pea_id not in self.pea_connections:
                    self.pea_connections[pea_id] = []
                
                if erb_id not in self.pea_connections[pea_id]:
                    self.pea_connections[pea_id].append(erb_id)
            
            logger.info(f"   ✅ {len(self.pea_connections)} PEAs with connections loaded")
        except Exception as e:
            logger.error(f"❌ Error while loading connections: {e}")
    
    def _latlon_to_local_meters(self, lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        lat_m = (lat - ref_lat) * 111000.0
        lon_m = (lon - ref_lon) * 111000.0 * np.cos(np.radians(ref_lat))
        return (lon_m, lat_m)
    
    
    def analyze_erb_distribution(self):
        logger.info("="*80)
        logger.info('STEP 1: EXPLORATORY ANALYSIS')
        logger.info("="*80)
        
        analysis_data = {
            'description': 'Exploratory analysis of the spatial distribution of ERBs and of PEA density',
            'timestamp': datetime.now().isoformat(),
            'erb_distribution': {},
            'pea_density': {},
            'spatial_statistics': {}
        }
        
        logger.info(f"📊 Spatial distribution of the ERBs:")
        logger.info(f"   - Total ERBs: {len(self.erb_positions)}")
        
        erb_positions_array = np.array(list(self.erb_positions.values()))
        erb_ids = list(self.erb_positions.keys())
        
        if len(erb_positions_array) > 0:
            centroid = np.mean(erb_positions_array, axis=0)
            std_dev = np.std(erb_positions_array, axis=0)
            min_coords = np.min(erb_positions_array, axis=0)
            max_coords = np.max(erb_positions_array, axis=0)
            
            analysis_data['erb_distribution'] = {
                'total_erbs': len(self.erb_positions),
                'centroid_m': centroid.tolist(),
                'std_dev_m': std_dev.tolist(),
                'min_coords_m': min_coords.tolist(),
                'max_coords_m': max_coords.tolist(),
                'erb_positions': {
                    erb_id: pos.tolist() for erb_id, pos in zip(erb_ids, erb_positions_array)
                }
            }
            
            logger.info(f"   - Centroide: ({centroid[0]:.2f}m, {centroid[1]:.2f}m)")
            logger.info(f"   - Standard deviation: ({std_dev[0]:.2f}m, {std_dev[1]:.2f}m)")
        
        if len(erb_positions_array) > 1:
            distances = pdist(erb_positions_array)
            min_dist = float(np.min(distances))
            max_dist = float(np.max(distances))
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            median_dist = float(np.median(distances))
            
            hist, bins = np.histogram(distances, bins=20)
            
            analysis_data['spatial_statistics'] = {
                'erb_distances': {
                    'min_m': min_dist,
                    'max_m': max_dist,
                    'mean_m': mean_dist,
                    'median_m': median_dist,
                    'std_m': std_dist,
                    'histogram': {
                        'counts': hist.tolist(),
                        'bins': bins.tolist()
                    }
                }
            }
            
            logger.info(f"   - Minimum distance between ERBs: {min_dist:.2f}m")
            logger.info(f"   - Maximum distance between ERBs: {max_dist:.2f}m")
            logger.info(f"   - Mean distance between ERBs: {mean_dist:.2f}m")
        
        logger.info(f"\n📊 PEA density per ERB:")
        erb_pea_counts = defaultdict(int)
        for pea_id, erbs in self.pea_connections.items():
            for erb_id in erbs:
                erb_pea_counts[erb_id] += 1
        
        erb_density_stats = []
        for erb_id, count in sorted(erb_pea_counts.items(), key=lambda x: x[1], reverse=True):
            erb_density_stats.append({
                'erb_id': erb_id,
                'pea_count': count,
                'erb_position': self.erb_positions[erb_id].tolist() if erb_id in self.erb_positions else None
            })
        
        analysis_data['pea_density'] = {
            'total_peas': len(self.pea_connections),
            'total_connections': sum(len(erbs) for erbs in self.pea_connections.values()),
            'erbs_with_peas': len(erb_pea_counts),
            'mean_peas_per_erb': float(np.mean(list(erb_pea_counts.values()))) if erb_pea_counts else 0.0,
            'median_peas_per_erb': float(np.median(list(erb_pea_counts.values()))) if erb_pea_counts else 0.0,
            'std_peas_per_erb': float(np.std(list(erb_pea_counts.values()))) if erb_pea_counts else 0.0,
            'min_peas_per_erb': int(min(erb_pea_counts.values())) if erb_pea_counts else 0,
            'max_peas_per_erb': int(max(erb_pea_counts.values())) if erb_pea_counts else 0,
            'top_10_erbs': erb_density_stats[:10],
            'all_erbs': erb_density_stats
        }
        
        for erb_stat in erb_density_stats[:10]:
            logger.info(f"   - {erb_stat['erb_id']}: {erb_stat['pea_count']} PEAs")
        
        connections_per_pea = [len(erbs) for erbs in self.pea_connections.values()]
        analysis_data['connections_per_pea'] = {
            'mean': float(np.mean(connections_per_pea)) if connections_per_pea else 0.0,
            'median': float(np.median(connections_per_pea)) if connections_per_pea else 0.0,
            'std': float(np.std(connections_per_pea)) if connections_per_pea else 0.0,
            'min': int(min(connections_per_pea)) if connections_per_pea else 0,
            'max': int(max(connections_per_pea)) if connections_per_pea else 0,
            'distribution': {
                '1_erb': int(sum(1 for c in connections_per_pea if c == 1)),
                '2_erbs': int(sum(1 for c in connections_per_pea if c == 2)),
                '3_plus_erbs': int(sum(1 for c in connections_per_pea if c >= 3))
            }
        }
        
        logger.info(f"   📊 PEAs com 1 ERB: {analysis_data['connections_per_pea']['distribution']['1_erb']}")
        logger.info(f"   📊 PEAs com 2 ERBs: {analysis_data['connections_per_pea']['distribution']['2_erbs']}")
        logger.info(f"   📊 PEAs com 3+ ERBs: {analysis_data['connections_per_pea']['distribution']['3_plus_erbs']}")
        
        logger.info('✅ Exploratory analysis done')
        
        self._save_intermediate_log('1_analysis', analysis_data)
        self.experiment_log['step1_analysis'] = analysis_data
    
    
    def create_clusters(self, max_cluster_size: int = 100, max_geographic_distance: float = 2000.0):
        logger.info("="*80)
        logger.info('STEP 2: HIERARCHICAL CLUSTERING')
        logger.info("="*80)
        
        clustering_data = {
            'description': 'Hierarchical clustering of PEAs by geographic proximity, then aggregating ERBs',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_cluster_size': max_cluster_size,
                'max_geographic_distance_m': max_geographic_distance,
                'clustering_method': 'Ward linkage (hierarchical)'
            },
            'strategy': 'Geographic clustering first, then aggregate ERBs from cluster members',
            'final_clusters': []
        }
        
        logger.info('🔄 Creating estimated positions for PEAs without coordinates...')
        
        pea_positions = {}
        for pea_id, connected_erbs in self.pea_connections.items():
            if pea_id in self.pea_initial_positions:
                pea_positions[pea_id] = self.pea_initial_positions[pea_id]
            elif len(connected_erbs) > 0:
                erb_id = connected_erbs[0]
                if erb_id in self.erb_positions:
                    erb_pos = self.erb_positions[erb_id]
                    distance = self._get_distance_from_csv(pea_id, erb_id)
                    if distance is None:
                        distance = 1000.0
                    
                    azimute = self._get_azimute_from_csv(pea_id, erb_id)
                    if azimute is None:
                        azimute = np.random.uniform(0, 360)
                    
                    azimute_rad = np.radians(azimute)
                    dx = distance * np.sin(azimute_rad)
                    dy = distance * np.cos(azimute_rad)
                    pea_positions[pea_id] = erb_pos + np.array([dx, dy])
        
        logger.info(f"   ✅ {len(pea_positions)} PEAs with positions (estimated or real)")
        
        logger.info('🔄 Grouping PEAs by geographic proximity...')
        
        if len(pea_positions) == 0:
            logger.warning('⚠️ No PEA with a position available for clustering')
            return
        
        pea_ids_list = list(pea_positions.keys())
        positions_array = np.array([pea_positions[pid] for pid in pea_ids_list])
        
        distances = pdist(positions_array)
        linkage_matrix = linkage(distances, method='ward')
        
        num_clusters = max(1, (len(pea_ids_list) + max_cluster_size - 1) // max_cluster_size)
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        logger.info(f"   ✅ {num_clusters} geographic clusters created")
        
        cluster_id = 0
        self.clusters = []
        self.pea_to_cluster = {}
        
        for cluster_label in np.unique(cluster_labels):
            cluster_pea_ids = [pea_ids_list[i] for i in range(len(pea_ids_list)) 
                             if cluster_labels[i] == cluster_label]
            
            if len(cluster_pea_ids) > 0:
                self._create_cluster(cluster_id, cluster_pea_ids, None)
                cluster_id += 1
        
        clustering_data['geographic_clustering'] = {
            'total_peas_clustered': len(pea_positions),
            'num_clusters_created': len(self.clusters),
            'clustering_method': 'Ward linkage'
        }
        
        logger.info(f"   ✅ {len(self.clusters)} clusters criados")
        
        cluster_details = []
        cluster_sizes = []
        erb_counts = []
        
        pea_positions_all = {}
        for pea_id in self.pea_to_cluster.keys():
            if pea_id in self.pea_initial_positions:
                pea_positions_all[pea_id] = self.pea_initial_positions[pea_id]
            elif pea_id in self.pea_connections and len(self.pea_connections[pea_id]) > 0:
                erb_id = self.pea_connections[pea_id][0]
                if erb_id in self.erb_positions:
                    distance = self._get_distance_from_csv(pea_id, erb_id) or 1000.0
                    azimute = self._get_azimute_from_csv(pea_id, erb_id) or np.random.uniform(0, 360)
                    azimute_rad = np.radians(azimute)
                    dx = distance * np.sin(azimute_rad)
                    dy = distance * np.cos(azimute_rad)
                    pea_positions_all[pea_id] = self.erb_positions[erb_id] + np.array([dx, dy])
        
        for cluster in self.clusters:
            positions = [pea_positions_all[p] for p in cluster.pea_ids if p in pea_positions_all]
            if len(positions) > 0:
                positions_array = np.array(positions)
                cluster_std = np.std(positions_array, axis=0)
                cluster_radius = float(np.max([np.linalg.norm(p - cluster.centroid) for p in positions]))
            else:
                cluster_std = [0.0, 0.0]
                cluster_radius = 0.0
            
            cluster_details.append({
                'cluster_id': cluster.cluster_id,
                'pea_count': len(cluster.pea_ids),
                'available_erbs': list(cluster.available_erbs),
                'erb_count': len(cluster.available_erbs),
                'centroid_m': cluster.centroid.tolist(),
                'std_dev_m': cluster_std.tolist(),
                'max_radius_m': cluster_radius,
                'has_3_plus_erbs': len(cluster.available_erbs) >= 3,
                'pea_ids_sample': cluster.pea_ids[:20]
            })
            
            cluster_sizes.append(len(cluster.pea_ids))
            erb_counts.append(len(cluster.available_erbs))
        
        clustering_data['final_clusters'] = cluster_details
        
        clustering_data['statistics'] = {
            'total_clusters': len(self.clusters),
            'cluster_sizes': {
                'mean': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
                'median': float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
                'std': float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
                'min': int(min(cluster_sizes)) if cluster_sizes else 0,
                'max': int(max(cluster_sizes)) if cluster_sizes else 0
            },
            'erb_counts_per_cluster': {
                'mean': float(np.mean(erb_counts)) if erb_counts else 0.0,
                'median': float(np.median(erb_counts)) if erb_counts else 0.0,
                'std': float(np.std(erb_counts)) if erb_counts else 0.0,
                'min': int(min(erb_counts)) if erb_counts else 0,
                'max': int(max(erb_counts)) if erb_counts else 0
            },
            'clusters_with_3_plus_erbs': int(sum(1 for c in self.clusters if len(c.available_erbs) >= 3)),
            'clusters_with_2_erbs': int(sum(1 for c in self.clusters if len(c.available_erbs) == 2)),
            'clusters_with_1_erb': int(sum(1 for c in self.clusters if len(c.available_erbs) == 1)),
            'peas_in_clusters': len(self.pea_to_cluster),
            'total_peas_with_connections': len(self.pea_connections)
        }
        
        logger.info(f"   📊 Mean cluster size: {np.mean(cluster_sizes) if cluster_sizes else 0:.1f} PEAs")
        logger.info(f"   📊 Mean ERBs per cluster: {np.mean(erb_counts) if erb_counts else 0:.1f}")
        logger.info(f"   📊 Clusters com 3+ ERBs: {sum(1 for c in self.clusters if len(c.available_erbs) >= 3)}")
        logger.info(f"   📊 Clusters com 2 ERBs: {sum(1 for c in self.clusters if len(c.available_erbs) == 2)}")
        logger.info(f"   📊 Clusters com 1 ERB: {sum(1 for c in self.clusters if len(c.available_erbs) == 1)}")
        
        logger.info("✅ Clusterização concluída")
        
        self._save_intermediate_log('2_clustering', clustering_data)
        self.experiment_log['step2_clustering'] = clustering_data
    
    def _create_cluster(self, cluster_id: int, pea_ids: List[str], erb_key: Optional[Tuple[str, ...]]):
        available_erbs = set()
        for pea_id in pea_ids:
            if pea_id in self.pea_connections:
                available_erbs.update(self.pea_connections[pea_id])
        
        positions = []
        for pea_id in pea_ids:
            if pea_id in self.pea_initial_positions:
                positions.append(self.pea_initial_positions[pea_id])
            elif pea_id in self.pea_connections and len(self.pea_connections[pea_id]) > 0:
                erb_id = self.pea_connections[pea_id][0]
                if erb_id in self.erb_positions:
                    distance = self._get_distance_from_csv(pea_id, erb_id) or 1000.0
                    azimute = self._get_azimute_from_csv(pea_id, erb_id) or 0.0
                    azimute_rad = np.radians(azimute)
                    dx = distance * np.sin(azimute_rad)
                    dy = distance * np.cos(azimute_rad)
                    positions.append(self.erb_positions[erb_id] + np.array([dx, dy]))
        
        if len(positions) > 0:
            centroid = np.mean(positions, axis=0)
        else:
            centroid = np.array([0.0, 0.0])
        
        cluster = ClusterInfo(
            cluster_id=cluster_id,
            pea_ids=pea_ids,
            available_erbs=available_erbs,
            centroid=centroid
        )
        
        self.clusters.append(cluster)
        
        for pea_id in pea_ids:
            self.pea_to_cluster[pea_id] = cluster_id
    
    
    def localize_with_clustering(self, pea_id: str, method: str = 'RSSI') -> Tuple[np.ndarray, Dict[str, Any]]:
        localization_info = {
            'method': method,
            'pea_id': pea_id,
            'used_cluster': False,
            'num_erbs_used': 0,
            'localization_method': 'direct'
        }
        
        cluster_id = self.pea_to_cluster.get(pea_id)
        if cluster_id is None:
            localization_info['localization_method'] = 'direct_no_cluster'
            return self._localize_direct(pea_id), localization_info
        
        cluster = self.clusters[cluster_id]
        localization_info['cluster_id'] = cluster_id
        localization_info['available_erbs_in_cluster'] = len(cluster.available_erbs)
        
        connected_erbs = self.pea_connections.get(pea_id, [])
        if len(connected_erbs) == 0:
            return np.array([0.0, 0.0]), localization_info
        
        primary_erb = connected_erbs[0]
        available_erbs = list(cluster.available_erbs)
        
        if len(available_erbs) < 3:
            localization_info['localization_method'] = 'direct_insufficient_erbs'
            return self._localize_direct(pea_id), localization_info
        
        erb_positions_list = []
        distances_list = []
        erb_info_list = []
        
        for erb_id in available_erbs:
            if erb_id not in self.erb_positions:
                continue
            
            erb_pos = self.erb_positions[erb_id]
            erb_positions_list.append(erb_pos)
            
            if erb_id == primary_erb:
                distance = self._get_distance_from_csv(pea_id, erb_id)
                if distance is None:
                    distance = 1000.0
                erb_info_list.append({
                    'erb_id': erb_id,
                    'distance_source': 'csv',
                    'distance_m': distance
                })
            else:
                base_distance = np.linalg.norm(cluster.centroid - erb_pos)
                
                if method == 'RSSI':
                    noise_factor = np.random.normal(1.0, 0.3)
                    distance = base_distance * max(0.5, noise_factor)
                elif method == 'TOA':
                    toa_noise_ns = np.random.normal(0.0, self.toa_params['sigma_ns'])
                    toa_noise_m = (toa_noise_ns * 1e-9) * self.toa_params['c']
                    distance = base_distance + toa_noise_m
                elif method == 'TDOA':
                    tdoa_noise_ns = np.random.normal(0.0, self.tdoa_params['sigma_ns'])
                    tdoa_noise_m = (tdoa_noise_ns * 1e-9) * self.tdoa_params['c']
                    distance = base_distance + tdoa_noise_m
                else:
                    distance = base_distance
                
                distance = max(100.0, min(10000.0, distance))
                
                erb_info_list.append({
                    'erb_id': erb_id,
                    'distance_source': f'estimated_{method.lower()}',
                    'distance_m': distance,
                    'base_distance_from_centroid_m': float(base_distance)
                })
            
            distances_list.append(distance)
        
        if len(distances_list) < 3:
            localization_info['localization_method'] = 'direct_insufficient_measurements'
            return self._localize_direct(pea_id), localization_info
        
        erb_positions_array = np.array(erb_positions_list)
        distances_array = np.array(distances_list)
        
        def residuals(p):
            x, y = p
            distances_estimated = np.sqrt(np.sum((erb_positions_array - np.array([x, y])) ** 2, axis=1))
            return distances_estimated - distances_array
        
        initial_guess = self._localize_direct(pea_id)
        if np.allclose(initial_guess, [0.0, 0.0]):
            initial_guess = cluster.centroid
        
        try:
            result = least_squares(residuals, initial_guess, method='lm')
            localization_info['localization_method'] = 'trilateration'
            localization_info['used_cluster'] = True
            localization_info['num_erbs_used'] = len(erb_positions_list)
            localization_info['optimization_success'] = True
            localization_info['initial_guess'] = initial_guess.tolist()
            localization_info['final_position'] = result.x.tolist()
            localization_info['optimization_cost'] = float(np.sum(result.fun ** 2))
            localization_info['erbs_used'] = erb_info_list
            
            return result.x, localization_info
        except Exception as e:
            localization_info['optimization_success'] = False
            localization_info['optimization_error'] = str(e)
            return initial_guess, localization_info
    
    def _localize_direct(self, pea_id: str) -> np.ndarray:
        connected_erbs = self.pea_connections.get(pea_id, [])
        if len(connected_erbs) == 0:
            return np.array([0.0, 0.0])
        
        erb_id = connected_erbs[0]
        if erb_id not in self.erb_positions:
            return np.array([0.0, 0.0])
        
        erb_pos = self.erb_positions[erb_id]
        
        distance = self._get_distance_from_csv(pea_id, erb_id)
        azimute = self._get_azimute_from_csv(pea_id, erb_id)
        
        if distance and azimute is not None:
            azimute_rad = np.radians(azimute)
            dx = distance * np.sin(azimute_rad)
            dy = distance * np.cos(azimute_rad)
            return erb_pos + np.array([dx, dy])
        else:
            return erb_pos
    
    def _get_distance_from_csv(self, pea_id: str, erb_id: str) -> Optional[float]:
        if not self.edges_pea_erb_csv_path or not self.edges_pea_erb_csv_path.exists():
            return None
        
        try:
            df_edges = pd.read_csv(self.edges_pea_erb_csv_path)
            source_col = 'source_id' if 'source_id' in df_edges.columns else 'id_pea'
            target_col = 'target_id' if 'target_id' in df_edges.columns else 'id_erb'
            
            matching = df_edges[(df_edges[source_col] == pea_id) & (df_edges[target_col] == erb_id)]
            if len(matching) > 0 and pd.notna(matching.iloc[0].get('distancia_m')):
                return float(matching.iloc[0]['distancia_m'])
        except:
            pass
        
        return None
    
    def _get_azimute_from_csv(self, pea_id: str, erb_id: str) -> Optional[float]:
        if not self.edges_pea_erb_csv_path or not self.edges_pea_erb_csv_path.exists():
            return None
        
        try:
            df_edges = pd.read_csv(self.edges_pea_erb_csv_path)
            source_col = 'source_id' if 'source_id' in df_edges.columns else 'id_pea'
            target_col = 'target_id' if 'target_id' in df_edges.columns else 'id_erb'
            
            matching = df_edges[(df_edges[source_col] == pea_id) & (df_edges[target_col] == erb_id)]
            if len(matching) > 0 and pd.notna(matching.iloc[0].get('azimute_pea_erb')):
                return float(matching.iloc[0]['azimute_pea_erb'])
        except:
            pass
        
        return None
    
    def _estimate_distance_rssi(self, pea_id: str, erb_id: str) -> float:
        if pea_id not in self.pea_initial_positions or erb_id not in self.erb_positions:
            return 1000.0
        
        pea_pos = self.pea_initial_positions[pea_id]
        erb_pos = self.erb_positions[erb_id]
        distance = np.linalg.norm(pea_pos - erb_pos)
        
        noise_factor = np.random.normal(1.0, 0.3)
        return distance * max(0.5, noise_factor)
    
    def _estimate_distance_toa(self, pea_id: str, erb_id: str) -> float:
        if pea_id not in self.pea_initial_positions or erb_id not in self.erb_positions:
            return 1000.0
        
        pea_pos = self.pea_initial_positions[pea_id]
        erb_pos = self.erb_positions[erb_id]
        distance = np.linalg.norm(pea_pos - erb_pos)
        
        toa_noise_ns = np.random.normal(0.0, self.toa_params['sigma_ns'])
        toa_noise_m = (toa_noise_ns * 1e-9) * self.toa_params['c']
        return distance + toa_noise_m
    
    def _estimate_distance_tdoa(self, pea_id: str, erb_id: str) -> float:
        if pea_id not in self.pea_initial_positions or erb_id not in self.erb_positions:
            return 1000.0
        
        pea_pos = self.pea_initial_positions[pea_id]
        erb_pos = self.erb_positions[erb_id]
        distance = np.linalg.norm(pea_pos - erb_pos)
        
        tdoa_noise_ns = np.random.normal(0.0, self.tdoa_params['sigma_ns'])
        tdoa_noise_m = (tdoa_noise_ns * 1e-9) * self.tdoa_params['c']
        return distance + tdoa_noise_m
    
    
    def evaluate_method(self, method_name: str) -> Dict[str, Any]:
        logger.info("="*80)
        logger.info(f"STEP 4: EVALUATING METHOD {method_name}")
        logger.info("="*80)
        
        validation_data = {
            'description': f"Validation of method {method_name}",
            'timestamp': datetime.now().isoformat(),
            'method': method_name,
            'results': []
        }
        
        errors = []
        results = []
        localization_methods_used = defaultdict(int)
        
        valid_peas = {}
        
        for pea_id, pos in self.pea_initial_positions.items():
            if pea_id in self.pea_connections:
                valid_peas[pea_id] = {'position': pos, 'has_ground_truth': True}
        
        for pea_id, connected_erbs in self.pea_connections.items():
            if pea_id not in valid_peas and len(connected_erbs) > 0:
                erb_id = connected_erbs[0]
                if erb_id in self.erb_positions:
                    erb_pos = self.erb_positions[erb_id]
                    distance = self._get_distance_from_csv(pea_id, erb_id) or 1000.0
                    azimute = self._get_azimute_from_csv(pea_id, erb_id)
                    if azimute is not None:
                        azimute_rad = np.radians(azimute)
                        dx = distance * np.sin(azimute_rad)
                        dy = distance * np.cos(azimute_rad)
                        estimated_pos = erb_pos + np.array([dx, dy])
                        valid_peas[pea_id] = {'position': estimated_pos, 'has_ground_truth': False}
        
        logger.info(f"📊 Avaliando {len(valid_peas)} PEAs ({sum(1 for v in valid_peas.values() if v['has_ground_truth'])} com ground truth real)")
        
        total_peas = len(valid_peas)
        log_interval = max(1, total_peas // 20)
        processed = 0
        
        for pea_id, pea_info in valid_peas.items():
            initial_position = pea_info['position']
            has_ground_truth = pea_info['has_ground_truth']
            try:
                predicted_position, loc_info = self.localize_with_clustering(pea_id, method=method_name)
                error = np.linalg.norm(predicted_position - initial_position)
                errors.append(error)
                
                localization_methods_used[loc_info['localization_method']] += 1
                
                result_entry = {
                    'pea_id': pea_id,
                    'predicted_position_m': predicted_position.tolist(),
                    'initial_position_m': initial_position.tolist(),
                    'error_m': float(error),
                    'has_ground_truth': has_ground_truth,
                    'localization_info': loc_info
                }
                
                results.append(result_entry)
                validation_data['results'].append(result_entry)
                
                processed += 1
                if processed % log_interval == 0:
                    progress = (processed / total_peas) * 100
                    if len(errors) > 0:
                        mae = np.mean(errors)
                        rmse = np.sqrt(np.mean(np.array(errors)**2))
                        logger.info(f"   📈 Progresso: {processed}/{total_peas} ({progress:.1f}%) | "
                                  f"MAE: {mae:.2f}m | RMSE: {rmse:.2f}m")
            except Exception as e:
                logger.debug(f"Error while processing {pea_id}: {e}")
                continue
        
        if len(errors) == 0:
            return {'status': 'ERROR', 'method': method_name}
        
        errors_array = np.array(errors)
        
        mae = float(np.mean(errors_array))
        rmse = float(np.sqrt(np.mean(errors_array ** 2)))
        median_error = float(np.median(errors_array))
        std_error = float(np.std(errors_array))
        min_error = float(np.min(errors_array))
        max_error = float(np.max(errors_array))
        
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        error_percentiles = {f'p{p}': float(np.percentile(errors_array, p)) for p in percentiles}
        
        thresholds = [50, 100, 200, 300, 500, 1000]
        accuracy_at_thresholds = {f'acc@{t}m': float(np.mean(errors_array <= t) * 100.0) for t in thresholds}
        
        hist, bins = np.histogram(errors_array, bins=50)
        
        metrics = {
            'method': method_name,
            'num_peas': len(errors),
            'metrics': {
                'mae_m': mae,
                'rmse_m': rmse,
                'median_error_m': median_error,
                'std_error_m': std_error,
                'min_error_m': min_error,
                'max_error_m': max_error
            },
            'error_percentiles_m': error_percentiles,
            'accuracy_at_thresholds_percent': accuracy_at_thresholds,
            'error_distribution': {
                'histogram_counts': hist.tolist(),
                'histogram_bins': bins.tolist()
            },
            'localization_methods_used': dict(localization_methods_used),
            'status': 'SUCCESS'
        }
        
        validation_data['metrics'] = metrics
        validation_data['summary'] = {
            'total_peas_evaluated': len(errors),
            'mean_absolute_error_m': mae,
            'root_mean_squared_error_m': rmse,
            'median_error_m': median_error
        }
        
        logger.info(f"   ✅ {method_name}: MAE={mae:.2f}m, RMSE={rmse:.2f}m, Accuracy@500m={accuracy_at_thresholds['acc@500m']:.1f}%")
        logger.info(f"   📊 Localization methods used: {dict(localization_methods_used)}")
        
        self._save_intermediate_log(f'4_validation_{method_name.lower()}', validation_data)
        
        return metrics
    
    def run_all_tests(self):
        logger.info('🚀 STARTING TESTS WITH CLUSTERING')
        logger.info("="*80)
        
        self.load_data()
        
        self.analyze_erb_distribution()
        
        self.create_clusters()
        
        results = {}
        validation_logs = {}
        
        for method in ['RSSI', 'TOA', 'TDOA']:
            method_result = self.evaluate_method(method)
            results[method] = method_result
            validation_logs[method] = self.experiment_log.get(f'step4_validation_{method.lower()}', {})
        
        self.experiment_log['step4_validation'] = validation_logs
        
        self._save_complete_experiment_log(results)
        
        self._save_results(results)
        self._show_summary(results)
        
        logger.info("="*80)
        logger.info("✅ TESTES CONCLUÍDOS")
        logger.info("="*80)
        
        return results
    
    def _save_complete_experiment_log(self, results: Dict[str, Any]):
        logger.info("💾 Salvando log completo do experimento...")
        
        self.experiment_log['final_results'] = results
        
        log_file = self.output_dir / "complete_experiment_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        logger.info(f"✅ Full log saved to {log_file}")
    
    def _save_results(self, results: Dict[str, Any]):
        logger.info("💾 Salvando resultados...")
        
        results_file = self.output_dir / "classical_methods_clustering_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        logger.info(f"✅ Results saved to {results_file}")
    
    def _show_summary(self, results: Dict[str, Any]):
        logger.info("\n" + "="*80)
        logger.info('📊 SUMMARY OF THE RESULTS')
        logger.info("="*80)
        
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('metrics', {}).get('mae_m', float('inf'))
        )
        
        logger.info(f"{'Método':<10} {'MAE (m)':<12} {'RMSE (m)':<12} {'Mediana':<12} "
                   f"{'Acc@100m':<12} {'Acc@200m':<12} {'Acc@500m':<12}")
        logger.info("-" * 80)
        
        for method_name, metrics in sorted_results:
            if metrics.get('status') == 'ERROR':
                logger.info(f"{method_name:<10} {'ERROR':<12}")
            else:
                m = metrics.get('metrics', {})
                acc = metrics.get('accuracy_at_thresholds_percent', {})
                logger.info(
                    f"{method_name:<10} "
                    f"{m.get('mae_m', 0):<12.2f} "
                    f"{m.get('rmse_m', 0):<12.2f} "
                    f"{m.get('median_error_m', 0):<12.2f} "
                    f"{acc.get('acc@100m', 0):<12.1f}% "
                    f"{acc.get('acc@200m', 0):<12.1f}% "
                    f"{acc.get('acc@500m', 0):<12.1f}%"
                )
        
        logger.info("="*80)


def main():
    try:
        logger.info('🧪 Starting test with clustering...')
        
        tester = RigorousClassicalTester()
        results = tester.run_all_tests()
        
        logger.info('✅ Test finished successfully!')
        return results
        
    except Exception as e:
        logger.error(f"❌ ERRO FATAL: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
