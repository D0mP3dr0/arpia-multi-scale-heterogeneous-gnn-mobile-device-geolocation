#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced variant of the classical localization baselines (RSSI, TOA, TDOA)."""

import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from scipy.optimize import least_squares, minimize, differential_evolution
from scipy.stats import norm, chi2, kstest, anderson
from scipy.spatial.distance import pdist, cdist
from scipy.interpolate import griddata
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_classical_methods.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13


@dataclass
class LocalizationResult:
    pea_id: str
    method: str
    algorithm: str
    estimated_position: np.ndarray
    ground_truth_position: Optional[np.ndarray]
    error_m: Optional[float]
    confidence_score: float
    computation_time_ms: float
    num_connected_erbs: int
    num_non_connected_erbs: int
    convergence_info: Dict[str, Any]
    probability_map: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    mean_error: float
    median_error: float
    std_error: float
    rmse: float
    mae: float
    
    percentiles: Dict[int, float]
    
    accuracy_at_thresholds: Dict[float, float]
    
    normality_test: Dict[str, Any]
    outliers: Dict[str, Any]
    
    convergence_rate: float
    failed_cases: int
    
    consistency_score: float
    spatial_coherence: float
    
    confidence_intervals: Dict[str, Tuple[float, float]]


class EnhancedClassicalLocalizer:
    
    def __init__(self, output_dir: str = "enhanced_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.erb_positions: Dict[str, np.ndarray] = {}
        self.pea_positions: Dict[str, np.ndarray] = {}
        self.connections: Dict[str, List[str]] = {}
        self.non_connections: Dict[str, List[str]] = {}
        
        self.propagation_params = {
            'rssi': {
                'path_loss_exponent': 3.5,
                'reference_power_dbm': -30,
                'shadowing_std_db': 8.0,
                'max_range_m': 5000.0
            },
            'toa': {
                'speed_of_light_mps': 299792458.0,
                'timing_error_ns': 100.0,
                'clock_drift_ppm': 10.0
            },
            'tdoa': {
                'speed_of_light_mps': 299792458.0,
                'sync_error_ns': 50.0
            }
        }
        
        self.algorithm_configs = {
            'weighted_trilateration': {
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'weight_decay': 0.5
            },
            'mle': {
                'optimization_method': 'L-BFGS-B',
                'max_iterations': 2000,
                'initial_points': 10
            },
            'particle_filter': {
                'num_particles': 1000,
                'resampling_threshold': 0.5,
                'noise_std_m': 50.0
            }
        }
        
        self.results: List[LocalizationResult] = []
        self.experiment_metadata: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'approach': 'Connection-Aware Trilateration (CAT)',
            'algorithms': list(self.algorithm_configs.keys()),
            'propagation_models': list(self.propagation_params.keys())
        }
        
        logger.info("Enhanced Classical Localizer initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self, data_dir: str = "NORMALIZACAO_OUTPUT"):
        logger.info("Loading real data from parquet files...")
        data_path = Path(data_dir)
        
        pea_file = data_path / "nodes_pea_v3.parquet"
        if pea_file.exists():
            df_peas = pd.read_parquet(pea_file)
        else:
            df_peas = pd.read_csv(data_path / "nodes_pea_v3.csv")
        logger.info(f"Loaded {len(df_peas)} PEAs")
        
        pea_id_col = 'node_id' if 'node_id' in df_peas.columns else 'pea_id'
        lat_col = 'pea_lat' if 'pea_lat' in df_peas.columns else 'lat'
        lon_col = 'pea_lon' if 'pea_lon' in df_peas.columns else 'lon'
        
        if 'x' in df_peas.columns and 'y' in df_peas.columns:
            for _, row in df_peas.iterrows():
                pea_id = str(row[pea_id_col])
                if pd.notna(row['x']) and pd.notna(row['y']):
                    self.pea_positions[pea_id] = np.array([float(row['x']), float(row['y'])])
        else:
            for _, row in df_peas.iterrows():
                pea_id = str(row[pea_id_col])
                if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
                    lat, lon = float(row[lat_col]), float(row[lon_col])
                    x, y = self._latlon_to_meters(lat, lon)
                    self.pea_positions[pea_id] = np.array([x, y])
        
        logger.info(f"Found {len(self.pea_positions)} PEAs with positions")
        
        erb_file = data_path / "nodes_erb_v3.parquet"
        if erb_file.exists():
            df_erbs = pd.read_parquet(erb_file)
        else:
            df_erbs = pd.read_csv(data_path / "nodes_erb_v3.csv")
        logger.info(f"Loaded {len(df_erbs)} ERBs")
        
        erb_id_col = 'node_id' if 'node_id' in df_erbs.columns else 'erb_id'
        
        if 'x' in df_erbs.columns and 'y' in df_erbs.columns:
            for _, row in df_erbs.iterrows():
                erb_id = str(row[erb_id_col])
                if pd.notna(row['x']) and pd.notna(row['y']):
                    self.erb_positions[erb_id] = np.array([float(row['x']), float(row['y'])])
        else:
            erb_lat_col = 'lat_erb' if 'lat_erb' in df_erbs.columns else 'lat'
            erb_lon_col = 'lon_erb' if 'lon_erb' in df_erbs.columns else 'lon'
            
            for _, row in df_erbs.iterrows():
                erb_id = str(row[erb_id_col])
                if pd.notna(row.get(erb_lat_col)) and pd.notna(row.get(erb_lon_col)):
                    lat, lon = float(row[erb_lat_col]), float(row[erb_lon_col])
                    x, y = self._latlon_to_meters(lat, lon)
                    self.erb_positions[erb_id] = np.array([x, y])
        
        logger.info(f"Found {len(self.erb_positions)} ERBs with positions")
        
        conn_file = data_path / "edges_conexao_pea_erb_v3.parquet"
        if conn_file.exists():
            df_connections = pd.read_parquet(conn_file)
        else:
            df_connections = pd.read_csv(data_path / "edges_conexao_pea_erb_v3.csv")
        
        source_col = 'source_id' if 'source_id' in df_connections.columns else 'id_pea'
        target_col = 'target_id' if 'target_id' in df_connections.columns else 'id_erb'
        
        self.connection_details = {}
        for _, row in df_connections.iterrows():
            pea_id = str(row[source_col])
            erb_id = str(row[target_col])
            
            if pea_id not in self.connections:
                self.connections[pea_id] = []
                self.connection_details[pea_id] = {}
            
            if erb_id in self.erb_positions:
                self.connections[pea_id].append(erb_id)
                self.connection_details[pea_id][erb_id] = {
                    'distancia_m': float(row['distancia_m']) if pd.notna(row.get('distancia_m')) else None,
                    'azimute': float(row['azimute_pea_erb']) if pd.notna(row.get('azimute_pea_erb')) else None,
                    'operadora': row.get('operadora_erb', '')
                }
        
        logger.info(f"Loaded connections for {len(self.connections)} PEAs")
        
        non_conn_file = data_path / "edges_nao_conexao_v3.parquet"
        if non_conn_file.exists():
            df_non_connections = pd.read_parquet(non_conn_file)
        else:
            non_conn_file = data_path / "edges_nao_conexao_v3.csv"
            if non_conn_file.exists():
                df_non_connections = pd.read_csv(non_conn_file)
            else:
                df_non_connections = None
        
        if df_non_connections is not None:
            
            self.non_connection_details = {}
            for _, row in df_non_connections.iterrows():
                pea_id = str(row[source_col])
                erb_id = str(row[target_col])
                
                if pea_id not in self.non_connections:
                    self.non_connections[pea_id] = []
                    self.non_connection_details[pea_id] = {}
                
                if erb_id in self.erb_positions:
                    self.non_connections[pea_id].append(erb_id)
                    self.non_connection_details[pea_id][erb_id] = {
                        'distancia_m': float(row['distancia_m']) if pd.notna(row.get('distancia_m')) else None,
                        'azimute': float(row['azimute_pea_erb']) if pd.notna(row.get('azimute_pea_erb')) else None,
                        'operadora': row.get('operadora_erb', '')
                    }
            
            logger.info(f"Loaded non-connections for {len(self.non_connections)} PEAs")
        
        self._load_triangulations(data_path)
        
        id_mapping_file = data_path / "id_mapping_v3.json"
        if id_mapping_file.exists():
            with open(id_mapping_file, 'r') as f:
                self.id_mapping = json.load(f)
            logger.info(f"Loaded ID mapping with {len(self.id_mapping)} entries")
        else:
            self.id_mapping = None
        
        self._compute_data_statistics()
    
    def _latlon_to_meters(self, lat: float, lon: float) -> Tuple[float, float]:
        ref_lat, ref_lon = -23.5505, -46.6333
        
        x = (lon - ref_lon) * 111000.0 * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * 111000.0
        
        return x, y
    
    def _load_triangulations(self, data_path: Path):
        logger.info("Loading pre-computed triangulations...")
        
        self.triangulations = {}
        
        tri_conn_file = data_path / "edges_intra_triangulo_conexao_v3.parquet"
        if tri_conn_file.exists():
            df_tri_conn = pd.read_parquet(tri_conn_file)
            logger.info(f"Loaded {len(df_tri_conn)} connection triangulations")
            
            for _, row in df_tri_conn.iterrows():
                tri_id = row.get('id_triangulo')
                if tri_id:
                    if tri_id not in self.triangulations:
                        self.triangulations[tri_id] = {
                            'type': 'connection',
                            'peas': set(),
                            'erbs': set()
                        }
                    self.triangulations[tri_id]['peas'].add(str(row['id_pea']))
                    self.triangulations[tri_id]['erbs'].add(str(row['id_erb']))
                    if 'id_erb_nc' in row:
                        self.triangulations[tri_id]['erbs'].add(str(row['id_erb_nc']))
        
        tri_non_conn_file = data_path / "edges_intra_triangulo_nao_conexao_v3.parquet"
        if tri_non_conn_file.exists():
            df_tri_non_conn = pd.read_parquet(tri_non_conn_file)
            logger.info(f"Loaded {len(df_tri_non_conn)} non-connection triangulations")
            
            for _, row in df_tri_non_conn.iterrows():
                tri_id = row.get('id_triangulo')
                if tri_id:
                    if tri_id not in self.triangulations:
                        self.triangulations[tri_id] = {
                            'type': 'non_connection',
                            'peas': set(),
                            'erbs': set()
                        }
                    self.triangulations[tri_id]['peas'].add(str(row['id_pea']))
                    self.triangulations[tri_id]['erbs'].add(str(row['id_erb']))
                    if 'id_erb_nc' in row:
                        self.triangulations[tri_id]['erbs'].add(str(row['id_erb_nc']))
        
        logger.info(f"Total triangulations loaded: {len(self.triangulations)}")
    
    def _use_real_distance(self, pea_id: str, erb_id: str, connection_type: str = 'connection') -> Optional[float]:
        if connection_type == 'connection' and pea_id in self.connection_details:
            if erb_id in self.connection_details[pea_id]:
                return self.connection_details[pea_id][erb_id].get('distancia_m')
        elif connection_type == 'non_connection' and pea_id in self.non_connection_details:
            if erb_id in self.non_connection_details[pea_id]:
                return self.non_connection_details[pea_id][erb_id].get('distancia_m')
        return None
    
    def _get_triangulation_erbs(self, pea_id: str) -> List[str]:
        erbs = []
        for tri_id, tri_data in self.triangulations.items():
            if pea_id in tri_data['peas']:
                erbs.extend(list(tri_data['erbs']))
        
        return list(set(erbs))
    
    def _compute_data_statistics(self):
        stats = {
            'total_peas': len(set(list(self.connections.keys()) + list(self.non_connections.keys()))),
            'peas_with_ground_truth': len(self.pea_positions),
            'total_erbs': len(self.erb_positions),
            'connections': {
                'total': sum(len(erbs) for erbs in self.connections.values()),
                'mean_per_pea': np.mean([len(erbs) for erbs in self.connections.values()]) if self.connections else 0,
                'max_per_pea': max([len(erbs) for erbs in self.connections.values()]) if self.connections else 0,
                'min_per_pea': min([len(erbs) for erbs in self.connections.values()]) if self.connections else 0
            },
            'non_connections': {
                'total': sum(len(erbs) for erbs in self.non_connections.values()),
                'mean_per_pea': np.mean([len(erbs) for erbs in self.non_connections.values()]) if self.non_connections else 0,
                'max_per_pea': max([len(erbs) for erbs in self.non_connections.values()]) if self.non_connections else 0,
                'min_per_pea': min([len(erbs) for erbs in self.non_connections.values()]) if self.non_connections else 0
            }
        }
        
        self.experiment_metadata['data_statistics'] = stats
        
        logger.info("Data Statistics:")
        logger.info(f"  Total PEAs: {stats['total_peas']}")
        logger.info(f"  PEAs with ground truth: {stats['peas_with_ground_truth']}")
        logger.info(f"  Total ERBs: {stats['total_erbs']}")
        logger.info(f"  Mean connections per PEA: {stats['connections']['mean_per_pea']:.2f}")
        logger.info(f"  Mean non-connections per PEA: {stats['non_connections']['mean_per_pea']:.2f}")
    
    def localize_weighted_trilateration(self, 
                                       pea_id: str,
                                       method: str = 'rssi') -> LocalizationResult:
        start_time = datetime.now()
        
        connected_erbs = self.connections.get(pea_id, [])
        non_connected_erbs = self.non_connections.get(pea_id, [])
        
        triangulation_erbs = self._get_triangulation_erbs(pea_id)
        if triangulation_erbs and len(triangulation_erbs) >= 3:
            logger.debug(f"Using pre-computed triangulation for {pea_id} with {len(triangulation_erbs)} ERBs")
            for erb_id in triangulation_erbs:
                if erb_id not in connected_erbs and erb_id in self.erb_positions:
                    non_connected_erbs.append(erb_id)
        
        if len(connected_erbs) == 0:
            return self._create_failed_result(pea_id, 'weighted_trilateration', method)
        
        primary_erb_pos = self.erb_positions[connected_erbs[0]]
        grid_size = 3000
        resolution = 20
        
        x_range = np.arange(primary_erb_pos[0] - grid_size/2, 
                           primary_erb_pos[0] + grid_size/2, 
                           resolution)
        y_range = np.arange(primary_erb_pos[1] - grid_size/2,
                           primary_erb_pos[1] + grid_size/2,
                           resolution)
        
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        probabilities = np.zeros(len(grid_points))
        
        for i, point in enumerate(grid_points):
            p_connected = 1.0
            for erb_id in connected_erbs:
                erb_pos = self.erb_positions[erb_id]
                distance = np.linalg.norm(point - erb_pos)
                
                real_distance = self._use_real_distance(pea_id, erb_id, 'connection')
                
                if method == 'rssi':
                    pl = self._path_loss_db(distance)
                    p_conn = self._connection_probability_rssi(pl)
                    
                    if real_distance is not None:
                        distance_error = abs(distance - real_distance) / real_distance
                        p_conn *= np.exp(-distance_error * 2)
                else:
                    if real_distance is not None:
                        distance_error = abs(distance - real_distance)
                        p_conn = np.exp(-distance_error / 100.0)
                    else:
                        max_range = self.propagation_params['rssi']['max_range_m']
                        p_conn = np.exp(-distance / (max_range / 3))
                
                p_connected *= p_conn
            
            p_not_connected = 1.0
            for erb_id in non_connected_erbs[:10]:
                if erb_id not in self.erb_positions:
                    continue
                erb_pos = self.erb_positions[erb_id]
                distance = np.linalg.norm(point - erb_pos)
                
                real_nc_distance = self._use_real_distance(pea_id, erb_id, 'non_connection')
                
                if method == 'rssi':
                    pl = self._path_loss_db(distance)
                    p_not_conn = 1 - self._connection_probability_rssi(pl)
                    
                    if real_nc_distance is not None and real_nc_distance < 2000:
                        if distance < real_nc_distance * 0.8:
                            p_not_conn *= 0.5
                else:
                    if real_nc_distance is not None:
                        if distance < real_nc_distance * 0.8:
                            p_not_conn = 0.3
                        else:
                            p_not_conn = 0.9
                    else:
                        max_range = self.propagation_params['rssi']['max_range_m']
                        p_not_conn = 1 - np.exp(-distance / (max_range / 3))
                
                p_not_connected *= p_not_conn
            
            probabilities[i] = p_connected * p_not_connected
        
        best_idx = np.argmax(probabilities)
        estimated_position = grid_points[best_idx]
        confidence_score = float(probabilities[best_idx] / np.sum(probabilities))
        
        error_m = None
        ground_truth = None
        if pea_id in self.pea_positions:
            ground_truth = self.pea_positions[pea_id]
            error_m = float(np.linalg.norm(estimated_position - ground_truth))
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        probability_map = probabilities.reshape(len(y_range), len(x_range))
        
        return LocalizationResult(
            pea_id=pea_id,
            method=method,
            algorithm='weighted_trilateration',
            estimated_position=estimated_position,
            ground_truth_position=ground_truth,
            error_m=error_m,
            confidence_score=confidence_score,
            computation_time_ms=computation_time,
            num_connected_erbs=len(connected_erbs),
            num_non_connected_erbs=len(non_connected_erbs),
            convergence_info={
                'grid_size': grid_size,
                'resolution': resolution,
                'max_probability': float(np.max(probabilities)),
                'entropy': float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
            },
            probability_map=probability_map,
            metadata={
                'grid_bounds': {
                    'x_min': float(x_range[0]),
                    'x_max': float(x_range[-1]),
                    'y_min': float(y_range[0]),
                    'y_max': float(y_range[-1])
                }
            }
        )
    
    def localize_mle(self, pea_id: str, method: str = 'rssi') -> LocalizationResult:
        start_time = datetime.now()
        
        connected_erbs = self.connections.get(pea_id, [])
        non_connected_erbs = self.non_connections.get(pea_id, [])
        
        if len(connected_erbs) == 0:
            return self._create_failed_result(pea_id, 'mle', method)
        
        def neg_log_likelihood(pos):
            x, y = pos
            point = np.array([x, y])
            
            ll = 0.0
            
            for erb_id in connected_erbs:
                erb_pos = self.erb_positions[erb_id]
                distance = np.linalg.norm(point - erb_pos)
                
                if method == 'rssi':
                    pl = self._path_loss_db(distance)
                    expected_rssi = self.propagation_params['rssi']['reference_power_dbm'] - pl
                    std = self.propagation_params['rssi']['shadowing_std_db']
                    ll += norm.logpdf(-30, expected_rssi, std)
                else:
                    ll += -distance / 1000.0
            
            for erb_id in non_connected_erbs[:5]:
                erb_pos = self.erb_positions[erb_id]
                distance = np.linalg.norm(point - erb_pos)
                
                if method == 'rssi':
                    pl = self._path_loss_db(distance)
                    expected_rssi = self.propagation_params['rssi']['reference_power_dbm'] - pl
                    std = self.propagation_params['rssi']['shadowing_std_db']
                    ll += norm.logcdf(-50, expected_rssi, std)
                else:
                    ll += distance / 2000.0
            
            return -ll
        
        primary_erb_pos = self.erb_positions[connected_erbs[0]]
        results = []
        
        for _ in range(self.algorithm_configs['mle']['initial_points']):
            initial_guess = primary_erb_pos + np.random.randn(2) * 500
            
            bounds = [(primary_erb_pos[0] - 3000, primary_erb_pos[0] + 3000),
                     (primary_erb_pos[1] - 3000, primary_erb_pos[1] + 3000)]
            
            result = minimize(neg_log_likelihood, 
                            initial_guess,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': self.algorithm_configs['mle']['max_iterations']})
            
            results.append(result)
        
        best_result = min(results, key=lambda r: r.fun)
        estimated_position = best_result.x
        
        try:
            from scipy.optimize import approx_fprime
            hessian = np.zeros((2, 2))
            eps = 1e-5
            for i in range(2):
                for j in range(2):
                    e_i = np.zeros(2)
                    e_i[i] = eps
                    e_j = np.zeros(2)
                    e_j[j] = eps
                    
                    hessian[i, j] = (neg_log_likelihood(estimated_position + e_i + e_j) -
                                    neg_log_likelihood(estimated_position + e_i - e_j) -
                                    neg_log_likelihood(estimated_position - e_i + e_j) +
                                    neg_log_likelihood(estimated_position - e_i - e_j)) / (4 * eps * eps)
            
            cov_matrix = np.linalg.inv(hessian)
            confidence_score = 1.0 / (1.0 + np.trace(cov_matrix) / 1000.0)
        except:
            confidence_score = 0.5
            cov_matrix = np.eye(2) * 100
        
        error_m = None
        ground_truth = None
        if pea_id in self.pea_positions:
            ground_truth = self.pea_positions[pea_id]
            error_m = float(np.linalg.norm(estimated_position - ground_truth))
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return LocalizationResult(
            pea_id=pea_id,
            method=method,
            algorithm='mle',
            estimated_position=estimated_position,
            ground_truth_position=ground_truth,
            error_m=error_m,
            confidence_score=float(confidence_score),
            computation_time_ms=computation_time,
            num_connected_erbs=len(connected_erbs),
            num_non_connected_erbs=len(non_connected_erbs),
            convergence_info={
                'success': bool(best_result.success),
                'iterations': int(best_result.nit),
                'function_evaluations': int(best_result.nfev),
                'final_likelihood': float(-best_result.fun),
                'covariance_trace': float(np.trace(cov_matrix))
            },
            metadata={
                'optimization_message': best_result.message
            }
        )
    
    def localize_particle_filter(self, pea_id: str, method: str = 'rssi') -> LocalizationResult:
        start_time = datetime.now()
        
        connected_erbs = self.connections.get(pea_id, [])
        non_connected_erbs = self.non_connections.get(pea_id, [])
        
        if len(connected_erbs) == 0:
            return self._create_failed_result(pea_id, 'particle_filter', method)
        
        num_particles = self.algorithm_configs['particle_filter']['num_particles']
        primary_erb_pos = self.erb_positions[connected_erbs[0]]
        
        particles = primary_erb_pos + np.random.randn(num_particles, 2) * 1000
        weights = np.ones(num_particles) / num_particles
        
        for iteration in range(10):
            for i, particle in enumerate(particles):
                weight = 1.0
                
                for erb_id in connected_erbs:
                    erb_pos = self.erb_positions[erb_id]
                    distance = np.linalg.norm(particle - erb_pos)
                    
                    if method == 'rssi':
                        pl = self._path_loss_db(distance)
                        prob = self._connection_probability_rssi(pl)
                    else:
                        max_range = self.propagation_params['rssi']['max_range_m']
                        prob = np.exp(-distance / (max_range / 3))
                    
                    weight *= prob
                
                for erb_id in non_connected_erbs[:5]:
                    erb_pos = self.erb_positions[erb_id]
                    distance = np.linalg.norm(particle - erb_pos)
                    
                    if method == 'rssi':
                        pl = self._path_loss_db(distance)
                        prob = 1 - self._connection_probability_rssi(pl)
                    else:
                        max_range = self.propagation_params['rssi']['max_range_m']
                        prob = 1 - np.exp(-distance / (max_range / 3))
                    
                    weight *= prob
                
                weights[i] = weight
            
            weights /= np.sum(weights)
            
            n_eff = 1.0 / np.sum(weights ** 2)
            if n_eff < num_particles * self.algorithm_configs['particle_filter']['resampling_threshold']:
                indices = np.random.choice(num_particles, num_particles, p=weights)
                particles = particles[indices]
                weights = np.ones(num_particles) / num_particles
            
            noise_std = self.algorithm_configs['particle_filter']['noise_std_m'] * (0.9 ** iteration)
            particles += np.random.randn(num_particles, 2) * noise_std
        
        estimated_position = np.average(particles, weights=weights, axis=0)
        
        cov = np.cov(particles.T, aweights=weights)
        confidence_score = 1.0 / (1.0 + np.trace(cov) / 1000.0)
        
        error_m = None
        ground_truth = None
        if pea_id in self.pea_positions:
            ground_truth = self.pea_positions[pea_id]
            error_m = float(np.linalg.norm(estimated_position - ground_truth))
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return LocalizationResult(
            pea_id=pea_id,
            method=method,
            algorithm='particle_filter',
            estimated_position=estimated_position,
            ground_truth_position=ground_truth,
            error_m=error_m,
            confidence_score=float(confidence_score),
            computation_time_ms=computation_time,
            num_connected_erbs=len(connected_erbs),
            num_non_connected_erbs=len(non_connected_erbs),
            convergence_info={
                'num_particles': num_particles,
                'effective_sample_size': float(n_eff),
                'particle_spread': float(np.std(particles)),
                'covariance_trace': float(np.trace(cov))
            },
            metadata={
                'final_particle_positions': particles[:10].tolist()
            }
        )
    
    def _path_loss_db(self, distance_m: float) -> float:
        if distance_m < 1.0:
            distance_m = 1.0
        
        d0 = 1.0
        n = self.propagation_params['rssi']['path_loss_exponent']
        
        pl = 10 * n * np.log10(distance_m / d0)
        
        return pl
    
    def _connection_probability_rssi(self, path_loss_db: float) -> float:
        reference_power = self.propagation_params['rssi']['reference_power_dbm']
        rssi = reference_power - path_loss_db
        
        connection_threshold = -85
        
        prob = 1.0 / (1.0 + np.exp(-(rssi - connection_threshold) / 5))
        
        return prob
    
    def _create_failed_result(self, pea_id: str, algorithm: str, method: str) -> LocalizationResult:
        return LocalizationResult(
            pea_id=pea_id,
            method=method,
            algorithm=algorithm,
            estimated_position=np.array([0.0, 0.0]),
            ground_truth_position=self.pea_positions.get(pea_id),
            error_m=None,
            confidence_score=0.0,
            computation_time_ms=0.0,
            num_connected_erbs=len(self.connections.get(pea_id, [])),
            num_non_connected_erbs=len(self.non_connections.get(pea_id, [])),
            convergence_info={'failed': True},
            metadata={'failure_reason': 'No connections available'}
        )
    
    def run_experiments(self, 
                       methods: List[str] = ['rssi', 'toa', 'tdoa'],
                       algorithms: List[str] = ['weighted_trilateration', 'mle', 'particle_filter'],
                       sample_size: Optional[int] = None):
        logger.info("="*80)
        logger.info("STARTING ENHANCED CLASSICAL METHODS EXPERIMENTS")
        logger.info("="*80)
        
        all_peas = list(set(list(self.connections.keys()) + list(self.non_connections.keys())))
        
        if sample_size and sample_size < len(all_peas):
            peas_with_gt = [p for p in all_peas if p in self.pea_positions]
            peas_without_gt = [p for p in all_peas if p not in self.pea_positions]
            
            n_with_gt = min(len(peas_with_gt), int(sample_size * 0.8))
            n_without_gt = min(len(peas_without_gt), sample_size - n_with_gt)
            
            selected_peas = (np.random.choice(peas_with_gt, n_with_gt, replace=False).tolist() +
                           np.random.choice(peas_without_gt, n_without_gt, replace=False).tolist())
        else:
            selected_peas = all_peas
        
        logger.info(f"Testing {len(selected_peas)} PEAs")
        logger.info(f"Methods: {methods}")
        logger.info(f"Algorithms: {algorithms}")
        
        total_experiments = len(selected_peas) * len(methods) * len(algorithms)
        completed = 0
        
        for method in methods:
            for algorithm in algorithms:
                logger.info(f"\n{'-'*60}")
                logger.info(f"Running {algorithm} with {method}")
                logger.info(f"{'-'*60}")
                
                for i, pea_id in enumerate(selected_peas):
                    if algorithm == 'weighted_trilateration':
                        result = self.localize_weighted_trilateration(pea_id, method)
                    elif algorithm == 'mle':
                        result = self.localize_mle(pea_id, method)
                    elif algorithm == 'particle_filter':
                        result = self.localize_particle_filter(pea_id, method)
                    else:
                        continue
                    
                    self.results.append(result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        progress = (completed / total_experiments) * 100
                        logger.info(f"Progress: {completed}/{total_experiments} ({progress:.1f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENTS COMPLETED")
        logger.info("="*80)
        
        self._compute_metrics()
        
        self._save_results()
        
        self._generate_visualizations()
    
    def _compute_metrics(self):
        logger.info("\nComputing scientific metrics...")
        
        metrics_by_config = {}
        
        for result in self.results:
            config_key = f"{result.method}_{result.algorithm}"
            
            if config_key not in metrics_by_config:
                metrics_by_config[config_key] = []
            
            if result.error_m is not None:
                metrics_by_config[config_key].append(result)
        
        self.metrics = {}
        
        for config_key, results in metrics_by_config.items():
            if len(results) == 0:
                continue
            
            errors = [r.error_m for r in results if r.error_m is not None]
            
            if len(errors) == 0:
                continue
            
            errors_array = np.array(errors)
            
            mean_error = float(np.mean(errors_array))
            median_error = float(np.median(errors_array))
            std_error = float(np.std(errors_array))
            rmse = float(np.sqrt(np.mean(errors_array ** 2)))
            mae = float(np.mean(np.abs(errors_array)))
            
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95, 99]:
                percentiles[p] = float(np.percentile(errors_array, p))
            
            thresholds = [50, 100, 200, 300, 500, 1000, 2000]
            accuracy_at_thresholds = {}
            for t in thresholds:
                accuracy_at_thresholds[t] = float(np.mean(errors_array <= t) * 100)
            
            if len(errors) > 30:
                _, p_value_shapiro = stats.shapiro(errors_array[:5000])
                anderson_result = anderson(errors_array)
                
                normality_test = {
                    'shapiro_p_value': float(p_value_shapiro),
                    'is_normal_shapiro': p_value_shapiro > 0.05,
                    'anderson_statistic': float(anderson_result.statistic),
                    'anderson_critical_values': anderson_result.critical_values.tolist()
                }
            else:
                normality_test = {'sample_size_too_small': True}
            
            q1, q3 = np.percentile(errors_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = errors_array[(errors_array < lower_bound) | (errors_array > upper_bound)]
            
            outliers_info = {
                'num_outliers': len(outliers),
                'percentage': float(len(outliers) / len(errors) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            convergence_rate = float(np.mean([r.convergence_info.get('success', True) 
                                             for r in results]))
            failed_cases = sum(1 for r in results 
                             if r.convergence_info.get('failed', False))
            
            consistency_score = float(np.mean([r.confidence_score for r in results]))
            
            if len(results) > 10:
                positions = np.array([r.estimated_position for r in results])
                errors_for_coherence = np.array([r.error_m for r in results 
                                                if r.error_m is not None])
                
                if len(positions) > 1 and len(errors_for_coherence) > 1:
                    distances_matrix = cdist(positions, positions)
                    error_diffs = np.abs(errors_for_coherence[:, np.newaxis] - errors_for_coherence)
                    
                    mask = distances_matrix > 0
                    if np.any(mask):
                        correlation = np.corrcoef(distances_matrix[mask].flatten(),
                                                 error_diffs[mask].flatten())[0, 1]
                        spatial_coherence = float(1 - abs(correlation))
                    else:
                        spatial_coherence = 0.5
                else:
                    spatial_coherence = 0.5
            else:
                spatial_coherence = 0.5
            
            confidence_intervals = {}
            if len(errors) > 30:
                n_bootstrap = 1000
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(errors_array, len(errors_array), replace=True)
                    bootstrap_means.append(np.mean(sample))
                
                confidence_intervals['mean_95ci'] = (
                    float(np.percentile(bootstrap_means, 2.5)),
                    float(np.percentile(bootstrap_means, 97.5))
                )
            else:
                from scipy import stats
                se = std_error / np.sqrt(len(errors))
                ci = stats.t.interval(0.95, len(errors)-1, loc=mean_error, scale=se)
                confidence_intervals['mean_95ci'] = (float(ci[0]), float(ci[1]))
            
            self.metrics[config_key] = ExperimentMetrics(
                mean_error=mean_error,
                median_error=median_error,
                std_error=std_error,
                rmse=rmse,
                mae=mae,
                percentiles=percentiles,
                accuracy_at_thresholds=accuracy_at_thresholds,
                normality_test=normality_test,
                outliers=outliers_info,
                convergence_rate=convergence_rate,
                failed_cases=failed_cases,
                consistency_score=consistency_score,
                spatial_coherence=spatial_coherence,
                confidence_intervals=confidence_intervals
            )
            
            logger.info(f"\n{config_key}:")
            logger.info(f"  MAE: {mae:.2f}m")
            logger.info(f"  RMSE: {rmse:.2f}m")
            logger.info(f"  Median: {median_error:.2f}m")
            logger.info(f"  Acc@100m: {accuracy_at_thresholds.get(100, 0):.1f}%")
            logger.info(f"  Acc@500m: {accuracy_at_thresholds.get(500, 0):.1f}%")
    
    def _save_results(self):
        logger.info("\nSaving results...")
        
        results_data = {
            'metadata': self.experiment_metadata,
            'timestamp': datetime.now().isoformat(),
            'results': [],
            'metrics': {},
            'summary': {}
        }
        
        for result in self.results:
            result_dict = {
                'pea_id': result.pea_id,
                'method': result.method,
                'algorithm': result.algorithm,
                'estimated_position': result.estimated_position.tolist(),
                'ground_truth_position': result.ground_truth_position.tolist() if result.ground_truth_position is not None else None,
                'error_m': result.error_m,
                'confidence_score': result.confidence_score,
                'computation_time_ms': result.computation_time_ms,
                'num_connected_erbs': result.num_connected_erbs,
                'num_non_connected_erbs': result.num_non_connected_erbs,
                'convergence_info': result.convergence_info,
                'metadata': result.metadata
            }
            results_data['results'].append(result_dict)
        
        for config_key, metrics in self.metrics.items():
            results_data['metrics'][config_key] = {
                'mean_error': metrics.mean_error,
                'median_error': metrics.median_error,
                'std_error': metrics.std_error,
                'rmse': metrics.rmse,
                'mae': metrics.mae,
                'percentiles': metrics.percentiles,
                'accuracy_at_thresholds': metrics.accuracy_at_thresholds,
                'normality_test': metrics.normality_test,
                'outliers': metrics.outliers,
                'convergence_rate': metrics.convergence_rate,
                'failed_cases': metrics.failed_cases,
                'consistency_score': metrics.consistency_score,
                'spatial_coherence': metrics.spatial_coherence,
                'confidence_intervals': metrics.confidence_intervals
            }
        
        if self.metrics:
            best_config = min(self.metrics.items(), key=lambda x: x[1].mae)
            results_data['summary'] = {
                'best_configuration': best_config[0],
                'best_mae': best_config[1].mae,
                'best_rmse': best_config[1].rmse,
                'total_experiments': len(self.results),
                'experiments_with_ground_truth': sum(1 for r in self.results if r.error_m is not None)
            }
        
        output_file = self.data_dir / 'enhanced_classical_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
        summary_file = self.data_dir / 'results_summary.json'
        summary_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'approach': 'Connection-Aware Trilateration (CAT)',
                'total_peas': results_data['metadata']['data_statistics']['total_peas'],
                'total_erbs': results_data['metadata']['data_statistics']['total_erbs']
            },
            'metrics_by_configuration': results_data['metrics'],
            'best_configuration': results_data['summary']
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")
    
    def _generate_visualizations(self):
        logger.info("\nGenerating scientific visualizations...")
        
        self._plot_comparison_boxplot()
        
        self._plot_cdf_curves()
        
        self._plot_spatial_error_heatmap()
        
        self._plot_convergence_analysis()
        
        self._plot_probability_maps()
        
        logger.info("Visualizations completed")
    
    def _plot_comparison_boxplot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['rssi', 'toa', 'tdoa']
        algorithms = ['weighted_trilateration', 'mle', 'particle_filter']
        
        for i, method in enumerate(methods):
            ax = axes[i]
            
            data_to_plot = []
            labels = []
            
            for algorithm in algorithms:
                config_key = f"{method}_{algorithm}"
                errors = [r.error_m for r in self.results 
                         if r.method == method and r.algorithm == algorithm and r.error_m is not None]
                
                if errors:
                    data_to_plot.append(errors)
                    labels.append(algorithm.replace('_', '\n'))
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                colors = ['lightblue', 'lightgreen', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_ylabel('Error (m)')
                ax.set_title(f'{method.upper()}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, min(2000, ax.get_ylim()[1]))
        
        plt.suptitle('Localization Error Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.figures_dir / 'comparison_boxplot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_file}")
    
    def _plot_cdf_curves(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for config_key in self.metrics.keys():
            errors = [r.error_m for r in self.results 
                     if f"{r.method}_{r.algorithm}" == config_key and r.error_m is not None]
            
            if errors:
                sorted_errors = np.sort(errors)
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                
                label = config_key.replace('_', ' ').title()
                ax.plot(sorted_errors, cdf, label=label, linewidth=2)
        
        ax.set_xlabel('Localization Error (m)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function of Localization Errors')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 2000)
        
        for threshold in [100, 500, 1000]:
            ax.axvline(threshold, color='gray', linestyle='--', alpha=0.5)
            ax.text(threshold, 0.05, f'{threshold}m', rotation=90, va='bottom')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / 'cdf_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_file}")
    
    def _plot_spatial_error_heatmap(self):
        if not self.metrics:
            return
        
        best_config = min(self.metrics.items(), key=lambda x: x[1].mae)[0]
        
        results = [r for r in self.results 
                  if f"{r.method}_{r.algorithm}" == best_config and r.error_m is not None]
        
        if len(results) < 10:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        positions = np.array([r.estimated_position for r in results])
        errors = np.array([r.error_m for r in results])
        
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        
        from scipy.interpolate import griddata
        grid_errors = griddata(positions, errors, (grid_x, grid_y), method='cubic')
        
        im = ax.contourf(grid_x, grid_y, grid_errors, levels=20, cmap='RdYlGn_r')
        plt.colorbar(im, ax=ax, label='Error (m)')
        
        for erb_id, erb_pos in list(self.erb_positions.items())[:50]:
            ax.plot(erb_pos[0], erb_pos[1], 'k^', markersize=8, alpha=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Spatial Error Distribution - {best_config}')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / 'spatial_error_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_file}")
    
    def _plot_convergence_analysis(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        
        algorithms = ['weighted_trilateration', 'mle', 'particle_filter']
        colors = ['blue', 'green', 'orange']
        
        for algorithm, color in zip(algorithms, colors):
            times = [r.computation_time_ms for r in self.results if r.algorithm == algorithm]
            
            if times:
                ax1.hist(times, bins=30, alpha=0.6, label=algorithm.replace('_', ' ').title(), 
                        color=color, edgecolor='black')
        
        ax1.set_xlabel('Computation Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Computational Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        
        convergence_data = []
        labels = []
        
        for config_key, metrics in self.metrics.items():
            convergence_data.append(metrics.convergence_rate * 100)
            labels.append(config_key.replace('_', '\n'))
        
        if convergence_data:
            bars = ax2.bar(range(len(convergence_data)), convergence_data)
            ax2.set_xticks(range(len(convergence_data)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_ylabel('Convergence Rate (%)')
            ax2.set_title('Algorithm Convergence Success Rate')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for i, bar in enumerate(bars):
                if convergence_data[i] > 90:
                    bar.set_color('green')
                elif convergence_data[i] > 70:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / 'convergence_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_file}")
    
    def _plot_probability_maps(self):
        results_with_maps = [r for r in self.results 
                            if r.probability_map is not None and r.error_m is not None]
        
        if len(results_with_maps) == 0:
            return
        
        results_with_maps.sort(key=lambda r: r.error_m)
        
        cases = {
            'Best Case': results_with_maps[0],
            'Median Case': results_with_maps[len(results_with_maps)//2],
            'Worst Case': results_with_maps[-1]
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (title, result) in enumerate(cases.items()):
            ax = axes[i]
            
            im = ax.imshow(result.probability_map, cmap='hot', origin='lower', 
                          extent=result.metadata['grid_bounds'].values())
            
            ax.plot(result.estimated_position[0], result.estimated_position[1], 
                   'b*', markersize=15, label='Estimated')
            
            if result.ground_truth_position is not None:
                ax.plot(result.ground_truth_position[0], result.ground_truth_position[1],
                       'g^', markersize=12, label='Ground Truth')
            
            for erb_id in self.connections.get(result.pea_id, [])[:3]:
                if erb_id in self.erb_positions:
                    erb_pos = self.erb_positions[erb_id]
                    ax.plot(erb_pos[0], erb_pos[1], 'ko', markersize=8, alpha=0.7)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{title}\nError: {result.error_m:.1f}m')
            ax.legend()
            
            plt.colorbar(im, ax=ax, label='Probability')
        
        plt.suptitle('Probability Maps - Weighted Trilateration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.figures_dir / 'probability_maps.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_file}")


def main():
    logger.info("="*80)
    logger.info("ENHANCED CLASSICAL LOCALIZATION METHODS")
    logger.info("Connection-Aware Trilateration (CAT)")
    logger.info("Using REAL DATA - No simulations")
    logger.info("="*80)
    
    localizer = EnhancedClassicalLocalizer(output_dir="TESTE_MODELOS_CLASSICOS/enhanced_results")
    
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "NORMALIZACAO_OUTPUT"
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Loading data from: {data_dir}")
    
    localizer.load_data(str(data_dir))
    
    localizer.run_experiments(
        methods=['rssi', 'toa', 'tdoa'],
        algorithms=['weighted_trilateration', 'mle', 'particle_filter'],
        sample_size=None
    )
    
    logger.info("\n" + "="*80)
    logger.info("ENHANCED CLASSICAL METHODS COMPLETED")
    logger.info("Results saved in: TESTE_MODELOS_CLASSICOS/enhanced_results/")
    logger.info("="*80)


if __name__ == "__main__":
    from scipy import stats
    
    main()
