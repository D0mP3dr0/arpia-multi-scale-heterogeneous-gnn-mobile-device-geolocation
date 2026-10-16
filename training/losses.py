#!/usr/bin/env python3
"""Physics-informed loss of ARPIA, including the non-connection force terms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import math

try:
    from numba import cuda, jit, prange
    import numba as nb
    NUMBA_AVAILABLE = True
    
    try:
        CUDA_AVAILABLE = cuda.is_available()
        if CUDA_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.info('🚀 Numba CUDA available - GPU acceleration enabled!')
        else:
            CUDA_AVAILABLE = False
            logger = logging.getLogger(__name__)
            logger.info('🔶 Numba available (CPU) - GPU not available')
    except Exception as e:
        CUDA_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.info(f"ℹ️ Numba CUDA not available: {e} - Using accelerated Numba CPU")
        
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning('⚠️ Numba not available - using plain CPU')

logger = logging.getLogger(__name__)


class PhysicsInformedLoss(nn.Module):
    
    def __init__(self, 
                 position_weight=1.0,
                 physics_weight=0.5,
                 connection_weight=1.0,
                 non_connection_weight=1.0,
                 triangular_weight=0.5,
                 smoothness_weight=0.1,
                 feature_reconstruction_weight=0.0,
                 movement_penalty_weight=0.1,
                 spatial_dispersion_weight=0.2,
                 precision_weight=2.0,
                 boundary_weight=0.5,
                 topology_weight=0.3,
                 local_precision_weight=1.0,
                 adaptive_threshold_weight=0.8,
                 rf_physics_weight=1.5,
                 use_weight_annealing=True,
                 use_haversine=True,
                 adaptive_thresholds=True,
                 use_numba_acceleration=True):
        super().__init__()
        
        self.base_weights = {
            'position_weight': min(15.0, position_weight),
            'physics_weight': physics_weight,
            'connection_weight': max(2.0, connection_weight),
            'non_connection_weight': max(3.0, non_connection_weight),
            'triangular_weight': triangular_weight,
            'smoothness_weight': smoothness_weight,
            'feature_reconstruction_weight': feature_reconstruction_weight,
            'movement_penalty_weight': movement_penalty_weight,
            'spatial_dispersion_weight': spatial_dispersion_weight,
            'precision_weight': precision_weight,
            'boundary_weight': boundary_weight,
            'topology_weight': topology_weight,
            'local_precision_weight': local_precision_weight,
            'adaptive_threshold_weight': adaptive_threshold_weight,
            'rf_physics_weight': rf_physics_weight,
        }
        
        self.current_weights = self.base_weights.copy()
        
        self.rf_physics_weight = rf_physics_weight
        self.position_weight = position_weight
        self.physics_weight = physics_weight
        self.connection_weight = connection_weight
        self.non_connection_weight = non_connection_weight
        self.triangular_weight = triangular_weight
        self.smoothness_weight = smoothness_weight
        self.feature_reconstruction_weight = feature_reconstruction_weight
        self.movement_penalty_weight = movement_penalty_weight
        self.spatial_dispersion_weight = spatial_dispersion_weight
        self.precision_weight = precision_weight
        self.boundary_weight = boundary_weight
        self.topology_weight = topology_weight
        self.local_precision_weight = local_precision_weight
        self.adaptive_threshold_weight = adaptive_threshold_weight
        
        self.use_weight_annealing = use_weight_annealing
        self.use_haversine = use_haversine
        self.adaptive_thresholds = adaptive_thresholds
        self.use_numba_acceleration = use_numba_acceleration and NUMBA_AVAILABLE
        
        self.position_annealing_factor = 1.0
        self.geometric_loss_history = []
        self.error_history = []
        self.previous_positions = None
        
        self.connection_distances_cache = {}
        self.adaptive_threshold_cache = {}
        
        self.log_verbose = True
        self._log_precision_stats = True
        
        self.rf_frequency = 2.4e9
        self.speed_of_light = 299792458
        self.earth_radius = 6371000
        
        logger.info('🚀 ULTRA-ADVANCED PhysicsInformedLoss initialized!')
        logger.info(f"   📊 {len(self.base_weights)} componentes de loss")
        logger.info(f"   🔬 RF physics enabled: {self.rf_physics_weight}")
        logger.info(f"   ⚡ Numba: {'✅' if self.use_numba_acceleration else '❌'}")
        logger.info(f"   🎯 Adaptive thresholds: {'✅' if self.adaptive_thresholds else '❌'}")
    
    def update_weights(self, weight_dict: Dict[str, float]):
        for key, value in weight_dict.items():
            if key in self.base_weights:
                self.base_weights[key] = value
                self.current_weights[key] = value
                
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def apply_weight_annealing(self, epoch, geometric_loss):
        if not self.use_weight_annealing:
            return
            
        self.geometric_loss_history.append(geometric_loss)
        if len(self.geometric_loss_history) > 10:
            self.geometric_loss_history.pop(0)
        
        avg_geometric_loss = sum(self.geometric_loss_history) / len(self.geometric_loss_history)
        
        if avg_geometric_loss < 0.1 and len(self.geometric_loss_history) >= 8:
            self.position_annealing_factor *= 0.9999
            
            if self.position_annealing_factor < 0.33:
                self.position_annealing_factor = 0.33
            
            self.current_weights['position_weight'] = (
                self.base_weights['position_weight'] * self.position_annealing_factor
            )
            
            if self.current_weights['position_weight'] < 5.0:
                self.current_weights['position_weight'] = 5.0
            elif self.current_weights['position_weight'] > 15.0:
                self.current_weights['position_weight'] = 15.0
            
            if epoch % 10 == 0:
                logger.info(f"[ANNEALING] Epoch {epoch}: "
                           f"geometric_loss={avg_geometric_loss:.3f}, "
                           f"position_weight={self.current_weights['position_weight']:.3f}")
    
    def haversine_distance(self, pos1, pos2):
        if pos1.dim() == 1:
            pos1 = pos1.unsqueeze(0)
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)
        
        coord_range = max(pos1.max() - pos1.min(), pos2.max() - pos2.min())
        if coord_range < 10:
            euclidean_dist = torch.norm(pos1 - pos2, dim=1)
            return euclidean_dist * 450.0
        
        lat1, lon1 = pos1[:, 0], pos1[:, 1]
        lat2, lon2 = pos2[:, 0], pos2[:, 1]
        
        lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (torch.sin(dlat/2)**2 + 
             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2)
        c = 2 * torch.asin(torch.sqrt(a))
        
        return self.earth_radius * c
    
    def position_loss(self, pred_locations, true_locations):
        if self.use_haversine:
            distances = self.haversine_distance(pred_locations, true_locations)
        else:
            distances = torch.norm(pred_locations - true_locations, dim=1)
            
            coord_range = pred_locations.max() - pred_locations.min()
            if coord_range < 10:
                distances = distances * 450.0
        
        return torch.mean(distances)
    
    def connection_consistency_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            
            if not hasattr(batch, 'edge_index_dict'):
                return self._safe_tensor(0.0, device)
            
            connection_type = ('PEA', 'CONEXAO', 'ERB')
            if connection_type not in batch.edge_index_dict:
                return self._safe_tensor(0.0, device)
            
            edge_index = batch.edge_index_dict[connection_type]
            if edge_index.size(1) == 0:
                return self._safe_tensor(0.0, device)
            
            pea_positions = pred_locations
            erb_positions = batch.x_dict.get('ERB', None)
            
            if erb_positions is None:
                return self._safe_tensor(0.0, device)
            
            erb_positions = erb_positions[:, :2]
            
            pea_indices = edge_index[0]
            erb_indices = edge_index[1]
            
            if pea_indices.max() >= len(pea_positions) or erb_indices.max() >= len(erb_positions):
                return self._safe_tensor(0.0, device)
            
            pea_indices = torch.clamp(pea_indices, 0, len(pea_positions) - 1)
            erb_indices = torch.clamp(erb_indices, 0, len(erb_positions) - 1)
            
            connected_pea_pos = pea_positions[pea_indices]
            connected_erb_pos = erb_positions[erb_indices]
            
            if torch.isnan(connected_pea_pos).any() or torch.isnan(connected_erb_pos).any():
                return self._safe_tensor(0.0, device)
            
            if self.use_haversine:
                try:
                    connection_distances = self.haversine_distance(connected_pea_pos, connected_erb_pos)
                except:
                    connection_distances = torch.norm(connected_pea_pos - connected_erb_pos, dim=1) * 450.0
            else:
                connection_distances = torch.norm(connected_pea_pos - connected_erb_pos, dim=1)
                
                coord_range = pred_locations.max() - pred_locations.min()
                if coord_range < 10:
                    connection_distances = connection_distances * 450.0
            
            if torch.isnan(connection_distances).any() or torch.isinf(connection_distances).any():
                return self._safe_tensor(0.0, device)
            
            if self.adaptive_thresholds:
                adaptive_threshold = torch.mean(connection_distances) * 1.2
                adaptive_threshold = torch.clamp(adaptive_threshold, 200.0, 800.0)
            else:
                adaptive_threshold = 500.0
            
            excess_distances = torch.relu(connection_distances - adaptive_threshold)
            loss = torch.mean(excess_distances ** 2) / 10000.0
            
            if torch.isnan(loss) or torch.isinf(loss):
                return self._safe_tensor(0.0, device)
            
            return loss
            
        except Exception as e:
            logger.warning(f"Error in connection_consistency_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def non_connection_penalty(self, pred_locations, batch):
        try:
            device = pred_locations.device
            
            if not hasattr(batch, 'edge_index_dict'):
                return self._safe_tensor(0.0, device)
            
            non_connection_type = ('PEA', 'NAO_CONEXAO', 'ERB')
            if non_connection_type not in batch.edge_index_dict:
                return self._safe_tensor(0.0, device)
            
            edge_index = batch.edge_index_dict[non_connection_type]
            if edge_index.size(1) == 0:
                return self._safe_tensor(0.0, device)
            
            pea_positions = pred_locations
            erb_positions = batch.x_dict.get('ERB', None)
            
            if erb_positions is None:
                return self._safe_tensor(0.0, device)
            
            erb_positions = erb_positions[:, :2]
            
            pea_indices = edge_index[0]
            erb_indices = edge_index[1]
            
            if pea_indices.max() >= len(pea_positions) or erb_indices.max() >= len(erb_positions):
                return self._safe_tensor(0.0, device)
            
            pea_indices = torch.clamp(pea_indices, 0, len(pea_positions) - 1)
            erb_indices = torch.clamp(erb_indices, 0, len(erb_positions) - 1)
            
            non_connected_pea_pos = pea_positions[pea_indices]
            non_connected_erb_pos = erb_positions[erb_indices]
            
            if torch.isnan(non_connected_pea_pos).any() or torch.isnan(non_connected_erb_pos).any():
                return self._safe_tensor(0.0, device)
            
            if self.use_haversine:
                try:
                    non_connection_distances = self.haversine_distance(non_connected_pea_pos, non_connected_erb_pos)
                except:
                    non_connection_distances = torch.norm(non_connected_pea_pos - non_connected_erb_pos, dim=1) * 450.0
            else:
                non_connection_distances = torch.norm(non_connected_pea_pos - non_connected_erb_pos, dim=1)
                
                coord_range = pred_locations.max() - pred_locations.min()
                if coord_range < 10:
                    non_connection_distances = non_connection_distances * 450.0
            
            if torch.isnan(non_connection_distances).any() or torch.isinf(non_connection_distances).any():
                return self._safe_tensor(0.0, device)
            
            if self.adaptive_thresholds:
                adaptive_threshold = 300.0
            else:
                adaptive_threshold = 400.0
            
            safe_distances = torch.clamp(non_connection_distances, min=10.0)
            
            repulsive_force = adaptive_threshold ** 2 / (safe_distances ** 2)
            
            violation_mask = non_connection_distances < adaptive_threshold
            if violation_mask.sum() > 0:
                violation_forces = repulsive_force[violation_mask]
                loss = torch.mean(violation_forces)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    return self._safe_tensor(0.0, device)
                
                return loss
            else:
                return self._safe_tensor(0.0, device)
            
        except Exception as e:
            logger.warning(f"Error in non_connection_penalty: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def spatial_dispersion_loss(self, pred_locations):
        try:
            device = pred_locations.device
            
            if pred_locations.shape[0] < 2:
                return self._safe_tensor(0.0, device)
            
            if torch.isnan(pred_locations).any() or torch.isinf(pred_locations).any():
                logger.warning('⚠️ Invalid pred_locations in spatial_dispersion_loss')
                return self._safe_tensor(0.0, device)
            
            distances = torch.cdist(pred_locations, pred_locations)
            
            if torch.isnan(distances).any() or torch.isinf(distances).any():
                return self._safe_tensor(0.0, device)
            
            distances = distances + torch.eye(len(pred_locations), device=distances.device) * 1e6
            
            min_distances = distances.min(dim=1)[0]
            
            if torch.isnan(min_distances).any() or torch.isinf(min_distances).any():
                return self._safe_tensor(0.0, device)
            
            coord_range = pred_locations.max() - pred_locations.min()
            if coord_range < 10:
                min_distances = min_distances * 450.0
            
            min_threshold = 50.0
            violations = torch.relu(min_threshold - min_distances)
            
            loss = torch.mean(torch.exp(violations / 10.0) - 1.0)
            
            if torch.isnan(loss) or torch.isinf(loss):
                return self._safe_tensor(0.0, device)
            
            return loss
            
        except Exception as e:
            logger.warning(f"Error in spatial_dispersion_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def precision_loss(self, pred_locations, true_locations):
        try:
            device = pred_locations.device
            
            if pred_locations.shape[0] == 0:
                return self._safe_tensor(0.0, device)
            
            if torch.isnan(pred_locations).any() or torch.isnan(true_locations).any():
                logger.warning('⚠️ Invalid inputs in precision_loss')
                return self._safe_tensor(0.0, device)
            
            if self.use_haversine:
                try:
                    error_distances = self.haversine_distance(pred_locations, true_locations)
                except:
                    error_distances = torch.norm(pred_locations - true_locations, dim=1) * 450.0
            else:
                euclidean_dist = torch.norm(pred_locations - true_locations, dim=1)
                coord_range = pred_locations.max() - pred_locations.min()
                if coord_range < 10:
                    error_distances = euclidean_dist * 450.0
                else:
                    error_distances = euclidean_dist
            
            if torch.isnan(error_distances).any() or torch.isinf(error_distances).any():
                return self._safe_tensor(0.0, device)
            
            error_distances = torch.clamp(error_distances, 0.0, 10000.0)
            
            normalized_errors = error_distances / 100.0
            
            precision_loss = torch.where(
                normalized_errors < 1.0,
                normalized_errors.pow(2),
                normalized_errors
            )
            
            result = torch.mean(precision_loss)
            
            if torch.isnan(result) or torch.isinf(result):
                return self._safe_tensor(0.0, device)
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in precision_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def geometric_consistency_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            
            if pred_locations.shape[0] < 3:
                return self._safe_tensor(0.0, device)
            
            if torch.isnan(pred_locations).any() or torch.isinf(pred_locations).any():
                logger.warning('⚠️ Invalid pred_locations in geometric_consistency_loss')
                return self._safe_tensor(0.0, device)
            
            distances = torch.cdist(pred_locations, pred_locations)
            
            if torch.isnan(distances).any() or torch.isinf(distances).any():
                return self._safe_tensor(0.0, device)
            
            n = pred_locations.shape[0]
            consistency_violations = []
            
            for i in range(min(n, 20)):
                for j in range(i + 1, min(n, 20)):
                    for k in range(j + 1, min(n, 20)):
                        try:
                            d_ij = distances[i, j]
                            d_jk = distances[j, k]
                            d_ik = distances[i, k]
                            
                            if torch.isnan(d_ij) or torch.isnan(d_jk) or torch.isnan(d_ik):
                                continue
                            
                            violation_1 = torch.relu(d_ij + d_jk - d_ik - 1e-6)
                            violation_2 = torch.relu(d_ij + d_ik - d_jk - 1e-6)
                            violation_3 = torch.relu(d_jk + d_ik - d_ij - 1e-6)
                            
                            for violation in [violation_1, violation_2, violation_3]:
                                if torch.isfinite(violation):
                                    consistency_violations.append(violation)
                        except Exception as e:
                            logger.debug(f"Error in triangle ({i},{j},{k}): {e}")
                            continue
            
            if consistency_violations:
                violations_tensor = torch.stack(consistency_violations)
                result = torch.mean(violations_tensor)
                
                if torch.isnan(result) or torch.isinf(result):
                    return self._safe_tensor(0.0, device)
                
                return result
            else:
                return self._safe_tensor(0.0, device)
                
        except Exception as e:
            logger.warning(f"Error in geometric_consistency_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def _safe_tensor(self, value, device, requires_grad=True):
        try:
            return torch.tensor(value, device=device, requires_grad=requires_grad)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(f"⚠️ CUDA failed to create a tensor, using CPU: {e}")
                tensor_cpu = torch.tensor(value, device='cpu', requires_grad=requires_grad)
                try:
                    return tensor_cpu.to(device)
                except:
                    logger.warning('⚠️ Could not move the tensor to GPU, keeping it on CPU')
                    return tensor_cpu
            else:
                raise e

    def triangular_coherence_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            
            if not hasattr(batch, 'edge_index_dict'):
                return self._safe_tensor(0.0, device)
            
            pea_triangular_edges = [
                edge_type for edge_type in batch.edge_index_dict.keys()
                if 'INTRA_TRIANGULO' in edge_type and edge_type[0] == 'PEA' and edge_type[2] == 'PEA'
            ]
            
            if not pea_triangular_edges:
                return self._safe_tensor(0.0, device)
            
            total_loss = 0.0
            count = 0
            
            for edge_type in pea_triangular_edges:
                try:
                    edge_index = batch.edge_index_dict[edge_type]
                    
                    if edge_index.size(1) == 0:
                        continue
                    
                    max_idx = pred_locations.size(0) - 1
                    valid_mask = (edge_index[0] <= max_idx) & (edge_index[1] <= max_idx)
                    
                    if not valid_mask.any():
                        continue
                    
                    valid_edge_index = edge_index[:, valid_mask]
                    
                    if valid_edge_index.size(1) < 2:
                        continue
                    
                    src_pos = pred_locations[valid_edge_index[0]]
                    dst_pos = pred_locations[valid_edge_index[1]]
                    
                    distances = torch.norm(src_pos - dst_pos, dim=1)
                    
                    if distances.numel() > 1:
                        coherence_loss = torch.var(distances)
                        total_loss += coherence_loss
                        count += 1
                    
                except Exception as e:
                    continue
            
            if count > 0:
                return self._safe_tensor(total_loss / count, device)
            else:
                return self._safe_tensor(0.0, device)
                
        except Exception as e:
            return self._safe_tensor(0.0, device)

    def smoothness_loss(self, pred_locations):
        try:
            device = pred_locations.device
            
            if pred_locations.shape[0] < 2:
                return self._safe_tensor(0.0, device)
            
            if torch.isnan(pred_locations).any() or torch.isinf(pred_locations).any():
                logger.warning('⚠️ Invalid pred_locations in smoothness_loss')
                return self._safe_tensor(0.0, device)
            
            center = torch.mean(pred_locations, dim=0)
            
            if torch.isnan(center).any():
                return self._safe_tensor(0.0, device)
            
            deviations = pred_locations - center
            
            variance = torch.mean(torch.sum(deviations ** 2, dim=1))
            
            if torch.isnan(variance) or torch.isinf(variance):
                return self._safe_tensor(0.0, device)
            
            return variance / 1000.0
            
        except Exception as e:
            logger.warning(f"Error in smoothness_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))

    def feature_reconstruction_loss(self, node_embeddings, batch):
        try:
            if node_embeddings is None:
                return self._safe_tensor(0.0, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            device = node_embeddings.device if hasattr(node_embeddings, 'device') else torch.device('cpu')
            return self._safe_tensor(0.0, device)
            
        except Exception as e:
            logger.warning(f"Error in feature_reconstruction_loss: {e}")
            return self._safe_tensor(0.0, torch.device('cpu'))
    
    def movement_penalty(self, pred_locations, max_movement=None):
        try:
            device = pred_locations.device
            
            if self.previous_positions is None:
                self.previous_positions = pred_locations.detach().clone()
                return self._safe_tensor(0.0, device)
            
            if torch.isnan(pred_locations).any() or torch.isinf(pred_locations).any():
                logger.warning('⚠️ Invalid pred_locations in movement_penalty')
                return self._safe_tensor(0.0, device)
            
            movement = torch.norm(pred_locations - self.previous_positions, dim=1)
            
            if torch.isnan(movement).any() or torch.isinf(movement).any():
                return self._safe_tensor(0.0, device)
            
            coord_range = pred_locations.max() - pred_locations.min()
            if coord_range < 10:
                movement = movement * 450.0
            
            if max_movement is not None:
                excessive_movement = torch.relu(movement - max_movement)
                penalty = torch.mean(excessive_movement ** 2)
            else:
                penalty = torch.mean(movement ** 2) / 10000.0
            
            if torch.isnan(penalty) or torch.isinf(penalty):
                penalty = self._safe_tensor(0.0, device)
            
            self.previous_positions = pred_locations.detach().clone()
            
            return penalty
            
        except Exception as e:
            logger.warning(f"Error in movement_penalty: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def boundary_constraints_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            return self._safe_tensor(0.0, device)
        except Exception as e:
            logger.warning(f"Error in boundary_constraints_loss: {e}")
            return self._safe_tensor(0.0, torch.device('cpu'))
    
    def topology_integrity_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            return self._safe_tensor(0.0, device)
        except Exception as e:
            logger.warning(f"Error in topology_integrity_loss: {e}")
            return self._safe_tensor(0.0, torch.device('cpu'))
    
    def local_precision_loss(self, pred_locations, true_locations):
        try:
            device = pred_locations.device
            
            if torch.isnan(pred_locations).any() or torch.isnan(true_locations).any():
                logger.warning('⚠️ Invalid inputs in local_precision_loss')
                return self._safe_tensor(0.0, device)
            
            errors = torch.norm(pred_locations - true_locations, dim=1)
            
            if torch.isnan(errors).any() or torch.isinf(errors).any():
                return self._safe_tensor(0.0, device)
            
            coord_range = pred_locations.max() - pred_locations.min()
            if coord_range < 10:
                errors = errors * 450.0
            
            refineable_mask = (errors >= 50.0) & (errors <= 200.0)
            
            if refineable_mask.sum() > 0:
                refineable_errors = errors[refineable_mask]
                result = torch.mean((refineable_errors / 100.0) ** 2)
                
                if torch.isnan(result) or torch.isinf(result):
                    return self._safe_tensor(0.0, device)
                
                return result
            else:
                return self._safe_tensor(0.0, device)
                
        except Exception as e:
            logger.warning(f"Error in local_precision_loss: {e}")
            return self._safe_tensor(0.0, pred_locations.device if pred_locations is not None else torch.device('cpu'))
    
    def adaptive_threshold_loss(self, pred_locations, batch):
        try:
            device = pred_locations.device
            return self._safe_tensor(0.0, device)
        except Exception as e:
            logger.warning(f"Error in adaptive_threshold_loss: {e}")
            return self._safe_tensor(0.0, torch.device('cpu'))
    
    def forward(self, pred_locations, true_locations, batch, 
                epoch=0, total_epochs=1, node_embeddings=None,
                max_movement=None):
        try:
            device = pred_locations.device
            loss_components = {}
            
            
            if torch.isnan(pred_locations).any() or torch.isinf(pred_locations).any():
                logger.error('❌ pred_locations contains NaN or Inf')
                pred_locations = torch.nan_to_num(pred_locations, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            if torch.isnan(true_locations).any() or torch.isinf(true_locations).any():
                logger.error('❌ true_locations contains NaN or Inf')
                true_locations = torch.nan_to_num(true_locations, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            pred_locations = torch.clamp(pred_locations, -1000.0, 1000.0)
            true_locations = torch.clamp(true_locations, -1000.0, 1000.0)
            
            if pred_locations.size(0) != true_locations.size(0):
                logger.error(f"❌ Incompatible sizes: pred={pred_locations.size()}, true={true_locations.size()}")
                min_size = min(pred_locations.size(0), true_locations.size(0))
                pred_locations = pred_locations[:min_size]
                true_locations = true_locations[:min_size]
            
            if pred_locations.size(1) != 2 or true_locations.size(1) != 2:
                logger.error(f"❌ Incorrect dimensions: pred={pred_locations.size()}, true={true_locations.size()}")
                return self._safe_tensor(1.0, device), {}
            
            if self.use_haversine:
                try:
                    current_errors = self.haversine_distance(pred_locations, true_locations)
                    current_errors = torch.clamp(current_errors, 0.0, 10000.0)
                except Exception as e:
                    logger.warning(f"Error in the haversine: {e}. Using Euclidean.")
                    current_errors = torch.norm(pred_locations - true_locations, dim=1) * 450.0
            else:
                euclidean_dist = torch.norm(pred_locations - true_locations, dim=1)
                coord_range = pred_locations.max() - pred_locations.min()
                if coord_range < 10:
                    current_errors = euclidean_dist * 450.0
                else:
                    current_errors = euclidean_dist
            
            if torch.isnan(current_errors).any() or torch.isinf(current_errors).any():
                current_errors = torch.nan_to_num(current_errors, nan=100.0, posinf=1000.0, neginf=0.0)
            
            avg_error = current_errors.mean().item()
            if not np.isfinite(avg_error):
                avg_error = 100.0
            
            self.error_history.append(avg_error)
            if len(self.error_history) > 10:
                self.error_history.pop(0)
            
            
            try:
                connection_loss = self.connection_consistency_loss(pred_locations, batch)
                if torch.isnan(connection_loss) or torch.isinf(connection_loss):
                    connection_loss = self._safe_tensor(0.0, device)
                connection_weight = max(self.current_weights.get('connection_weight', 3.0), 2.0)
                loss_components['connection'] = connection_loss * connection_weight
            except Exception as e:
                logger.debug(f"Error in connection_loss: {e}")
                loss_components['connection'] = self._safe_tensor(0.0, device)
            
            try:
                non_connection_loss = self.non_connection_penalty(pred_locations, batch)
                if torch.isnan(non_connection_loss) or torch.isinf(non_connection_loss):
                    non_connection_loss = self._safe_tensor(0.0, device)
                non_connection_weight = max(self.current_weights.get('non_connection_weight', 5.0), 3.0)
                loss_components['non_connection'] = non_connection_loss * non_connection_weight
            except Exception as e:
                logger.debug(f"Error in non_connection_loss: {e}")
                loss_components['non_connection'] = self._safe_tensor(0.0, device)
            
            try:
                position_loss = self.position_loss(pred_locations, true_locations)
                if torch.isnan(position_loss) or torch.isinf(position_loss):
                    position_loss = self._safe_tensor(1.0, device)
                position_weight = min(max(self.current_weights.get('position_weight', 10.0), 5.0), 15.0)
                loss_components['position'] = position_loss * position_weight
            except Exception as e:
                logger.debug(f"Error in position_loss: {e}")
                loss_components['position'] = self._safe_tensor(1.0, device)
            
            try:
                geometric_loss = self.geometric_consistency_loss(pred_locations, batch)
                if torch.isnan(geometric_loss) or torch.isinf(geometric_loss):
                    geometric_loss = self._safe_tensor(0.0, device)
                loss_components['geometric'] = geometric_loss * self.current_weights.get('physics_weight', 2.0)
            except Exception as e:
                logger.debug(f"Error in geometric_loss: {e}")
                loss_components['geometric'] = self._safe_tensor(0.0, device)
            
            try:
                precision_loss = self.precision_loss(pred_locations, true_locations)
                if torch.isnan(precision_loss) or torch.isinf(precision_loss):
                    precision_loss = self._safe_tensor(0.0, device)
                loss_components['precision'] = precision_loss * self.current_weights.get('precision_weight', 1.5)
            except Exception as e:
                logger.debug(f"Error in precision_loss: {e}")
                loss_components['precision'] = self._safe_tensor(0.0, device)
            
            try:
                dispersion_loss = self.spatial_dispersion_loss(pred_locations)
                if torch.isnan(dispersion_loss) or torch.isinf(dispersion_loss):
                    dispersion_loss = self._safe_tensor(0.0, device)
                loss_components['spatial_dispersion'] = dispersion_loss * self.current_weights.get('spatial_dispersion_weight', 0.5)
            except Exception as e:
                logger.debug(f"Error in dispersion_loss: {e}")
                loss_components['spatial_dispersion'] = self._safe_tensor(0.0, device)
            
            try:
                triangular_loss = self.triangular_coherence_loss(pred_locations, batch)
                if torch.isnan(triangular_loss) or torch.isinf(triangular_loss):
                    triangular_loss = self._safe_tensor(0.0, device)
                loss_components['triangular'] = triangular_loss * self.current_weights.get('triangular_weight', 1.0)
            except Exception as e:
                logger.debug(f"Error in triangular_loss: {e}")
                loss_components['triangular'] = self._safe_tensor(0.0, device)
            
            try:
                smoothness_loss = self.smoothness_loss(pred_locations)
                if torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss):
                    smoothness_loss = self._safe_tensor(0.0, device)
                loss_components['smoothness'] = smoothness_loss * self.current_weights.get('smoothness_weight', 0.1)
            except Exception as e:
                logger.debug(f"Error in smoothness_loss: {e}")
                loss_components['smoothness'] = self._safe_tensor(0.0, device)
            
            try:
                feature_loss = self.feature_reconstruction_loss(node_embeddings, batch)
                if torch.isnan(feature_loss) or torch.isinf(feature_loss):
                    feature_loss = self._safe_tensor(0.0, device)
                loss_components['feature_reconstruction'] = feature_loss * self.current_weights.get('feature_reconstruction_weight', 0.001)
            except Exception as e:
                logger.debug(f"Error in feature_loss: {e}")
                loss_components['feature_reconstruction'] = self._safe_tensor(0.0, device)
            
            try:
                movement_loss = self.movement_penalty(pred_locations, max_movement)
                if torch.isnan(movement_loss) or torch.isinf(movement_loss):
                    movement_loss = self._safe_tensor(0.0, device)
                loss_components['movement'] = movement_loss * self.current_weights.get('movement_penalty_weight', 0.1)
            except Exception as e:
                logger.debug(f"Error in movement_loss: {e}")
                loss_components['movement'] = self._safe_tensor(0.0, device)
            
            try:
                boundary_loss = self.boundary_constraints_loss(pred_locations, batch)
                if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
                    boundary_loss = self._safe_tensor(0.0, device)
                loss_components['boundary'] = boundary_loss * self.current_weights.get('boundary_weight', 0.5)
            except Exception as e:
                logger.debug(f"Error in boundary_loss: {e}")
                loss_components['boundary'] = self._safe_tensor(0.0, device)
            
            try:
                topology_loss = self.topology_integrity_loss(pred_locations, batch)
                if torch.isnan(topology_loss) or torch.isinf(topology_loss):
                    topology_loss = self._safe_tensor(0.0, device)
                loss_components['topology'] = topology_loss * self.current_weights.get('topology_weight', 0.3)
            except Exception as e:
                logger.debug(f"Error in topology_loss: {e}")
                loss_components['topology'] = self._safe_tensor(0.0, device)
            
            try:
                local_precision_loss = self.local_precision_loss(pred_locations, true_locations)
                if torch.isnan(local_precision_loss) or torch.isinf(local_precision_loss):
                    local_precision_loss = self._safe_tensor(0.0, device)
                loss_components['local_precision'] = local_precision_loss * self.current_weights.get('local_precision_weight', 1.0)
            except Exception as e:
                logger.debug(f"Error in local_precision_loss: {e}")
                loss_components['local_precision'] = self._safe_tensor(0.0, device)
            
            try:
                adaptive_loss = self.adaptive_threshold_loss(pred_locations, batch)
                if torch.isnan(adaptive_loss) or torch.isinf(adaptive_loss):
                    adaptive_loss = self._safe_tensor(0.0, device)
                loss_components['adaptive_threshold'] = adaptive_loss * self.current_weights.get('adaptive_threshold_weight', 0.8)
            except Exception as e:
                logger.debug(f"Error in adaptive_loss: {e}")
                loss_components['adaptive_threshold'] = self._safe_tensor(0.0, device)
            
            try:
                rf_physics_components = ['connection', 'non_connection', 'geometric', 'triangular']
                rf_physics_loss = sum(loss_components[comp] for comp in rf_physics_components if comp in loss_components)
                if torch.isnan(rf_physics_loss) or torch.isinf(rf_physics_loss):
                    rf_physics_loss = self._safe_tensor(0.0, device)
                loss_components['rf_physics'] = rf_physics_loss * self.current_weights.get('rf_physics_weight', 0.5)
            except Exception as e:
                logger.debug(f"Error in rf_physics_loss: {e}")
                loss_components['rf_physics'] = self._safe_tensor(0.0, device)
            
            try:
                geometric_loss_value = loss_components.get('geometric', self._safe_tensor(0.0, device)).item()
                if np.isfinite(geometric_loss_value):
                    self.apply_weight_annealing(epoch, geometric_loss_value)
            except Exception as e:
                logger.debug(f"Error in the annealing: {e}")
            
            total_loss = self._safe_tensor(0.0, device)
            for comp_name, comp_loss in loss_components.items():
                if torch.isnan(comp_loss) or torch.isinf(comp_loss):
                    logger.debug(f"Invalid component {comp_name}, ignoring it")
                    continue
                total_loss = total_loss + comp_loss
            
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error('❌ Total loss is NaN or Inf, using fallback')
                total_loss = self._safe_tensor(1.0, device)
            
            if total_loss > 10000.0:
                logger.warning(f"⚠️ Very high loss ({total_loss:.2f}), applying adaptive normalization")
                total_loss = torch.log(1.0 + total_loss / 1000.0) * 1000.0
            
            if self.log_verbose and epoch % 10 == 0:
                logger.info(f"[PHYSICS-INFORMED LOSS] Epoch {epoch}:")
                logger.info(f"  🔗 connection={loss_components.get('connection', 0.0):.4f}")
                logger.info(f"  ❌ non_connection={loss_components.get('non_connection', 0.0):.4f}")
                logger.info(f"  📍 position={loss_components.get('position', 0.0):.4f}")
                logger.info(f"  🎯 precision={loss_components.get('precision', 0.0):.4f}")
                logger.info(f"  🔬 rf_physics={loss_components.get('rf_physics', 0.0):.4f}")
                logger.info(f"  ✅ TOTAL={total_loss:.4f}")
                logger.info(f"  📊 avg_error={avg_error:.1f}m")
            
            return total_loss, loss_components
            
        except Exception as e:
            logger.error(f"Error in the forward of the loss: {e}")
            device = pred_locations.device if pred_locations is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return self._safe_tensor(1.0, device), {}


def create_physics_informed_loss(config):
    loss_config = config.get('loss', {})
    
    loss_fn = PhysicsInformedLoss(
        position_weight=loss_config.get('position_weight', 10.0),
        physics_weight=loss_config.get('physics_weight', 2.0),
        connection_weight=loss_config.get('connection_weight', 3.0),
        non_connection_weight=loss_config.get('non_connection_weight', 5.0),
        triangular_weight=loss_config.get('triangular_weight', 1.0),
        precision_weight=loss_config.get('precision_weight', 2.0),
        spatial_dispersion_weight=loss_config.get('spatial_dispersion_weight', 0.8),
        rf_physics_weight=loss_config.get('rf_physics_weight', 1.5),
        use_weight_annealing=loss_config.get('use_weight_annealing', True),
        use_haversine=loss_config.get('use_haversine', True),
        adaptive_thresholds=loss_config.get('adaptive_thresholds', True),
        use_numba_acceleration=loss_config.get('use_numba_acceleration', True)
    )
    
    logger.info('✅ ULTRA-ADVANCED PhysicsInformedLoss created successfully!')
    
    return loss_fn


def extract_loss_components(loss_output):
    if isinstance(loss_output, tuple) and len(loss_output) == 2:
        return loss_output[1]
    return {}

def get_total_loss(loss_output):
    if isinstance(loss_output, tuple) and len(loss_output) == 2:
        return loss_output[0]
    return loss_output 
