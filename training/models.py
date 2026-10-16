#!/usr/bin/env python3
"""ARPIA multi-scale heterogeneous GNN (local, regional and global components)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.nn import HeteroConv, GATConv, TransformerConv, global_mean_pool
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class AdvancedFeatureProcessor(nn.Module):
    
    def __init__(self, input_dims_dict, hidden_dim=128, use_rf_attention=True):
        super().__init__()
        self.input_dims_dict = input_dims_dict
        self.hidden_dim = hidden_dim
        self.use_rf_attention = use_rf_attention
        
        
        if use_rf_attention:
            self.rf_attention = RFAttentionModule(hidden_dim)
        
        self.final_projections = nn.ModuleDict()
        for node_type, input_dim in input_dims_dict.items():
            self.final_projections[node_type] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
    
    def forward(self, node_features_dict):
        final_features = {}
        
        for node_type, features in node_features_dict.items():
            if node_type in self.final_projections:
                final_features[node_type] = self.final_projections[node_type](features)
            else:
                if features.size(1) != self.hidden_dim:
                    temp_proj = nn.Linear(features.size(1), self.hidden_dim).to(features.device)
                    final_features[node_type] = temp_proj(features)
                else:
                    final_features[node_type] = features
        
        return final_features


class CompositeFeatureGenerator(nn.Module):
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.pea_generator = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.erb_generator = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(self, node_features_dict):
        enhanced = {}
        
        for node_type, features in node_features_dict.items():
            if node_type == 'PEA' and features.size(1) >= 2:
                composite = self.pea_generator(features[:, :2])
                enhanced[node_type] = torch.cat([features, composite], dim=1)
            elif node_type == 'ERB' and features.size(1) >= 2:
                composite = self.erb_generator(features[:, :2])
                enhanced[node_type] = torch.cat([features, composite], dim=1)
            else:
                enhanced[node_type] = features
                
        return enhanced


class RFAttentionModule(nn.Module):
    
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.rf_projector = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, raio, azimute, duracao, potencia=None):
        if potencia is None:
            potencia = torch.ones_like(raio)
        
        rf_features = torch.cat([raio, azimute, duracao, potencia], dim=1)
        
        projected = self.rf_projector(rf_features)
        
        projected = projected.unsqueeze(1)
        
        attended, _ = self.multihead_attn(projected, projected, projected)
        attended = attended.squeeze(1)
        
        output = self.output_proj(attended)
        
        return output


class LocalScaleGNN(nn.Module):
    
    def __init__(self, node_types, edge_types, node_feature_dims, 
                 hidden_channels=128, out_channels=64, num_layers=3, 
                 heads=8, dropout=0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(node_feature_dims[node_type], hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src, rel, dst = edge_type
                
                conv = GATConv(
                    (hidden_channels, hidden_channels), 
                    hidden_channels // heads,
                    heads=heads, 
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False
                )
                conv_dict[edge_type] = conv
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        self.final_projection = nn.ModuleDict()
        for node_type in node_types:
            self.final_projection[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU()
            )
    
    def forward(self, x_dict, edge_index_dict, batch_info=None):
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                h_dict[node_type] = self.node_encoders[node_type](x)
                
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ Invalid values detected in {node_type}, applying correction...")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
        
        for conv_idx, conv in enumerate(self.convs):
            try:
                valid_edge_dict = {}
                for edge_type, edge_index in edge_index_dict.items():
                    if edge_index.numel() > 0:
                        src_type, _, dst_type = edge_type
                        if src_type in h_dict and dst_type in h_dict:
                            max_src_idx = h_dict[src_type].size(0) - 1
                            max_dst_idx = h_dict[dst_type].size(0) - 1
                            
                            if edge_index.size(0) >= 2:
                                src_indices = edge_index[0]
                                dst_indices = edge_index[1]
                                
                                src_indices = torch.clamp(src_indices, 0, max_src_idx)
                                dst_indices = torch.clamp(dst_indices, 0, max_dst_idx)
                                
                                valid_edge_dict[edge_type] = torch.stack([src_indices, dst_indices])
                            else:
                                valid_edge_dict[edge_type] = edge_index
                        else:
                            valid_edge_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                    else:
                        valid_edge_dict[edge_type] = edge_index
                
                h_dict_new = conv(h_dict, valid_edge_dict)
                
                for node_type, h in h_dict_new.items():
                    if torch.isnan(h).any() or torch.isinf(h).any():
                        logger.warning(f"⚠️ Convolution {conv_idx} produced invalid values for {node_type}")
                        h_dict_new[node_type] = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                        h_dict_new[node_type] = torch.clamp(h_dict_new[node_type], -10.0, 10.0)
                
                h_dict = h_dict_new
                
            except Exception as e:
                logger.error(f"❌ Error in convolution {conv_idx}: {e}")
                logger.warning(f"🔧 Skipping convolution {conv_idx}, keeping the previous features")
                continue
            
            for node_type in h_dict:
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ Invalid values before the activation in {node_type}")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
                
                h_dict[node_type] = F.relu(h_dict[node_type])
                
                h_dict[node_type] = F.dropout(h_dict[node_type], 
                                            p=0.1, training=self.training)
                
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ Invalid values after dropout in {node_type}")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
        
        output_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.final_projection:
                if torch.isnan(h).any() or torch.isinf(h).any():
                    logger.warning(f"⚠️ Invalid values before the final projection in {node_type}")
                    h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                    h = torch.clamp(h, -10.0, 10.0)
                
                output_dict[node_type] = self.final_projection[node_type](h)
                
                if torch.isnan(output_dict[node_type]).any() or torch.isinf(output_dict[node_type]).any():
                    logger.warning(f"⚠️ Invalid values after the final projection in {node_type}")
                    output_dict[node_type] = torch.nan_to_num(output_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    output_dict[node_type] = torch.clamp(output_dict[node_type], -10.0, 10.0)
        
        return output_dict


class RegionalScaleGNN(nn.Module):
    
    def __init__(self, node_types, edge_types, node_feature_dims,
                 hidden_channels=128, out_channels=64, num_layers=2,
                 heads=4, dropout=0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(node_feature_dims[node_type], hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src, rel, dst = edge_type
                
                conv = TransformerConv(
                    (hidden_channels, hidden_channels), 
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout
                )
                conv_dict[edge_type] = conv
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        self.final_projection = nn.ModuleDict()
        for node_type in node_types:
            self.final_projection[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU()
            )
    
    def forward(self, x_dict, edge_index_dict, batch_info=None):
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_encoders:
                h_dict[node_type] = self.node_encoders[node_type](x)
                
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ [REGIONAL] Invalid values detected in {node_type}, applying correction...")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
        
        for conv_idx, conv in enumerate(self.convs):
            try:
                valid_edge_dict = {}
                for edge_type, edge_index in edge_index_dict.items():
                    if edge_index.numel() > 0:
                        src_type, _, dst_type = edge_type
                        if src_type in h_dict and dst_type in h_dict:
                            max_src_idx = h_dict[src_type].size(0) - 1
                            max_dst_idx = h_dict[dst_type].size(0) - 1
                            
                            if edge_index.size(0) >= 2:
                                src_indices = edge_index[0]
                                dst_indices = edge_index[1]
                                
                                src_indices = torch.clamp(src_indices, 0, max_src_idx)
                                dst_indices = torch.clamp(dst_indices, 0, max_dst_idx)
                                
                                valid_edge_dict[edge_type] = torch.stack([src_indices, dst_indices])
                            else:
                                valid_edge_dict[edge_type] = edge_index
                        else:
                            valid_edge_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                    else:
                        valid_edge_dict[edge_type] = edge_index
                
                h_dict_new = conv(h_dict, valid_edge_dict)
                
                for node_type, h in h_dict_new.items():
                    if torch.isnan(h).any() or torch.isinf(h).any():
                        logger.warning(f"⚠️ [REGIONAL] Convolution {conv_idx} produced invalid values for {node_type}")
                        h_dict_new[node_type] = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                        h_dict_new[node_type] = torch.clamp(h_dict_new[node_type], -10.0, 10.0)
                
                h_dict = h_dict_new
                
            except Exception as e:
                logger.error(f"❌ [REGIONAL] Error in convolution {conv_idx}: {e}")
                logger.warning(f"🔧 [REGIONAL] Skipping convolution {conv_idx}, keeping the previous features")
                continue
            
            for node_type in h_dict:
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ [REGIONAL] Invalid values before the activation in {node_type}")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
                
                h_dict[node_type] = F.relu(h_dict[node_type])
                
                h_dict[node_type] = F.dropout(h_dict[node_type], 
                                            p=0.1, training=self.training)
                
                if torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any():
                    logger.warning(f"⚠️ [REGIONAL] Invalid values after dropout in {node_type}")
                    h_dict[node_type] = torch.nan_to_num(h_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    h_dict[node_type] = torch.clamp(h_dict[node_type], -10.0, 10.0)
        
        output_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.final_projection:
                if torch.isnan(h).any() or torch.isinf(h).any():
                    logger.warning(f"⚠️ [REGIONAL] Invalid values before the final projection in {node_type}")
                    h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                    h = torch.clamp(h, -10.0, 10.0)
                
                output_dict[node_type] = self.final_projection[node_type](h)
                
                if torch.isnan(output_dict[node_type]).any() or torch.isinf(output_dict[node_type]).any():
                    logger.warning(f"⚠️ [REGIONAL] Invalid values after the final projection in {node_type}")
                    output_dict[node_type] = torch.nan_to_num(output_dict[node_type], nan=0.0, posinf=1.0, neginf=-1.0)
                    output_dict[node_type] = torch.clamp(output_dict[node_type], -10.0, 10.0)
        
        return output_dict


class GlobalContextModule(nn.Module):
    
    def __init__(self, node_types, hidden_channels=128, out_channels=64,
                 num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.MultiheadAttention(
                hidden_channels, num_heads, 
                dropout=dropout, batch_first=True
            )
            self.transformer_layers.append(layer)
        
        self.global_pooling = nn.ModuleDict()
        for node_type in node_types:
            self.global_pooling[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU()
            )
    
    def forward(self, x_dict, node_positions_dict):
        global_features = {}
        
        for node_type, features in x_dict.items():
            if node_type in self.node_encoders:
                try:
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        logger.warning(f"⚠️ [GLOBAL] Invalid values detected in {node_type}, applying correction...")
                        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                        features = torch.clamp(features, -10.0, 10.0)
                    
                    h = self.node_encoders[node_type](features)
                    
                    if torch.isnan(h).any() or torch.isinf(h).any():
                        logger.warning(f"⚠️ [GLOBAL] Invalid values after the encoder in {node_type}")
                        h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                        h = torch.clamp(h, -10.0, 10.0)
                    
                    if node_type in node_positions_dict:
                        pos_features = node_positions_dict[node_type]
                        
                        if torch.isnan(pos_features).any() or torch.isinf(pos_features).any():
                            logger.warning(f"⚠️ [GLOBAL] Invalid positions for {node_type}")
                            pos_features = torch.nan_to_num(pos_features, nan=0.0, posinf=1.0, neginf=-1.0)
                            pos_features = torch.clamp(pos_features, -1000.0, 1000.0)
                        
                        pos_encoded = self.position_encoder(pos_features)
                        
                        if torch.isnan(pos_encoded).any() or torch.isinf(pos_encoded).any():
                            logger.warning(f"⚠️ [GLOBAL] Invalid encoded positions for {node_type}")
                            pos_encoded = torch.nan_to_num(pos_encoded, nan=0.0, posinf=1.0, neginf=-1.0)
                            pos_encoded = torch.clamp(pos_encoded, -10.0, 10.0)
                        
                        combined = h + pos_encoded
                        
                        if torch.isnan(combined).any() or torch.isinf(combined).any():
                            logger.warning(f"⚠️ [GLOBAL] Invalid combination for {node_type}")
                            combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
                            combined = torch.clamp(combined, -10.0, 10.0)
                        
                        h = combined
                    
                    for layer_idx, layer in enumerate(self.transformer_layers):
                        try:
                            if h.dim() == 2:
                                h = h.unsqueeze(1)
                            
                            h_transformed, _ = layer(h, h, h)
                            
                            if torch.isnan(h_transformed).any() or torch.isinf(h_transformed).any():
                                logger.warning(f"⚠️ [GLOBAL] Transformer layer {layer_idx} produced invalid values for {node_type}")
                                h_transformed = torch.nan_to_num(h_transformed, nan=0.0, posinf=1.0, neginf=-1.0)
                                h_transformed = torch.clamp(h_transformed, -10.0, 10.0)
                            
                            h = h_transformed
                            
                        except Exception as e:
                            logger.error(f"❌ [GLOBAL] Error in transformer layer {layer_idx} for {node_type}: {e}")
                            logger.warning(f"🔧 [GLOBAL] Pulando transformer layer {layer_idx} para {node_type}")
                            continue
                    
                    if h.dim() == 3:
                        h = h.squeeze(1)
                    
                    if node_type in self.global_pooling:
                        pooled = self.global_pooling[node_type](h)
                        
                        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
                            logger.warning(f"⚠️ [GLOBAL] Pooling produced invalid values for {node_type}")
                            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0)
                            pooled = torch.clamp(pooled, -10.0, 10.0)
                        
                        global_features[node_type] = pooled
                    else:
                        global_features[node_type] = h
                        
                except Exception as e:
                    logger.error(f"❌ [GLOBAL] General error while processing {node_type}: {e}")
                    global_features[node_type] = torch.zeros(
                        features.size(0), self.out_channels,
                        device=features.device, dtype=features.dtype
                    )
        
        return global_features


class MultiScaleGNN(nn.Module):
    
    def __init__(self, node_types, edge_types, node_feature_dims,
                 hidden_channels=128, out_channels=64, dropout=0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.local_gnn = LocalScaleGNN(
            node_types, edge_types, node_feature_dims,
            hidden_channels, out_channels, 
            num_layers=3, heads=8, dropout=dropout
        )
        
        self.regional_gnn = RegionalScaleGNN(
            node_types, edge_types, node_feature_dims,
            hidden_channels, out_channels,
            num_layers=2, heads=4, dropout=dropout
        )
        
        self.global_context = GlobalContextModule(
            node_types, hidden_channels, out_channels,
            num_heads=8, num_layers=2, dropout=dropout
        )
        
        self.scale_fusion = nn.ModuleDict()
        for node_type in node_types:
            self.scale_fusion[node_type] = nn.Sequential(
                nn.Linear(out_channels * 3, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels)
            )
        
        self.scale_attention = nn.ModuleDict()
        for node_type in node_types:
            self.scale_attention[node_type] = nn.Sequential(
                nn.Linear(out_channels * 3, 3),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, x_dict, edge_index_dict, batch_info=None):
        node_positions_dict = {}
        for node_type, features in x_dict.items():
            if features.size(1) >= 2:
                node_positions_dict[node_type] = features[:, :2]
        
        local_features = self.local_gnn(x_dict, edge_index_dict, batch_info)
        regional_features = self.regional_gnn(x_dict, edge_index_dict, batch_info)
        global_features = self.global_context(x_dict, node_positions_dict)
        
        multi_scale_dict = {}
        for node_type in self.node_types:
            if node_type in local_features:
                local_feat = local_features[node_type]
                
                regional_feat = regional_features.get(node_type, local_feat)
                global_feat = global_features.get(node_type, local_feat)
                
                try:
                    if local_feat.size(1) == self.out_channels:
                        target_dim = self.out_channels
                    else:
                        target_dim = local_feat.size(1)
                    
                    if regional_feat.size(1) != target_dim:
                        regional_feat = local_feat
                    if global_feat.size(1) != target_dim:
                        global_feat = local_feat
                    
                    combined = torch.cat([local_feat, regional_feat, global_feat], dim=1)
                    
                    if node_type in self.scale_fusion:
                        attention_weights = self.scale_attention[node_type](combined)
                        
                        weighted_features = (
                            attention_weights[:, 0:1] * local_feat +
                            attention_weights[:, 1:2] * regional_feat +
                            attention_weights[:, 2:3] * global_feat
                        )
                        
                        multi_scale_dict[node_type] = self.scale_fusion[node_type](combined)
                    else:
                        multi_scale_dict[node_type] = local_feat
                        
                except Exception as e:
                    logger.warning(f"Error while processing {node_type}: {e}. Using only local.")
                    multi_scale_dict[node_type] = local_feat
            
            elif node_type in regional_features:
                multi_scale_dict[node_type] = regional_features[node_type]
            elif node_type in global_features:
                multi_scale_dict[node_type] = global_features[node_type]
        
        return multi_scale_dict


class LocalPrecisionRefiner(nn.Module):
    
    def __init__(self, input_dim=64, hidden_dim=32, max_refinement_distance=100.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_refinement_distance = max_refinement_distance
        
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.position_refiner = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"🎯 LocalPrecisionRefiner V2 inicializado:")
        logger.info(f"   📊 Input dim: {input_dim}")
        logger.info(f"   🔧 Hidden dim: {hidden_dim}")
        logger.info(f"   📏 Max refinement: {max_refinement_distance}m")
    
    def compute_local_context(self, pea_embeddings, erb_positions, erb_embeddings, max_distance=500.0):
        batch_size = pea_embeddings.size(0)
        device = pea_embeddings.device
        
        local_contexts = []
        
        for i in range(batch_size):
            pea_emb = pea_embeddings[i]
            
            distances = torch.norm(erb_positions - erb_positions[0], dim=1)
            
            close_mask = distances < max_distance
            if close_mask.sum() == 0:
                close_mask[distances.argmin()] = True
            
            close_erb_embs = erb_embeddings[close_mask]
            aggregated_erb = torch.mean(close_erb_embs, dim=0)
            
            combined = torch.cat([pea_emb, aggregated_erb], dim=0)
            local_contexts.append(combined)
        
        return torch.stack(local_contexts, dim=0)
    
    def forward(self, pea_embeddings, rough_positions, erb_positions, erb_embeddings):
        local_context = self.compute_local_context(
            pea_embeddings, erb_positions, erb_embeddings
        )
        
        context_features = self.context_encoder(local_context)
        
        erb_pos_mean = torch.mean(erb_positions, dim=0, keepdim=True)
        erb_pos_mean = erb_pos_mean.expand(rough_positions.size(0), -1)
        
        refiner_input = torch.cat([
            context_features,
            rough_positions,
            erb_pos_mean
        ], dim=1)
        
        refinement_offset = self.position_refiner(refiner_input)
        
        offset_magnitude = torch.norm(refinement_offset, dim=1, keepdim=True)
        max_offset = self.max_refinement_distance / 450.0
        refinement_offset = torch.where(
            offset_magnitude > max_offset,
            refinement_offset * (max_offset / offset_magnitude),
            refinement_offset
        )
        
        confidence = self.confidence_estimator(context_features)
        
        refined_positions = rough_positions + refinement_offset * confidence
        
        return refined_positions, confidence, refinement_offset


class ARPIAGNNv4(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config.get('model', {})
        
        self.hidden_channels = model_config.get('hidden_channels', 128)
        self.out_channels = model_config.get('out_channels', 64)
        self.num_layers = model_config.get('num_layers', 4)
        self.dropout = model_config.get('dropout', 0.1)
        self.heads = model_config.get('heads', 8)
        
        self.use_local_refiner = model_config.get('use_local_refiner', True)
        
        self.node_types = ['PEA', 'ERB', 'CPS', 'CTC', 'CTNC']
        self.edge_types = [
            ('PEA', 'CONEXAO', 'ERB'),
            ('PEA', 'NAO_CONEXAO', 'ERB'),
            ('PEA', 'PROXIMIDADE', 'PEA'),
            ('ERB', 'PROXIMIDADE', 'ERB'),
            ('CPS', 'PROXIMIDADE', 'CPS'),
            ('CTC', 'PROXIMIDADE', 'CTC'),
            ('CTNC', 'PROXIMIDADE', 'CTNC'),
            ('CTC', 'INTRA_TRIANGULO', 'PEA'),
            ('PEA', 'INTRA_TRIANGULO', 'CTC'),
            ('CTC', 'INTRA_TRIANGULO', 'ERB'),
            ('ERB', 'INTRA_TRIANGULO', 'CTC'),
            ('CTC', 'INTRA_TRIANGULO', 'CPS'),
            ('CPS', 'INTRA_TRIANGULO', 'CTC'),
            ('CTNC', 'INTRA_TRIANGULO', 'PEA'),
            ('PEA', 'INTRA_TRIANGULO', 'CTNC'),
            ('CTNC', 'INTRA_TRIANGULO', 'ERB'),
            ('ERB', 'INTRA_TRIANGULO', 'CTNC'),
            ('CTNC', 'INTRA_TRIANGULO', 'CPS'),
            ('CPS', 'INTRA_TRIANGULO', 'CTNC')
        ]
        
        self.node_feature_dims = {}
        
        self.encoder = None
        
        self.location_decoder = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_channels, 2)
        )
        
        if self.use_local_refiner:
            self.local_refiner = LocalPrecisionRefiner(
                input_dim=self.out_channels,
                hidden_dim=self.out_channels // 2,
                max_refinement_distance=100.0
            )
        else:
            self.local_refiner = None
        
        self.feature_processor = None
        
        self.model_config = model_config
        
        logger.info('🚀 IMPROVED ARPIA GNN v4 V2 initialized (the encoder will be created after the data)')
        if self.use_local_refiner:
            logger.info("   🎯 LocalPrecisionRefiner V2 ativado")

    def initialize_encoder_with_dims(self, node_feature_dims):
        logger.info('🔧 Initializing the ROBUST encoder with the real dimensions:')
        
        self.node_feature_dims = node_feature_dims.copy()
        
        for node_type, dims in self.node_feature_dims.items():
            logger.info(f"   {node_type}: {dims} features")
        
        self.feature_processor = AdvancedFeatureProcessor(
            self.node_feature_dims,
            hidden_dim=self.hidden_channels,
            use_rf_attention=False
        )
        
        processed_dims = {}
        for node_type in self.node_types:
            if node_type in self.node_feature_dims:
                processed_dims[node_type] = self.hidden_channels
        
        logger.info('🔧 Dimensions after the feature processing:')
        for node_type, dims in processed_dims.items():
            logger.info(f"   {node_type}: {dims} -> {self.hidden_channels}")
        
        self.encoder = MultiScaleGNN(
            node_types=self.node_types,
            edge_types=self.edge_types,
            node_feature_dims=processed_dims,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            dropout=self.dropout
        )
        
        logger.info("✅ Encoder Multi-Scale ROBUSTO inicializado!")
        logger.info(f"   📊 Input dims: {processed_dims}")
        logger.info(f"   🔄 Hidden channels: {self.hidden_channels}")
        logger.info(f"   📤 Output channels: {self.out_channels}")
        
        return self
    
    def forward(self, batch):
        if self.encoder is None:
            raise RuntimeError('Encoder was not initialized! Call initialize_encoder_with_dims() first.')
        
        x_dict, edge_index_dict = self._prepare_batch_data(batch)
        
        if self.feature_processor is not None:
            x_dict = self.feature_processor(x_dict)
        
        embeddings = self.encoder(x_dict, edge_index_dict, batch)
        
        pea_embeddings = embeddings.get('PEA')
        if pea_embeddings is None:
            raise ValueError('Could not obtain the PEA embeddings')
        
        rough_positions = self.location_decoder(pea_embeddings)
        
        final_positions = rough_positions
        refinement_info = {}
        
        if self.local_refiner is not None and 'ERB' in embeddings:
            erb_embeddings = embeddings['ERB']
            erb_positions = x_dict.get('ERB', torch.zeros(1, 2, device=rough_positions.device))[:, :2]
            
            try:
                refined_positions, confidence, offset = self.local_refiner(
                    pea_embeddings, rough_positions, erb_positions, erb_embeddings
                )
                
                high_confidence_mask = confidence.squeeze() > 0.7
                if high_confidence_mask.any():
                    final_positions = torch.where(
                        high_confidence_mask.unsqueeze(1),
                        refined_positions,
                        rough_positions
                    )
                    
                    refinement_info = {
                        'confidence': confidence,
                        'offset': offset,
                        'refined_count': high_confidence_mask.sum().item()
                    }
                    
                    logger.debug(f"🎯 Refinamento aplicado a {refinement_info['refined_count']} PEAs")
                
            except Exception as e:
                logger.warning(f"⚠️ Error in the local refinement: {e}")
                final_positions = rough_positions
        
        output = {
            'predictions': {
                'PEA': final_positions
            },
            'embeddings': embeddings,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'rough_positions': rough_positions,
            'refinement_info': refinement_info
        }
        
        return output
    
    def _prepare_batch_data(self, batch):
        x_dict = {}
        edge_index_dict = {}
        
        for node_type in self.node_types:
            try:
                if hasattr(batch, node_type) and hasattr(batch[node_type], 'x'):
                    x_dict[node_type] = batch[node_type].x
                elif node_type in batch.x_dict:
                    x_dict[node_type] = batch.x_dict[node_type]
                else:
                    logger.warning(f"Node {node_type} not found in the batch")
            except Exception as e:
                logger.warning(f"Error while extracting the features of node {node_type}: {e}")
        
        for edge_type in self.edge_types:
            try:
                if hasattr(batch, 'edge_index_dict') and edge_type in batch.edge_index_dict:
                    edge_index_dict[edge_type] = batch.edge_index_dict[edge_type]
                elif hasattr(batch, edge_type) and hasattr(batch[edge_type], 'edge_index'):
                    edge_index_dict[edge_type] = batch[edge_type].edge_index
                else:
                    src_type, _, dst_type = edge_type
                    if src_type in x_dict and dst_type in x_dict:
                        edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=x_dict[src_type].device)
            except Exception as e:
                logger.debug(f"Error while extracting edges {edge_type}: {e}")
        
        logger.debug(f"Batch prepared: {len(x_dict)} node types, {len(edge_index_dict)} edge types")
        
        return x_dict, edge_index_dict
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'model_size_mb': model_size_mb,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'architecture': 'Multi-Scale GNN Robusto'
        }


def create_model(config, graph_data, use_ensemble=False, **kwargs):
    logger.info('🏗️ Creating the ROBUST ARPIA GNN v4 model...')
    
    node_feature_dims = {}
    for node_type in ['PEA', 'ERB', 'CPS', 'CTC', 'CTNC']:
        if node_type in graph_data.node_types:
            node_feature_dims[node_type] = graph_data[node_type].x.shape[1]
    
    model = ARPIAGNNv4(config)
    
    model.initialize_encoder_with_dims(node_feature_dims)
    
    logger.info('✅ ROBUST ARPIA GNN v4 model created successfully!')
    
    return model


def extract_pred_positions(output):
    if isinstance(output, dict) and 'predictions' in output:
        return output['predictions'].get('PEA')
    return output

def prepare_batch_positions(batch, device):
    if hasattr(batch['PEA'], 'x'):
        return batch['PEA'].x[:, :2].to(device)
    return None 
