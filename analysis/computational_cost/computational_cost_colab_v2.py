#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Computational cost of ARPIA and of the baseline GNNs on Colab: parameter count, inference time and peak GPU memory."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import gc
import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("🔬 ARPIA V3 — COMPUTATIONAL COST ANALYSIS v2 (Colab VRAM-Safe)")
print("=" * 70)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

_POSSIBLE_DIRS = [
    Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL"),
    Path('/content/drive/MyDrive/TREINAMENTO_REAL'),
    Path('/content/TREINAMENTO_REAL'),
]

PIPELINE_DIR = None
for _d in _POSSIBLE_DIRS:
    if (_d / 'arpia_hetero_graph_v3.pt').exists():
        PIPELINE_DIR = _d
        break

if PIPELINE_DIR is None:
    PIPELINE_DIR = _POSSIBLE_DIRS[0]
    print(f"⚠️  Graph NOT found in any of the paths:")
    for _d in _POSSIBLE_DIRS:
        print(f"   ❌ {_d / 'arpia_hetero_graph_v3.pt'}")
    print(f"   Usando fallback: {PIPELINE_DIR}")
else:
    print(f"✅ Pipeline dir encontrado: {PIPELINE_DIR}")

GRAPH_PATH   = PIPELINE_DIR / 'arpia_hetero_graph_v3.pt'
OUTPUT_DIR   = Path(f"{INTERNAL_DATA_ROOT}/RESULTADOS/computational_cost")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GATv2Conv, HeteroConv, GATConv
try:
    from torch_geometric.nn import HANConv
    HAN_AVAILABLE = True
    print("✅ HANConv disponível")
except ImportError:
    HAN_AVAILABLE = False
    print('⚠️  HANConv not available — HAN will be skipped')

from models import ARPIAGNNv4
print("✅ ARPIAGNNv4 importado de models.py")
sys.stdout.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger('comp_cost')
logger.setLevel(logging.INFO)

OFFICIAL_ARCH = {
    'hidden_channels': 512,
    'out_channels': 1024,
    'num_layers': 4,
    'heads': 8,
    'dropout': 0.1,
}

NODE_TYPES = ['PEA', 'ERB', 'CPS', 'CTC', 'CTNC']
EDGE_TYPES = [
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
    ('CPS', 'INTRA_TRIANGULO', 'CTNC'),
]

NUM_WARMUP = 10
NUM_RUNS   = 50

TRAINING_RESULTS = {
    'ARPIA': {
        'mae': 7.55, 'rmse': 8.32, 'median': 7.41,
        'acc100': 1.0, 'training_time_min': 672.0,
        'epochs': 300, 'expected_params_M': 97.6,
    },
    'GATv2': {
        'mae': 156.58, 'rmse': 170.90, 'median': 135.70,
        'acc100': 0.217, 'training_time_min': 1196.6,
        'epochs': 259, 'expected_params_M': 41.6,
    },
    'HAN': {
        'mae': 168.77, 'rmse': 186.32, 'median': 176.60,
        'acc100': 0.261, 'training_time_min': 801.2,
        'epochs': 259, 'expected_params_M': 8.0,
    },
    'RGCN': {
        'mae': 120.91, 'rmse': 136.31, 'median': 112.80,
        'acc100': 0.391, 'training_time_min': 728.9,
        'epochs': 398, 'expected_params_M': 41.6,
    },
}


class BaselineGNN(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_layers, heads, dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.node_feature_dims = {}
        self.node_encoders = None

        self.location_decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 2),
        )
        self.output_proj = None

    def initialize_encoder_with_dims(self, node_feature_dims):
        self.node_feature_dims = node_feature_dims.copy()
        self.node_encoders = nn.ModuleDict()
        for nt, dim in self.node_feature_dims.items():
            self.node_encoders[nt] = nn.Sequential(
                nn.Linear(dim, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
        if self.hidden_channels != self.out_channels:
            self.output_proj = nn.Linear(self.hidden_channels, self.out_channels)
        self._init_convolutions()
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"  [{self.__class__.__name__}] params={total:,}")
        return self

    def _init_convolutions(self):
        raise NotImplementedError

    def _validate_edges(self, edge_index_dict, h_dict):
        valid = {}
        for et, ei in edge_index_dict.items():
            if ei.numel() > 0:
                src_t, _, dst_t = et
                if src_t in h_dict and dst_t in h_dict:
                    max_src = h_dict[src_t].size(0) - 1
                    max_dst = h_dict[dst_t].size(0) - 1
                    valid[et] = torch.stack([
                        torch.clamp(ei[0], 0, max_src),
                        torch.clamp(ei[1], 0, max_dst),
                    ])
                else:
                    valid[et] = torch.empty((2, 0), dtype=torch.long, device=ei.device)
            else:
                valid[et] = ei
        return valid

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            nt: self.node_encoders[nt](x)
            for nt, x in x_dict.items()
            if nt in self.node_encoders
        }
        h_dict = self._convolve(h_dict, edge_index_dict)
        if self.output_proj is not None:
            h_dict = {nt: self.output_proj(h) for nt, h in h_dict.items()}
        pea = h_dict.get('PEA')
        if pea is not None:
            return self.location_decoder(pea)
        return None


class GATv2Model(BaselineGNN):

    def _init_convolutions(self):
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict = {
                et: GATv2Conv(
                    in_channels=(self.hidden_channels, self.hidden_channels),
                    out_channels=self.hidden_channels // self.heads,
                    heads=self.heads,
                    concat=True,
                    dropout=self.dropout,
                    add_self_loops=False,
                    share_weights=False,
                )
                for et in EDGE_TYPES
                if et[0] in self.node_feature_dims and et[2] in self.node_feature_dims
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(self.hidden_channels)
                for nt in self.node_feature_dims
            }))

    def _convolve(self, h_dict, edge_index_dict):
        edge_index_dict = self._validate_edges(edge_index_dict, h_dict)
        for i, conv in enumerate(self.convs):
            valid = {et: ei for et, ei in edge_index_dict.items()
                     if ei.numel() > 0 and et[0] in h_dict and et[2] in h_dict}
            if not valid:
                continue
            h_new = conv(h_dict, valid)
            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)
        return h_dict


class HANModel(BaselineGNN):

    def _init_convolutions(self):
        metadata = (list(self.node_feature_dims.keys()), EDGE_TYPES)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(HANConv(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                metadata=metadata,
                heads=self.heads,
                dropout=self.dropout,
            ))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(self.hidden_channels)
                for nt in self.node_feature_dims
            }))

    def _convolve(self, h_dict, edge_index_dict):
        edge_index_dict = self._validate_edges(edge_index_dict, h_dict)
        for i, conv in enumerate(self.convs):
            try:
                h_new = conv(h_dict, edge_index_dict)
            except Exception:
                continue
            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)
        return h_dict


class RGCNModel(BaselineGNN):

    def _init_convolutions(self):
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict = {
                et: GATConv(
                    in_channels=(self.hidden_channels, self.hidden_channels),
                    out_channels=self.hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                )
                for et in EDGE_TYPES
                if et[0] in self.node_feature_dims and et[2] in self.node_feature_dims
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(self.hidden_channels)
                for nt in self.node_feature_dims
            }))

    def _convolve(self, h_dict, edge_index_dict):
        edge_index_dict = self._validate_edges(edge_index_dict, h_dict)
        for i, conv in enumerate(self.convs):
            valid = {et: ei for et, ei in edge_index_dict.items()
                     if ei.numel() > 0 and et[0] in h_dict and et[2] in h_dict}
            if not valid:
                continue
            h_new = conv(h_dict, valid)
            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)
        return h_dict


def create_arpia_model(graph_data, device):
    config = {
        'model': {
            'hidden_channels': OFFICIAL_ARCH['hidden_channels'],
            'out_channels': OFFICIAL_ARCH['out_channels'],
            'num_layers': OFFICIAL_ARCH['num_layers'],
            'heads': OFFICIAL_ARCH['heads'],
            'dropout': OFFICIAL_ARCH['dropout'],
            'use_local_refiner': True,
        }
    }
    model = ARPIAGNNv4(config)
    node_feature_dims = {
        nt: graph_data[nt].x.shape[1]
        for nt in NODE_TYPES if nt in graph_data.node_types
    }
    model.initialize_encoder_with_dims(node_feature_dims)
    return model.to(device)


def create_baseline(cls, graph_data, device):
    model = cls(
        hidden_channels=OFFICIAL_ARCH['hidden_channels'],
        out_channels=OFFICIAL_ARCH['out_channels'],
        num_layers=OFFICIAL_ARCH['num_layers'],
        heads=OFFICIAL_ARCH['heads'],
        dropout=OFFICIAL_ARCH['dropout'],
    )
    node_feature_dims = {
        nt: graph_data[nt].x.shape[1]
        for nt in NODE_TYPES if nt in graph_data.node_types
    }
    model.initialize_encoder_with_dims(node_feature_dims)
    return model.to(device)


def full_cuda_reset(device):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
    time.sleep(2)

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        logger.info(f"  VRAM after reset: {used:.0f}MB / {total:.0f}MB ({used/total*100:.1f}%)")


def check_vram_available(device, required_mb):
    if not torch.cuda.is_available():
        return True
    free = (torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)) / (1024 ** 2)
    if free < required_mb:
        logger.warning(f"  ⚠️ Not enough VRAM! Free: {free:.0f}MB, Required: ~{required_mb:.0f}MB")
        return False
    logger.info(f"  Free VRAM: {free:.0f}MB (required: ~{required_mb:.0f}MB) ✅")
    return True


def measure_model(model, graph_data, device, model_name, is_arpia=False):
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)

    expected = TRAINING_RESULTS[model_name]['expected_params_M']
    actual = total_params / 1e6
    logger.info(f"  Param check: expected ~{expected:.1f}M, got {actual:.2f}M")
    logger.info(f"  Size: {size_mb:.1f} MB (float32)")

    if is_arpia:
        graph_data_gpu = graph_data.to(device)
    else:
        x_dict = {nt: graph_data[nt].x.to(device) for nt in graph_data.node_types}
        edge_index_dict = {et: graph_data[et].edge_index.to(device)
                          for et in graph_data.edge_types}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    def run_forward():
        if is_arpia:
            return model(graph_data_gpu)
        else:
            return model(x_dict, edge_index_dict)

    with torch.no_grad():
        logger.info(f"  Warmup ({NUM_WARMUP} runs)...")
        for _ in range(NUM_WARMUP):
            _ = run_forward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"  Medindo ({NUM_RUNS} runs)...")
        times_ms = []

        if torch.cuda.is_available():
            for _ in range(NUM_RUNS):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt   = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _ = run_forward()
                end_evt.record()
                torch.cuda.synchronize()
                times_ms.append(start_evt.elapsed_time(end_evt))
        else:
            for _ in range(NUM_RUNS):
                t0 = time.perf_counter()
                _ = run_forward()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000)

    times_ms = np.array(times_ms)
    avg_ms    = float(np.mean(times_ms))
    std_ms    = float(np.std(times_ms))
    median_ms = float(np.median(times_ms))
    p95_ms    = float(np.percentile(times_ms, 95))
    min_ms    = float(np.min(times_ms))
    max_ms    = float(np.max(times_ms))

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if torch.cuda.is_available() else 0)

    total_nodes = sum(graph_data[nt].x.shape[0] for nt in graph_data.node_types)
    n_pea = graph_data['PEA'].x.shape[0]
    throughput = n_pea / (avg_ms / 1000) if avg_ms > 0 else 0

    logger.info(f"  Inference: {avg_ms:.2f} ± {std_ms:.2f} ms")
    logger.info(f"    median={median_ms:.2f}ms  p95={p95_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms")
    logger.info(f"  Peak memory: {peak_mem_mb:.1f} MB")
    logger.info(f"  Throughput: {throughput:,.0f} PEAs/s")

    if is_arpia:
        del graph_data_gpu
    else:
        del x_dict, edge_index_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'params_M': round(total_params / 1e6, 2),
        'params_total': total_params,
        'trainable_params': trainable_params,
        'size_mb': round(size_mb, 1),
        'inference': {
            'mean_ms': round(avg_ms, 2),
            'std_ms': round(std_ms, 2),
            'median_ms': round(median_ms, 2),
            'p95_ms': round(p95_ms, 2),
            'min_ms': round(min_ms, 2),
            'max_ms': round(max_ms, 2),
            'total_nodes': total_nodes,
            'n_pea': n_pea,
            'peak_memory_mb': round(peak_mem_mb, 1),
            'throughput_peas_s': round(throughput, 1),
        }
    }


def generate_figures(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

    model_names = [n for n in all_results['results'] if 'error' not in all_results['results'][n]]
    colors  = {'ARPIA': '#1f77b4', 'GATv2': '#ff7f0e', 'HAN': '#2ca02c', 'RGCN': '#d62728'}
    markers = {'ARPIA': 'o', 'GATv2': 's', 'HAN': '^', 'RGCN': 'D'}

    all_params = [all_results['results'][n]['params_M'] for n in model_names]
    all_maes   = [TRAINING_RESULTS[n]['mae'] for n in model_names]
    all_rmses  = [TRAINING_RESULTS[n]['rmse'] for n in model_names]

    def _annotate(ax, name, x, y, colors_map):
        if name == 'ARPIA':
            offset, va = (0, 20), 'bottom'
        elif name == 'GATv2':
            offset, va = (15, -25), 'top'
        elif name == 'HAN':
            offset, va = (-15, 20), 'bottom'
        else:
            offset, va = (15, 20), 'bottom'
        ax.annotate(
            f"{name}\n({y:.2f}m | {x:.1f}M)",
            (x, y), xytext=offset, textcoords='offset points',
            fontsize=11, fontweight='bold', ha='center', va=va,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=colors_map[name], alpha=0.9, lw=1.5)
        )

    fig, ax = plt.subplots(figsize=(14, 9))
    for name in model_names:
        r = all_results['results'][name]
        ax.scatter(r['params_M'], TRAINING_RESULTS[name]['mae'], s=350,
                   color=colors[name], marker=markers[name],
                   alpha=0.9, edgecolors='black', linewidth=1.5, label=name, zorder=10)
        _annotate(ax, name, r['params_M'], TRAINING_RESULTS[name]['mae'], colors)

    ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (meters)', fontweight='bold')
    ax.set_title('Performance vs. Model Complexity (MAE)\nLower Error & Fewer Parameters = Better',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.95, shadow=True, borderpad=1.2, labelspacing=1.5)
    ax.set_xlim(min(all_params) * 0.5, max(all_params) * 1.3)
    ax.set_ylim(0, max(all_maes) * 1.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'performance_vs_parameters_mean_error.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  ✅ performance_vs_parameters_mean_error.png")

    fig, ax = plt.subplots(figsize=(14, 9))
    for name in model_names:
        r = all_results['results'][name]
        ax.scatter(r['params_M'], TRAINING_RESULTS[name]['rmse'], s=350,
                   color=colors[name], marker=markers[name],
                   alpha=0.9, edgecolors='black', linewidth=1.5, label=name, zorder=10)
        _annotate(ax, name, r['params_M'], TRAINING_RESULTS[name]['rmse'], colors)

    ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Root Mean Square Error (meters)', fontweight='bold')
    ax.set_title('Performance vs. Model Complexity (RMSE)\nLower Error & Fewer Parameters = Better',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.95, shadow=True, borderpad=1.2, labelspacing=1.5)
    ax.set_xlim(min(all_params) * 0.5, max(all_params) * 1.3)
    ax.set_ylim(0, max(all_rmses) * 1.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'performance_vs_parameters_rmse.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  ✅ performance_vs_parameters_rmse.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    x = np.arange(len(model_names))
    bar_colors = [colors[n] for n in model_names]

    mae_per_M = [TRAINING_RESULTS[n]['mae'] / all_results['results'][n]['params_M'] for n in model_names]
    axes[0].bar(x, mae_per_M, color=bar_colors, edgecolor='white', width=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel('MAE / Million Params (m/M)')
    axes[0].set_title('Parameter Efficiency\n(lower is better)')
    for i, v in enumerate(mae_per_M):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

    inf_times = [all_results['results'][n]['inference']['mean_ms'] for n in model_names]
    axes[1].bar(x, inf_times, color=bar_colors, edgecolor='white', width=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel('Inference Time (ms)')
    axes[1].set_title('Inference Speed\n(lower is better)')
    for i, v in enumerate(inf_times):
        axes[1].text(i, v + max(inf_times) * 0.02, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

    train_times = [TRAINING_RESULTS[n]['training_time_min'] for n in model_names]
    axes[2].bar(x, train_times, color=bar_colors, edgecolor='white', width=0.6)
    axes[2].set_xticks(x); axes[2].set_xticklabels(model_names)
    axes[2].set_ylabel('Training Time (min)')
    axes[2].set_title('Training Cost\n(lower is better)')
    for i, v in enumerate(train_times):
        axes[2].text(i, v + max(train_times) * 0.02, f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')

    fig.suptitle('Computational Efficiency Comparison', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'computational_efficiency_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  ✅ computational_efficiency_comparison.png")

    fig, ax = plt.subplots(figsize=(14, 9))
    for name in model_names:
        r = all_results['results'][name]
        params = r['params_M']
        mae = TRAINING_RESULTS[name]['mae']
        inf_ms = r['inference']['mean_ms']
        bubble_size = inf_ms * 1.5
        ax.scatter(params, mae, s=bubble_size, color=colors[name], marker='o',
                   alpha=0.6, edgecolors=colors[name], linewidth=2,
                   label=f"{name} ({inf_ms:.0f}ms)", zorder=10)
        ax.annotate(
            f"{name}\n{mae:.1f}m | {params:.1f}M\n{inf_ms:.0f}ms",
            (params, mae), fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[name], alpha=0.85)
        )
    ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (meters)', fontweight='bold')
    ax.set_title('Performance-Complexity-Speed Trade-off\n(bubble size = inference time)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.95)
    ax.set_xlim(min(all_params) * 0.5, max(all_params) * 1.3)
    ax.set_ylim(0, max(all_maes) * 1.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'performance_complexity_speed_bubble.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  ✅ performance_complexity_speed_bubble.png")


def _log(msg):
    print(msg)
    sys.stdout.flush()
    logger.info(msg)


def main():
    _log("=" * 70)
    _log('🔬 STARTING COMPUTATIONAL COST MEASUREMENT...')
    _log("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        _log(f"🖥️ GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        gpu_name = "CPU"
        gpu_mem_gb = 0
        _log('⚠️ CUDA not available — running on the CPU')

    _log(f"🔧 PyTorch: {torch.__version__}")
    _log(f"📐 Arquitetura: hidden={OFFICIAL_ARCH['hidden_channels']} "
         f"out={OFFICIAL_ARCH['out_channels']} "
         f"layers={OFFICIAL_ARCH['num_layers']} heads={OFFICIAL_ARCH['heads']}")

    _log(f"\n📊 Loading graph: {GRAPH_PATH}")
    graph_data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
    for nt in graph_data.node_types:
        _log(f"  {nt}: {graph_data[nt].x.shape}")

    n_pea = graph_data['PEA'].x.shape[0]
    total_nodes = sum(graph_data[nt].x.shape[0] for nt in graph_data.node_types)
    total_edges = sum(graph_data[et].edge_index.shape[1] for et in graph_data.edge_types)
    _log(f"  Total: {total_nodes:,} nodes, {total_edges:,} edges, {n_pea:,} PEAs")

    all_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'gpu': gpu_name,
            'gpu_mem_gb': round(gpu_mem_gb, 1),
            'pytorch': torch.__version__,
            'architecture': OFFICIAL_ARCH,
            'graph': str(GRAPH_PATH.name),
            'warmup': NUM_WARMUP,
            'runs': NUM_RUNS,
            'node_counts': {nt: graph_data[nt].x.shape[0] for nt in graph_data.node_types},
            'total_edges': total_edges,
        },
        'results': {},
        'training_results': TRAINING_RESULTS,
    }

    model_specs = [
        ('ARPIA', None,      True,  1500),
        ('GATv2', GATv2Model, False, 800),
        ('HAN',   HANModel if HAN_AVAILABLE else None, False, 400),
        ('RGCN',  RGCNModel, False, 800),
    ]

    for idx, (name, cls, is_arpia, est_vram) in enumerate(model_specs):
        _log(f"\n{'=' * 50}")
        _log(f"📐 [{idx + 1}/{len(model_specs)}] {name}")
        _log(f"{'=' * 50}")

        if cls is None and not is_arpia:
            _log(f"  ⏭️ Skipped ({name} not available)")
            all_results['results'][name] = {'error': f'{name} not available'}
            continue

        full_cuda_reset(device)

        if not check_vram_available(device, est_vram):
            _log(f"  ❌ VRAM insuficiente para {name}! Pulando.")
            all_results['results'][name] = {'error': f'Insufficient VRAM (need ~{est_vram}MB)'}
            continue

        try:
            _log(f"  🔨 Building model {name}...")
            if is_arpia:
                model = create_arpia_model(graph_data, device)
            else:
                model = create_baseline(cls, graph_data, device)

            _log(f"  ⏱️ Measuring inference ({NUM_WARMUP} warmup + {NUM_RUNS} runs)...")
            results = measure_model(model, graph_data, device, name, is_arpia=is_arpia)
            results['mae']               = TRAINING_RESULTS[name]['mae']
            results['rmse']              = TRAINING_RESULTS[name]['rmse']
            results['training_time_min'] = TRAINING_RESULTS[name]['training_time_min']
            results['mae_per_M']         = round(TRAINING_RESULTS[name]['mae'] / results['params_M'], 4)

            all_results['results'][name] = results
            inf = results['inference']
            _log(f"  ✅ {name} done! Params={results['params_M']:.1f}M | Inf={inf['mean_ms']:.2f}±{inf['std_ms']:.2f}ms | Mem={inf['peak_memory_mb']:.0f}MB")

            del model
            full_cuda_reset(device)

        except Exception as e:
            _log(f"  ❌ ERROR in {name}: {e}")
            traceback.print_exc()
            all_results['results'][name] = {'error': str(e)}
            try:
                del model
            except NameError:
                pass
            full_cuda_reset(device)

    _log(f"\n{'=' * 90}")
    _log("📊 RESUMO FINAL")
    _log(f"{'=' * 90}")
    header = (f"{'Arch':<10} | {'Params(M)':>10} | {'MAE(m)':>8} | "
              f"{'Inf(ms)':>10} | {'Std(ms)':>8} | {'PeakMem':>10} | "
              f"{'MAE/M':>8} | {'Train(min)':>11}")
    _log(header)
    _log("-" * len(header))

    for name, res in all_results['results'].items():
        if 'error' in res:
            _log(f"{name:<10} | {'ERRO: ' + res['error'][:40]}")
            continue
        inf = res['inference']
        _log(
            f"{name:<10} | {res['params_M']:>10.2f} | {res['mae']:>8.2f} | "
            f"{inf['mean_ms']:>10.2f} | {inf['std_ms']:>8.2f} | "
            f"{inf['peak_memory_mb']:>8.1f}MB | "
            f"{res['mae_per_M']:>8.4f} | {res['training_time_min']:>11.1f}"
        )

    output_file = OUTPUT_DIR / 'computational_cost_results_colab_v2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    _log(f"\n💾 Results saved: {output_file}")

    valid_count = sum(1 for r in all_results['results'].values() if 'error' not in r)
    if valid_count >= 2:
        _log("\n📈 Gerando figuras...")
        try:
            generate_figures(all_results)
            _log("✅ Todas as figuras geradas!")
        except Exception as e:
            _log(f"❌ Error while generating figures: {e}")
            traceback.print_exc()
    else:
        _log('⚠️ Fewer than 2 valid models — figures not generated')

    _log(f"\n🏁 DONE! Results in: {OUTPUT_DIR}")
    return all_results


if __name__ == '__main__':
    results = main()
