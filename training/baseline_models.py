#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline architecture comparison on the same graph and splits."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import copy
import json
import time
import gc
import logging
import traceback
from pathlib import Path
from datetime import datetime

print("=" * 70)
print('🔬 ARPIA V3 — COMPARISON OF BASELINE ARCHITECTURES')
print("=" * 70)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

try:
    import google.colab
    IN_COLAB = True
    print("✅ Google Colab detectado")
except ImportError:
    IN_COLAB = False
    print("🏠 Ambiente local")
sys.stdout.flush()

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive montado")
    sys.stdout.flush()

PIPELINE_DIR = f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL"
if not Path(PIPELINE_DIR).exists():
    raise FileNotFoundError(
        f"❌ Directory not found: {PIPELINE_DIR}\nCheck that Google Drive is mounted correctly."
    )
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
os.chdir(PIPELINE_DIR)
print(f"✅ Pipeline dir: {PIPELINE_DIR}")
sys.stdout.flush()

import torch.optim.lr_scheduler as _lr_sched
_OriginalPlateau = _lr_sched.ReduceLROnPlateau
class _PatchedPlateau(_OriginalPlateau):
    def __init__(self, *args, **kwargs):
        kwargs.pop('verbose', None)
        super().__init__(*args, **kwargs)
_lr_sched.ReduceLROnPlateau = _PatchedPlateau
print("🔧 ReduceLROnPlateau patched (PyTorch 2.x)")
sys.stdout.flush()

print("📦 Importando módulos...")
sys.stdout.flush()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, HeteroConv, GATConv

try:
    from torch_geometric.nn import HANConv
    HAN_AVAILABLE = True
    print("✅ HANConv disponível")
except ImportError:
    HAN_AVAILABLE = False
    print("⚠️  HANConv not available — experiment 'han' will be skipped")
sys.stdout.flush()

import config_colab
import data_loaders
import models
import losses
import trainers
import train_colab

trainers.ReduceLROnPlateau = _PatchedPlateau

from config_colab import create_custom_config
from data_loaders import prepare_graph_data, create_data_loaders
from losses import create_physics_informed_loss
from train_colab import create_enhanced_trainer_with_monitoring, create_final_summary_corrected

print('✅ All modules were imported')
sys.stdout.flush()

logger = logging.getLogger(__name__)


class BaselineGNN(nn.Module):

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

    def __init__(self, config):
        super().__init__()
        model_config = (
            config.get('model', {}) if isinstance(config, dict)
            else config.model if hasattr(config, 'model')
            else {}
        )

        if isinstance(model_config, dict):
            self.hidden_channels = model_config.get('hidden_channels', 512)
            self.out_channels    = model_config.get('out_channels', 1024)
            self.num_layers      = model_config.get('num_layers', 4)
            self.dropout         = model_config.get('dropout', 0.1)
            self.heads           = model_config.get('heads', 8)
        else:
            self.hidden_channels = getattr(model_config, 'hidden_channels', 512)
            self.out_channels    = getattr(model_config, 'out_channels', 1024)
            self.num_layers      = getattr(model_config, 'num_layers', 4)
            self.dropout         = getattr(model_config, 'dropout', 0.1)
            self.heads           = getattr(model_config, 'heads', 8)

        self.node_types       = self.NODE_TYPES
        self.edge_types       = self.EDGE_TYPES
        self.node_feature_dims = {}

        self.node_encoders = None

        self.location_decoder = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_channels, 2),
        )

        self.output_proj = None

    def initialize_encoder_with_dims(self, node_feature_dims: dict):
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
        logger.info(
            f"[{self.__class__.__name__}] inicializado | "
            f"hidden={self.hidden_channels} out={self.out_channels} "
            f"layers={self.num_layers} heads={self.heads} | "
            f"params={total:,}"
        )
        return self

    def _init_convolutions(self):
        raise NotImplementedError

    def _convolve(self, h_dict, edge_index_dict):
        raise NotImplementedError

    def _prepare_batch_data(self, batch):
        x_dict         = {}
        edge_index_dict = {}

        for node_type in self.node_types:
            try:
                if hasattr(batch, node_type) and hasattr(batch[node_type], 'x'):
                    x_dict[node_type] = batch[node_type].x
                elif hasattr(batch, 'x_dict') and node_type in batch.x_dict:
                    x_dict[node_type] = batch.x_dict[node_type]
            except Exception:
                pass

        for edge_type in self.edge_types:
            try:
                if (hasattr(batch, 'edge_index_dict')
                        and edge_type in batch.edge_index_dict):
                    edge_index_dict[edge_type] = batch.edge_index_dict[edge_type]
                elif (hasattr(batch, edge_type)
                        and hasattr(batch[edge_type], 'edge_index')):
                    edge_index_dict[edge_type] = batch[edge_type].edge_index
                else:
                    src_type, _, dst_type = edge_type
                    if src_type in x_dict and dst_type in x_dict:
                        edge_index_dict[edge_type] = torch.empty(
                            (2, 0), dtype=torch.long,
                            device=x_dict[src_type].device
                        )
            except Exception:
                pass

        return x_dict, edge_index_dict

    def _encode_nodes(self, x_dict):
        return {
            nt: self.node_encoders[nt](x)
            for nt, x in x_dict.items()
            if nt in self.node_encoders
        }

    def _project_output(self, h_dict):
        if self.output_proj is not None:
            return {nt: self.output_proj(h) for nt, h in h_dict.items()}
        return h_dict

    def _validate_edges(self, edge_index_dict, h_dict):
        valid = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() > 0:
                src_type, _, dst_type = edge_type
                if src_type in h_dict and dst_type in h_dict:
                    max_src = h_dict[src_type].size(0) - 1
                    max_dst = h_dict[dst_type].size(0) - 1
                    if edge_index.size(0) >= 2:
                        valid[edge_type] = torch.stack([
                            torch.clamp(edge_index[0], 0, max_src),
                            torch.clamp(edge_index[1], 0, max_dst),
                        ])
                    else:
                        valid[edge_type] = edge_index
                else:
                    valid[edge_type] = torch.empty(
                        (2, 0), dtype=torch.long, device=edge_index.device
                    )
            else:
                valid[edge_type] = edge_index
        return valid

    def forward(self, batch):
        if self.node_encoders is None:
            raise RuntimeError("Chame initialize_encoder_with_dims() antes de forward()!")

        x_dict, edge_index_dict = self._prepare_batch_data(batch)
        h_dict  = self._encode_nodes(x_dict)
        h_dict  = self._convolve(h_dict, edge_index_dict)
        embeddings = self._project_output(h_dict)

        pea = embeddings.get('PEA')
        if pea is None:
            raise ValueError('PEA embeddings not found in the forward pass.')

        positions = self.location_decoder(pea)

        return {
            'predictions':     {'PEA': positions},
            'embeddings':      embeddings,
            'node_types':      self.node_types,
            'edge_types':      self.edge_types,
            'rough_positions': positions,
            'refinement_info': {},
        }


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
                for et in self.edge_types
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
            valid_edges = {
                et: ei for et, ei in edge_index_dict.items()
                if ei.numel() > 0 and et[0] in h_dict and et[2] in h_dict
            }
            if not valid_edges:
                continue

            h_new = conv(h_dict, valid_edges)

            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout,
                                           training=self.training)

        return h_dict


class HANModel(BaselineGNN):

    def _init_convolutions(self):
        metadata = (list(self.node_feature_dims.keys()), self.edge_types)

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
            except Exception as e:
                logger.warning(f"HANConv camada {i} falhou ({e}) — pulando")
                continue

            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout,
                                           training=self.training)

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
                for et in self.edge_types
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
            valid_edges = {
                et: ei for et, ei in edge_index_dict.items()
                if ei.numel() > 0 and et[0] in h_dict and et[2] in h_dict
            }
            if not valid_edges:
                continue

            h_new = conv(h_dict, valid_edges)

            for nt in h_dict:
                if nt in h_new:
                    h_dict[nt] = self.norms[i][nt](h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.elu(h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout,
                                           training=self.training)

        return h_dict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler()],
)


CHECKPOINT_FREQUENCY = 20
TOTAL_EPOCHS         = 300

OFFICIAL_ARCH = {
    'hidden_channels': 512,
    'num_layers':      4,
    'heads':           8,
    'out_channels':    1024,
    'dropout':         0.1,
}

BASELINE_CLASSES = {
    'gatv2': GATv2Model,
    'han':   HANModel,
    'rgcn':  RGCNModel,
}

EXPERIMENTS_TO_RUN = [
    'gatv2',
    'han',
    'rgcn',
]


def cleanup_memory(stage: str = '') -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    if stage:
        logger.info(f"🧹 Memory cleaned up: {stage}")


def add_reverse_edges(graph_data):
    added = []
    for (src, rel, dst) in list(graph_data.edge_types):
        rev_et = (dst, f'rev_{rel}', src)
        if rev_et not in graph_data.edge_types:
            ei = graph_data[src, rel, dst].edge_index
            graph_data[rev_et].edge_index = ei.flip(0)
            added.append(f'{src}-{rel}-{dst}')
    if added:
        logger.info(f'✅ Reverse edges adicionados: {added}')
    return graph_data

    from data_loaders import prepare_graph_data

    from data_loaders import prepare_graph_data


def load_base_data(pipeline_dir: Path, device):
    logger.info('📊 Loading the base data...')
    graph_data = torch.load(
        pipeline_dir / 'arpia_hetero_graph_v3.pt', map_location='cpu'
    )
    graph_data = prepare_graph_data(graph_data, device)
    graph_data = add_reverse_edges(graph_data)

    scalers = {}
    try:
        scalers = torch.load(pipeline_dir / 'scalers.pkl', map_location='cpu')
        logger.info('✅ Scalers carregados (torch.load)')
    except Exception:
        try:
            import pickle
            with open(pipeline_dir / 'scalers.pkl', 'rb') as fh:
                scalers = pickle.load(fh)
            logger.info('✅ Scalers carregados (pickle)')
        except Exception as e:
            logger.warning(f"⚠️ Scalers were not loaded: {e}")

    for nt in ['PEA', 'ERB', 'CPS', 'CTC', 'CTNC']:
        if nt in graph_data.node_types:
            logger.info(f'   {nt}: {graph_data[nt].x.shape}')

    return graph_data, scalers


def find_latest_checkpoint(save_dir: Path):
    ckpt_dir = save_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None, 0
    checkpoints = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    try:
        epoch = int(latest.stem.split('_')[-1])
    except (ValueError, IndexError):
        epoch = 0
    return str(latest), epoch


def save_periodic_checkpoint(trainer, save_dir: Path, epoch: int) -> None:
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filepath = ckpt_dir / f'checkpoint_epoch_{epoch:04d}.pt'
    trainer.save_checkpoint(str(filepath), epoch, metrics=trainer.best_metrics)
    logger.info(f'💾 Checkpoint salvo: {filepath.name}')
    all_ckpts = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    while len(all_ckpts) > 3:
        oldest = all_ckpts.pop(0)
        oldest.unlink()
        logger.info(f'🗑️  Checkpoint removido: {oldest.name}')


def create_baseline_model(model_name: str, config, graph_data):
    model_class = BASELINE_CLASSES[model_name]
    model = model_class(config.to_dict() if hasattr(config, 'to_dict') else config)

    node_feature_dims = {
        nt: graph_data[nt].x.shape[1]
        for nt in ['PEA', 'ERB', 'CPS', 'CTC', 'CTNC']
        if nt in graph_data.node_types
    }
    model.initialize_encoder_with_dims(node_feature_dims)
    return model


def _training_loop(trainer, data_loaders, save_dir: Path,
                   total_epochs: int, start_epoch: int, exp_name: str):
    history = trainer.training_history

    for epoch in range(start_epoch, total_epochs):
        try:
            if trainer.phase_manager.should_advance_phase():
                if not trainer.phase_manager.advance_phase():
                    logger.info('🏁 All phases are finished!')
                    break
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = trainer.phase_manager.current_phase.learning_rate

            train_loss = trainer.train_epoch(data_loaders['train'], epoch)

            val_metrics = None
            error       = None
            val_loader  = data_loaders.get('val')
            if val_loader is not None:
                val_metrics = trainer.validate_epoch(val_loader)
                error       = val_metrics.get('mean_error')

            trainer.phase_manager.step_epoch()

            current_epoch = epoch + 1
            history.setdefault('epoch', []).append(epoch)
            history.setdefault('phase', []).append(
                trainer.phase_manager.current_phase_number
            )
            history.setdefault('loss', []).append(train_loss)
            history.setdefault('metrics', []).append(val_metrics)
            history.setdefault('learning_rate', []).append(
                trainer.optimizer.param_groups[0]['lr']
            )
            history.setdefault('phase_info', []).append(
                trainer.phase_manager.get_current_info()
            )

            if (hasattr(trainer, 'eda_frequency')
                    and trainer.eda_frequency > 0
                    and current_epoch % trainer.eda_frequency == 0):
                try:
                    eda_dir = trainer.eda_config.get('save_dir', str(save_dir / 'eda'))
                    trainer.run_automated_eda(data_loaders['train'], current_epoch, eda_dir)
                except Exception as e:
                    logger.error(f"❌ EDA epoch {current_epoch}: {e}")

            if current_epoch % CHECKPOINT_FREQUENCY == 0:
                save_periodic_checkpoint(trainer, save_dir, current_epoch)

            if (val_metrics and
                    val_metrics.get('mean_error', float('inf')) <
                    trainer.best_metrics.get('error', float('inf'))):
                trainer.best_metrics['error'] = val_metrics['mean_error']
                trainer.patience_counter = 0
            else:
                trainer.patience_counter = getattr(trainer, 'patience_counter', 0) + 1
                max_patience = getattr(trainer, 'max_patience', 50)
                if trainer.patience_counter >= max_patience:
                    logger.info(
                        f"🛑 Early stopping after {trainer.patience_counter} epochs without improvement"
                    )
                    break

            if current_epoch % 5 == 0:
                phase_info = trainer.phase_manager.get_current_info()
                err_str = f'{error:.1f}m' if error is not None else 'N/A'
                lr_str  = f"{trainer.optimizer.param_groups[0]['lr']:.2e}"
                print(
                    f"  Epoch {current_epoch:3d}/{total_epochs} | Phase {phase_info.get('phase_number', '?')} | Loss {train_loss:.4f} | Error {err_str} | LR {lr_str}"
                )

        except KeyboardInterrupt:
            logger.info('🛑 Interrupted by the user')
            save_periodic_checkpoint(trainer, save_dir, epoch)
            break
        except Exception as e:
            logger.error(f"❌ Error in epoch {epoch}: {e}\n{traceback.format_exc()}")
            save_periodic_checkpoint(trainer, save_dir, epoch)
            continue

    save_periodic_checkpoint(trainer, save_dir, total_epochs)
    return history


def run_experiment(model_name: str, graph_data, scalers, output_root: Path, device):
    save_dir = output_root / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 72)
    print(f"🔬 COMPARISON: {model_name.upper()} ({BASELINE_CLASSES[model_name].__name__})")
    print(f'   Protocolo: hidden={OFFICIAL_ARCH["hidden_channels"]} '
          f'layers={OFFICIAL_ARCH["num_layers"]} '
          f'heads={OFFICIAL_ARCH["heads"]} '
          f'batch=128 epochs={TOTAL_EPOCHS}')
    print('=' * 72)

    if model_name == 'han':
        effective_batch    = 64
        effective_neighbors = [16, 12, 8]
        print('⚠️  HAN: batch_size reduced to 64 and num_neighbors=[16,12,8] (HANConv is ~2× heavier)')
    else:
        effective_batch    = 128
        effective_neighbors = [32, 24, 16]

    config_overrides = {
        'batch_size':       effective_batch,
        'hidden_channels':  OFFICIAL_ARCH['hidden_channels'],
        'out_channels':     OFFICIAL_ARCH['out_channels'],
        'num_layers':       OFFICIAL_ARCH['num_layers'],
        'heads':            OFFICIAL_ARCH['heads'],
        'dropout':          OFFICIAL_ARCH['dropout'],
        'num_neighbors':    effective_neighbors,
        'num_workers':      2,
        'pin_memory':       False,
        'persistent_workers': False,
        'total_epochs':     TOTAL_EPOCHS,
        'epochs':           TOTAL_EPOCHS,
        'eda_frequency':    20,
        'use_mixed_precision': True,
        'device':           str(device),
    }
    config = create_custom_config(**config_overrides)

    logger.info('📊 Criando data loaders...')
    data_loaders = create_data_loaders(
        graph_data,
        batch_size=effective_batch,
        num_neighbors=config.data.num_neighbors,
        num_workers=config.data.num_workers,
        device=device,
    )

    logger.info(f'🧠 Criando {BASELINE_CLASSES[model_name].__name__}...')
    model = create_baseline_model(model_name, config, graph_data)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📐 {model_name.upper()} — Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    logger.info('🔬 Criando loss function...')
    loss_fn = create_physics_informed_loss(config.to_dict())

    logger.info('🚀 Criando trainer...')
    trainer = create_enhanced_trainer_with_monitoring(
        model, loss_fn, config.to_dict(), device
    )
    trainer.set_scalers(scalers if scalers else {})
    trainer.eda_frequency = 20
    trainer.eda_config    = {'save_dir': str(save_dir / 'eda')}

    config_snapshot = {
        'experiment':   model_name,
        'model_class':  BASELINE_CLASSES[model_name].__name__,
        'architecture': {
            'hidden_channels': OFFICIAL_ARCH['hidden_channels'],
            'num_layers':      OFFICIAL_ARCH['num_layers'],
            'heads':           OFFICIAL_ARCH['heads'],
            'out_channels':    OFFICIAL_ARCH['out_channels'],
        },
        'total_params': total_params,
        'epochs':       TOTAL_EPOCHS,
        'batch_size':   effective_batch,
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_snapshot, f, indent=2)

    start_epoch = 0
    ckpt_path, ckpt_epoch = find_latest_checkpoint(save_dir)

    if ckpt_path and ckpt_epoch > 0:
        print(f"\n🔄 CHECKPOINT FOUND: epoch {ckpt_epoch}")
        try:
            start_epoch = trainer.load_checkpoint(ckpt_path)
            print(f"   ✅ Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f'   ⚠️ load_checkpoint falhou ({e}) — tentando carregamento manual...')
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print('   ✅ Model state carregado manualmente')
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as oe:
                    print(f'   ⚠️ Optimizer state ignorado: {oe}')
                start_epoch = ckpt.get('epoch', 0)
                trainer.training_history = ckpt.get(
                    'training_history', trainer.training_history
                )
                trainer.best_metrics = ckpt.get('best_metrics', trainer.best_metrics)
                pm_state = ckpt.get('phase_manager_state', {})
                if pm_state:
                    idx = pm_state.get('current_phase_idx', 1)
                    trainer.phase_manager.current_phase_idx = idx
                    trainer.phase_manager.epoch_in_phase = pm_state.get('epoch_in_phase', 0)
                    trainer.phase_manager.total_epochs_completed = pm_state.get(
                        'total_epochs_completed', start_epoch
                    )
                    if idx in trainer.phase_manager.phases:
                        trainer.phase_manager.current_phase = (
                            trainer.phase_manager.phases[idx]
                        )
                        print(f'   ✅ PhaseManager restaurado (fase {idx})')
                    else:
                        trainer.phase_manager.start_phase(1)
                print(f"   ✅ Manual resume succeeded — epoch {start_epoch}")
            except Exception as me:
                print(f"   ❌ Critical failure while loading: {me}")
                start_epoch = 0
    else:
        print('\n📝 No previous checkpoint — starting from scratch')

    remaining = TOTAL_EPOCHS - start_epoch
    if remaining <= 0:
        print(f"✅ {model_name} has already completed {TOTAL_EPOCHS} epochs!")
        result_file = save_dir / f'{model_name}_result.json'
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {'experiment': model_name, 'status': 'ALREADY_COMPLETE'}

    print(f"\n🚀 Training {model_name.upper()}: epochs {start_epoch + 1} → {TOTAL_EPOCHS} ({remaining} remaining)")
    print('=' * 72)

    cleanup_memory(f'antes de {model_name}')
    t0 = time.time()

    history = _training_loop(
        trainer=trainer,
        data_loaders=data_loaders,
        save_dir=save_dir,
        total_epochs=TOTAL_EPOCHS,
        start_epoch=start_epoch,
        exp_name=model_name,
    )

    training_time = (time.time() - t0) / 60.0

    logger.info('📊 Avaliação final...')
    evaluation_results = {}
    for split_name, loader in data_loaders.items():
        if loader is not None:
            try:
                evaluation_results[split_name] = (
                    trainer.evaluate_model_accurate_metrics(loader)
                )
            except Exception as e:
                logger.error(f'Erro avaliando {split_name}: {e}')

    final_loss = (history.get('loss') or [None])[-1]
    val_m  = evaluation_results.get('val', evaluation_results.get('test', {}))
    mae    = val_m.get('mean_error', val_m.get('mae'))
    rmse   = val_m.get('rmse')
    acc100 = val_m.get('accuracy_100m')
    max_err = val_m.get('max_error')

    result = {
        'experiment':   model_name,
        'model_class':  BASELINE_CLASSES[model_name].__name__,
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': {
            'hidden_channels': OFFICIAL_ARCH['hidden_channels'],
            'num_layers':      OFFICIAL_ARCH['num_layers'],
            'heads':           OFFICIAL_ARCH['heads'],
            'total_params':    total_params,
        },
        'training': {
            'epochs_completed':  len(history.get('epoch', [])) + start_epoch,
            'training_time_min': round(training_time, 1),
            'final_loss':        final_loss,
        },
        'results': {
            'mae':         mae,
            'rmse':        rmse,
            'max_error':   max_err,
            'accuracy_100m': acc100,
        },
        'eval_results_full': evaluation_results,
        'status': 'SUCCESS',
    }

    result_path = save_dir / f'{model_name}_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    try:
        create_final_summary_corrected(
            model=model,
            history=history,
            evaluation_results=evaluation_results,
            config_dict=config.to_dict(),
            total_time=training_time * 60,
            save_dir=str(save_dir),
        )
    except Exception as e:
        logger.warning(f"⚠️ Error while creating final_summary: {e}")

    print('\n' + '=' * 72)
    print(f'🏁 RESULTADO: {model_name.upper()} ({BASELINE_CLASSES[model_name].__name__})')
    print('=' * 72)
    print(f'   ⏱️  Tempo       : {training_time:.1f} min ({training_time / 60:.1f}h)')
    print(f"   📐 Parameters  : {total_params:,} ({total_params / 1e6:.1f}M)")
    if final_loss is not None:
        print(f'   📉 Loss final  : {final_loss:.4f}')
    if mae is not None:
        print(f'   📏 MAE         : {mae:.2f}m')
    if rmse is not None:
        print(f'   📏 RMSE        : {rmse:.2f}m')
    if acc100 is not None:
        print(f'   🎯 Acc@100m    : {acc100 * 100:.1f}%')
    print(f'   💾 Resultado   : {result_path}')

    mae_str  = f'{mae:.2f}' if mae is not None else '?'
    acc_str  = f'{acc100 * 100:.1f}' if acc100 is not None else '?'
    print(
        f'\n📋 COMPARISON: | {model_name:<8} | {total_params / 1e6:.1f}M | {mae_str}m | {acc_str}% | {time.strftime("%Y-%m-%d")} |'
    )

    del model
    cleanup_memory(f'após {model_name}')

    return result


def print_comparison_summary(all_results: dict, arpia_baseline_mae: float = 7.547):
    print('\n' + '=' * 72)
    print('📊 ARCHITECTURE COMPARISON — FINAL SUMMARY')
    print('=' * 72)
    print(f'  {"Modelo":<12} {"Classe":<18} {"Params":>10} {"MAE":>8} {"Acc@100m":>10}')
    print(f'  {"-"*12} {"-"*18} {"-"*10} {"-"*8} {"-"*10}')

    print(f'  {"ARPIA v4":<12} {"MultiScaleGNN":<18} {"97.6M":>10} '
          f'{arpia_baseline_mae:>7.3f}m {"95.7%":>10}  ← baseline')

    for name, result in all_results.items():
        if result.get('status') not in ('SUCCESS', 'ALREADY_COMPLETE'):
            print(f'  {name:<12} {"ERRO":<18}')
            continue
        params = result.get('architecture', {}).get('total_params', 0)
        mae    = result.get('results', {}).get('mae')
        acc100 = result.get('results', {}).get('accuracy_100m')
        cls    = result.get('model_class', '?')
        params_str = f'{params / 1e6:.1f}M' if params else '?'
        mae_str    = f'{mae:.3f}m' if mae is not None else '?'
        acc_str    = f'{acc100 * 100:.1f}%' if acc100 is not None else '?'
        print(f'  {name:<12} {cls:<18} {params_str:>10} {mae_str:>8} {acc_str:>10}')

    print('=' * 72)


try:
    from colab_complete_fix import apply_complete_fix
    apply_complete_fix()
    print('✅ colab_complete_fix aplicado')
except ImportError:
    pass

output_root = (
    Path(f"{INTERNAL_DATA_ROOT}/RESULTADOS")
    / 'EXPERIMENTOS_UNIFICADOS_v2'
    / 'comparacao_arquiteturas'
)
output_root.mkdir(parents=True, exist_ok=True)
print(f'📁 Output: {output_root}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n🖥️  Dispositivo: {device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'   GPU: {props.name} — {props.total_memory / 1e9:.1f} GB VRAM')

if not HAN_AVAILABLE and 'han' in EXPERIMENTS_TO_RUN:
    EXPERIMENTS_TO_RUN.remove('han')
    print("⚠️  'han' removed from the list (HANConv unavailable)")

print(f'\n📋 Experimentos: {EXPERIMENTS_TO_RUN}')
print(f'   Epochs: {TOTAL_EPOCHS} | Batch: 128 | Hidden: {OFFICIAL_ARCH["hidden_channels"]}')
sys.stdout.flush()

graph_data, scalers = load_base_data(Path(PIPELINE_DIR), device)

all_results = {}
for model_name in EXPERIMENTS_TO_RUN:
    if model_name not in BASELINE_CLASSES:
        print(f'⚠️  Modelo desconhecido: {model_name} — ignorado')
        continue
    try:
        result = run_experiment(model_name, graph_data, scalers, output_root, device)
        all_results[model_name] = result
    except Exception as e:
        logger.error(f"❌ Failure in {model_name}: {e}\n{traceback.format_exc()}")
        all_results[model_name] = {'experiment': model_name, 'status': 'ERROR',
                                   'error': str(e)}

consolidated_path = output_root / 'comparison_results.json'
with open(consolidated_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\n💾 Resultados consolidados: {consolidated_path}')

print_comparison_summary(all_results)
