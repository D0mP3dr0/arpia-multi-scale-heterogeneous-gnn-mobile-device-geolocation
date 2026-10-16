# -*- coding: utf-8 -*-
"""Sensitivity analysis of the force-system parameters, split into groups that run in parallel."""

INTERNAL_DATA_ROOT = __import__("os").environ.get("ARPIA_ROOT", ".")
import os
import sys
import subprocess
import time
import gc
import json
import csv
from pathlib import Path
from datetime import datetime

_GROUP_DEFAULT = 0
GROUP = int(os.environ.get('SENSITIVITY_GROUP', _GROUP_DEFAULT))

print("=" * 72)
print(f"🔬 ARPIA — SENSITIVITY ANALYSIS PARALELA  |  GRUPO {GROUP}")
print("=" * 72)
print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Goal: vary the parameters of the force system — group {GROUP}/3")
print("=" * 72)

if GROUP not in (0, 1, 2, 3):
    raise ValueError(f"GROUP must be 0, 1, 2 or 3 — received: {GROUP}")

import types as _types
import builtins as _builtins

_RUNNING_AS_SUBPROCESS = False
try:
    _ip_check = get_ipython()
    if _ip_check is None:
        _RUNNING_AS_SUBPROCESS = True
except NameError:
    _RUNNING_AS_SUBPROCESS = True

if _RUNNING_AS_SUBPROCESS:
    _mock_kernel = _types.SimpleNamespace(
        parent    = _types.SimpleNamespace(config={}),
        io_loop   = None,
        comm_info = {},
    )
    _mock_ip = _types.SimpleNamespace(
        kernel            = _mock_kernel,
        run_line_magic    = lambda *a, **kw: None,
        run_cell_magic    = lambda *a, **kw: None,
        events            = _types.SimpleNamespace(register=lambda *a, **kw: None),
        user_ns           = {},
        display_pub       = _types.SimpleNamespace(publish=lambda *a, **kw: None),
    )
    _builtins.get_ipython = lambda: _mock_ip
    print(f"🔧 Subprocess detectado (GROUP={GROUP}) — get_ipython() mockado")

try:
    import google.colab
    IN_COLAB = True
    print("✅ Google Colab detectado")
except ImportError:
    IN_COLAB = False
    print("🏠 Ambiente local detectado")

if IN_COLAB and not _RUNNING_AS_SUBPROCESS:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive montado")
elif IN_COLAB and _RUNNING_AS_SUBPROCESS:
    print('✅ Google Drive accessible (subprocess — no re-mount)')

PIPELINE_DIR = Path(f"{INTERNAL_DATA_ROOT}/TREINAMENTO_REAL") if IN_COLAB \
               else Path(__file__).parent.parent.parent / "GNN/TREINAMENTO_REAL-20250717T215639Z-1-001/TREINAMENTO_REAL"

OUTPUT_DIR = Path(f"{INTERNAL_DATA_ROOT}/sensitivity_analysis") if IN_COLAB \
             else PIPELINE_DIR.parent.parent / "internal/sensitivity_analysis/results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PIPELINE_DIR.exists():
    raise FileNotFoundError(f"❌ Pipeline dir not found: {PIPELINE_DIR}")

sys.path.insert(0, str(PIPELINE_DIR))
print(f"✅ Pipeline dir: {PIPELINE_DIR}")
print(f"✅ Output dir:   {OUTPUT_DIR}")

for fname in ["arpia_hetero_graph_v3.pt", "graph_metadata_v3.json", "scalers.pkl",
              "losses.py", "trainers.py", "models.py", "train_colab.py"]:
    fp = PIPELINE_DIR / fname
    if fp.exists():
        print(f"✅ {fname} ({fp.stat().st_size / (1024**2):.1f} MB)")
    else:
        raise FileNotFoundError(f"❌ {fname} not found in {PIPELINE_DIR}")

import torch.optim.lr_scheduler as _lr_sched
_OriginalPlateau = _lr_sched.ReduceLROnPlateau
class _PatchedPlateau(_OriginalPlateau):
    def __init__(self, *args, **kwargs):
        kwargs.pop('verbose', None)
        super().__init__(*args, **kwargs)
_lr_sched.ReduceLROnPlateau = _PatchedPlateau
print("🔧 ReduceLROnPlateau patched (PyTorch 2.x)")

print("\n📦 Importando módulos...")
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config_colab
import utils
import data_loaders
import models
import losses
import trainers
import train_colab

trainers.ReduceLROnPlateau = _PatchedPlateau

from config_colab import ColabConfig, create_custom_config, PhaseConfig
from losses import PhysicsInformedLoss, create_physics_informed_loss
from train_colab import (
    create_pipeline_ultra_avancado,
    execute_training_ultra_avancado,
    run_comprehensive_evaluation,
)
print('✅ All modules were imported')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"🖥️  GPU: {gpu_name} — {gpu_mem:.1f} GB VRAM")
else:
    print('⚠️  No GPU — this will be very slow!')

print("\n📊 Carregando dados...")
graph_data = torch.load(PIPELINE_DIR / "arpia_hetero_graph_v3.pt",
                        map_location=device, weights_only=False)
with open(PIPELINE_DIR / "graph_metadata_v3.json") as f:
    metadata = json.load(f)
with open(PIPELINE_DIR / "scalers.pkl", 'rb') as f:
    scalers = pickle.load(f)
print(f"✅ Dados carregados: {graph_data['PEA'].num_nodes:,} PEAs")

BASELINE = {
    'physics_weight':        2.0,
    'connection_weight':     3.0,
    'non_connection_weight': 5.0,
    'position_weight':       10.0,
    'precision_weight':      2.0,
    'adaptive_threshold':    300.0,
}

SENSITIVITY_EPOCHS  = 100
SENSITIVITY_BATCH   = 128
SENSITIVITY_HIDDEN  = 512
SENSITIVITY_LAYERS  = 4
SENSITIVITY_HEADS   = 8

ALL_EXPERIMENTS = []

ALL_EXPERIMENTS.append({
    'name': 'baseline',
    'description': 'Original configuration of the article',
    'loss_overrides': {},
    'threshold_override': None,
})

for pw in [0.1, 0.5, 1.0, 3.0, 5.0]:
    ALL_EXPERIMENTS.append({
        'name': f'physics_w_{pw}',
        'description': f'physics_weight = {pw}',
        'loss_overrides': {'physics_weight': pw},
        'threshold_override': None,
    })

for ncw in [1.0, 2.0, 3.0, 7.0, 10.0]:
    ALL_EXPERIMENTS.append({
        'name': f'non_conn_w_{ncw}',
        'description': f'non_connection_weight = {ncw}',
        'loss_overrides': {'non_connection_weight': ncw},
        'threshold_override': None,
    })

for cw in [1.0, 2.0, 5.0, 7.0]:
    ALL_EXPERIMENTS.append({
        'name': f'conn_w_{cw}',
        'description': f'connection_weight = {cw}',
        'loss_overrides': {'connection_weight': cw},
        'threshold_override': None,
    })

for thr in [200, 400, 500]:
    ALL_EXPERIMENTS.append({
        'name': f'threshold_{thr}m',
        'description': f'adaptive_threshold = {thr}m',
        'loss_overrides': {},
        'threshold_override': thr,
    })

ALL_EXPERIMENTS.append({
    'name': 'fixed_threshold',
    'description': 'adaptive_thresholds = False',
    'loss_overrides': {'adaptive_thresholds': False},
    'threshold_override': None,
})

assert len(ALL_EXPERIMENTS) == 19, f"Esperado 19 experimentos, obteve {len(ALL_EXPERIMENTS)}"

GROUPS = {
    0: ALL_EXPERIMENTS[0:5],
    1: ALL_EXPERIMENTS[5:10],
    2: ALL_EXPERIMENTS[10:15],
    3: ALL_EXPERIMENTS[15:19],
}

experiments = GROUPS[GROUP]
print(f"\n📋 GRUPO {GROUP} — {len(experiments)} experimentos:")
for i, exp in enumerate(experiments):
    print(f"   {i+1}. {exp['name']}: {exp['description']}")


def run_sensitivity_experiment(exp_config, graph_data, scalers, device,
                                output_dir, exp_index, total_experiments):
    exp_name = exp_config['name']
    exp_dir  = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"🔬 EXPERIMENTO {exp_index}/{total_experiments}: {exp_name}")
    print(f"📝 {exp_config['description']}")
    print(f"{'='*72}")

    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        mem_free = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated(0))
        print(f"💾 GPU livre: {mem_free / (1024**3):.1f} GB")

    start_time = time.time()
    original_non_connection_penalty = None

    try:
        config = ColabConfig()
        config.data.batch_size      = SENSITIVITY_BATCH
        config.model.hidden_channels = SENSITIVITY_HIDDEN
        config.model.num_layers     = SENSITIVITY_LAYERS
        config.model.heads          = SENSITIVITY_HEADS
        config.total_epochs         = SENSITIVITY_EPOCHS
        config.mixed_precision      = True
        config.eda.frequency        = 999
        config.data.num_workers       = 2
        config.data.pin_memory        = True
        config.data.persistent_workers = False

        phase1 = max(1, int(SENSITIVITY_EPOCHS * 0.50))
        phase2 = max(1, int(SENSITIVITY_EPOCHS * 0.30))
        phase3 = SENSITIVITY_EPOCHS - phase1 - phase2
        config.phases['phase_1'].epochs = phase1
        config.phases['phase_2'].epochs = phase2
        config.phases['phase_3'].epochs = phase3
        print(f"   📊 Phases: {phase1}+{phase2}+{phase3} = {SENSITIVITY_EPOCHS} epochs")
        config.validate_config()

        if exp_config.get('threshold_override'):
            new_threshold = exp_config['threshold_override']
            print(f"🔧 Monkey-patching threshold para {new_threshold}m...")
            original_non_connection_penalty = PhysicsInformedLoss.non_connection_penalty

            def patched_non_connection_penalty(self, pred_locations, batch):
                try:
                    dev = pred_locations.device
                    if not hasattr(batch, 'edge_index_dict'):
                        return self._safe_tensor(0.0, dev)
                    nc_type = ('PEA', 'NAO_CONEXAO', 'ERB')
                    if nc_type not in batch.edge_index_dict:
                        return self._safe_tensor(0.0, dev)
                    edge_index = batch.edge_index_dict[nc_type]
                    if edge_index.size(1) == 0:
                        return self._safe_tensor(0.0, dev)
                    pea_pos = pred_locations
                    erb_pos = batch.x_dict.get('ERB', None)
                    if erb_pos is None:
                        return self._safe_tensor(0.0, dev)
                    erb_pos = erb_pos[:, :2]
                    pi = torch.clamp(edge_index[0], 0, len(pea_pos) - 1)
                    ei = torch.clamp(edge_index[1], 0, len(erb_pos) - 1)
                    pp = pea_pos[pi]
                    ep = erb_pos[ei]
                    if torch.isnan(pp).any() or torch.isnan(ep).any():
                        return self._safe_tensor(0.0, dev)
                    if self.use_haversine:
                        try:
                            dists = self.haversine_distance(pp, ep)
                        except Exception:
                            dists = torch.norm(pp - ep, dim=1) * 450.0
                    else:
                        dists = torch.norm(pp - ep, dim=1)
                        cr = pred_locations.max() - pred_locations.min()
                        if cr < 10:
                            dists = dists * 450.0
                    if torch.isnan(dists).any() or torch.isinf(dists).any():
                        return self._safe_tensor(0.0, dev)
                    thr = float(new_threshold)
                    safe_d = torch.clamp(dists, min=10.0)
                    repulsive = thr ** 2 / (safe_d ** 2)
                    mask = dists < thr
                    if mask.sum() > 0:
                        loss = torch.mean(repulsive[mask])
                        if torch.isnan(loss) or torch.isinf(loss):
                            return self._safe_tensor(0.0, dev)
                        return loss
                    return self._safe_tensor(0.0, dev)
                except Exception:
                    return self._safe_tensor(0.0, pred_locations.device)

            PhysicsInformedLoss.non_connection_penalty = patched_non_connection_penalty

        from models import create_model
        from data_loaders import create_data_loaders
        from trainers import create_trainer

        model = create_model(config.to_dict(), graph_data, use_ensemble=False).to(device)
        loss_fn = create_physics_informed_loss(config.to_dict())
        data_loaders_dict = create_data_loaders(
            graph_data,
            batch_size=config.data.batch_size,
            num_neighbors=config.data.num_neighbors,
            num_workers=config.data.num_workers,
            device=device,
        )
        trainer = create_trainer(model, loss_fn, config.to_dict(), device)
        trainer.set_scalers(scalers)

        trainer.run_automated_eda = lambda *args, **kwargs: None

        if exp_config['loss_overrides']:
            for key, value in exp_config['loss_overrides'].items():
                if hasattr(loss_fn, key):
                    setattr(loss_fn, key, value)
                if hasattr(loss_fn, 'base_weights') and key in loss_fn.base_weights:
                    loss_fn.base_weights[key] = value
                    loss_fn.current_weights[key] = value
                print(f"   ✅ loss override: {key} = {value}")

        print(f"\n🚀 Training for {SENSITIVITY_EPOCHS} epochs...")
        history = execute_training_ultra_avancado(
            trainer=trainer,
            data_loaders=data_loaders_dict,
            config=config,
            save_checkpoints=False,
            save_dir=str(exp_dir),
        )

        del data_loaders_dict
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 DataLoaders liberados.")

        metrics_list = history.get('metrics', []) if isinstance(history, dict) else []
        valid_metrics = [m for m in metrics_list if m and isinstance(m, dict)
                         and m.get('mean_error', float('inf')) > 0]
        if valid_metrics:
            best_m     = min(valid_metrics, key=lambda m: m.get('mean_error', float('inf')))
            mae_val    = best_m.get('mean_error',    -1)
            rmse_val   = best_m.get('rmse_error',    best_m.get('rmse', -1))
            acc_100m   = best_m.get('accuracy_100m', -1)
            acc_50m    = best_m.get('accuracy_50m',  -1)
            median_err = best_m.get('median_error',  -1)
            print(f"📊 Best validation epoch: MAE={mae_val:.1f}m  Acc@100m={acc_100m:.1%}")
        else:
            mae_val    = trainer.best_metrics.get('error', -1)
            rmse_val   = -1
            acc_100m   = -1
            acc_50m    = -1
            median_err = -1
            print(f"📊 Metrics from trainer.best_metrics: MAE={mae_val:.1f}m")

        elapsed = time.time() - start_time
        metrics = {
            'experiment':             exp_name,
            'description':            exp_config['description'],
            'group':                  GROUP,
            'mae_meters':             mae_val,
            'rmse_meters':            rmse_val,
            'accuracy_100m':          acc_100m,
            'accuracy_50m':           acc_50m,
            'median_error':           median_err,
            'training_time_seconds':  elapsed,
            'epochs':                 SENSITIVITY_EPOCHS,
            'final_loss':             history.get('final_loss', -1) if isinstance(history, dict) else -1,
            'status':                 'SUCCESS',
            'physics_weight':         exp_config['loss_overrides'].get('physics_weight',        BASELINE['physics_weight']),
            'connection_weight':      exp_config['loss_overrides'].get('connection_weight',     BASELINE['connection_weight']),
            'non_connection_weight':  exp_config['loss_overrides'].get('non_connection_weight', BASELINE['non_connection_weight']),
            'adaptive_threshold':     exp_config.get('threshold_override', BASELINE['adaptive_threshold']),
        }

        print(f"\n✅ RESULTADO: MAE = {metrics['mae_meters']:.2f}m | "
              f"RMSE = {metrics['rmse_meters']:.2f}m | "
              f"Acc@100m = {metrics['accuracy_100m']:.1%}")
        print(f"⏱️  Tempo: {elapsed / 60:.1f} min")

        result_json = exp_dir / 'result.json'
        with open(result_json, 'w') as _f:
            json.dump(metrics, _f, indent=2, default=str)
        print(f"💾 result.json saved to {result_json}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR in {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        metrics = {
            'experiment':             exp_name,
            'description':            exp_config['description'],
            'group':                  GROUP,
            'mae_meters':             -1,
            'rmse_meters':            -1,
            'accuracy_100m':          -1,
            'accuracy_50m':           -1,
            'median_error':           -1,
            'training_time_seconds':  elapsed,
            'epochs':                 SENSITIVITY_EPOCHS,
            'final_loss':             -1,
            'status':                 f'ERROR: {str(e)[:120]}',
            'physics_weight':         exp_config['loss_overrides'].get('physics_weight',        BASELINE['physics_weight']),
            'connection_weight':      exp_config['loss_overrides'].get('connection_weight',     BASELINE['connection_weight']),
            'non_connection_weight':  exp_config['loss_overrides'].get('non_connection_weight', BASELINE['non_connection_weight']),
            'adaptive_threshold':     exp_config.get('threshold_override', BASELINE['adaptive_threshold']),
        }

    finally:
        if original_non_connection_penalty is not None:
            PhysicsInformedLoss.non_connection_penalty = original_non_connection_penalty
            print("🔧 Threshold restaurado")
        _locals = locals()
        for var in ('model', 'trainer', 'loss_fn', 'data_loaders_dict',
                    'train_loader', 'val_loader', 'test_loader'):
            if var in _locals:
                try:
                    del _locals[var]
                except Exception:
                    pass
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 Recursos do experimento liberados.")

    return metrics


print(f"\n{'🔬' * 36}")
print(f"INICIANDO GRUPO {GROUP} — {len(experiments)} experimentos")
print(f"Epochs per experiment: {SENSITIVITY_EPOCHS}")
print(f"Tempo estimado: ~{len(experiments) * 7}–{len(experiments) * 10}h")
print(f"{'🔬' * 36}\n")

all_results  = []
results_file = OUTPUT_DIR / f'sensitivity_results_group{GROUP}.csv'
log_file     = OUTPUT_DIR / f'sensitivity_log_group{GROUP}.txt'

CSV_COLUMNS = [
    'experiment', 'description', 'group',
    'mae_meters', 'rmse_meters', 'accuracy_100m', 'accuracy_50m', 'median_error',
    'training_time_seconds', 'epochs', 'final_loss', 'status',
    'physics_weight', 'connection_weight', 'non_connection_weight', 'adaptive_threshold',
]

if not results_file.exists():
    with open(results_file, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

completed_names = set()
for exp in experiments:
    rj = OUTPUT_DIR / exp['name'] / 'result.json'
    if rj.exists():
        try:
            prev = json.loads(rj.read_text())
            if prev.get('status') == 'SUCCESS':
                completed_names.add(exp['name'])
        except Exception:
            pass

if completed_names:
    print(f"♻️  RESUME: {len(completed_names)} already complete — they will be skipped:")
    for n in sorted(completed_names):
        print(f"   ✅ {n}")
    print(f"   Restam: {len(experiments) - len(completed_names)} experimentos\n")

total_start = time.time()

for i, exp in enumerate(experiments):
    exp_name = exp['name']

    if exp_name in completed_names:
        rj = OUTPUT_DIR / exp_name / 'result.json'
        metrics = json.loads(rj.read_text())
        all_results.append(metrics)
        print(f"⏭️  [{i+1}/{len(experiments)}] {exp_name} — already finished, skipping.")
        continue

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n[{datetime.now()}] {i+1}/{len(experiments)}: {exp_name}\n")

    metrics = run_sensitivity_experiment(
        exp_config=exp,
        graph_data=graph_data,
        scalers=scalers,
        device=device,
        output_dir=OUTPUT_DIR,
        exp_index=i + 1,
        total_experiments=len(experiments),
    )

    all_results.append(metrics)

    with open(results_file, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(metrics)

    elapsed_total = time.time() - total_start
    remaining     = (elapsed_total / (i + 1)) * (len(experiments) - i - 1)
    print(f"\n📊 PROGRESSO GRUPO {GROUP}: {i+1}/{len(experiments)} completos")
    print(f"⏱️  Decorrido: {elapsed_total/60:.1f} min | Restante est.: {remaining/60:.1f} min")

    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now()}] MAE={metrics['mae_meters']:.2f}m | "
                f"Status={metrics['status']}\n"
                f"Progresso: {i+1}/{len(experiments)} | "
                f"Decorrido: {elapsed_total/60:.1f}min | Restante: {remaining/60:.1f}min\n")

total_time = time.time() - total_start

print(f"\n{'🎉' * 36}")
print(f"GRUPO {GROUP} CONCLUÍDO!")
print(f"Tempo total: {total_time / 60:.1f} min ({total_time / 3600:.1f} h)")
print(f"CSV:         {results_file}")
print(f"{'🎉' * 36}")

ok = [r for r in all_results if r.get('status') == 'SUCCESS']
print(f"\n{'='*72}")
print(f"📋 SUMMARY OF GROUP {GROUP} — {len(ok)}/{len(all_results)} successful")
print(f"{'='*72}")
print(f"{'Experimento':<30} {'MAE (m)':>10} {'Acc@100m':>10} {'Tempo (min)':>12}")
print(f"{'-'*64}")
for r in all_results:
    mae  = f"{r['mae_meters']:.2f}"  if r['mae_meters']    > 0 else "ERRO"
    acc  = f"{r['accuracy_100m']:.1%}" if r['accuracy_100m'] > 0 else "ERRO"
    mins = f"{r['training_time_seconds']/60:.1f}"
    print(f"{r['experiment']:<30} {mae:>10} {acc:>10} {mins:>12}")

print(f"\n📅 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('🔔 When the 4 groups finish, run sensitivity_consolidate.py')
print('   to merge the CSVs and generate the plots/LaTeX table.')
