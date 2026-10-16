"""Figures of the computational cost comparison from the JSON written by the cost scripts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent
RESULTS_FILE = OUTPUT_DIR / 'computational_cost_results_local_v2.json'

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

R = data['results']
T = data['training_results']

MODELS = ['ARPIA', 'GATv2', 'HAN', 'RGCN']
COLORS = {'ARPIA': '#1f77b4', 'GATv2': '#ff7f0e', 'HAN': '#2ca02c', 'RGCN': '#d62728'}
MARKERS = {'ARPIA': 'o', 'GATv2': 's', 'HAN': '^', 'RGCN': 'D'}

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

gpu_name = data['metadata']['gpu']

fig, ax = plt.subplots(figsize=(14, 9))
for name in MODELS:
    params = R[name]['params_M']
    mae = T[name]['mae']
    ax.scatter(params, mae, s=350, color=COLORS[name], marker=MARKERS[name],
               alpha=0.9, edgecolors='black', linewidth=1.5, label=name, zorder=10)

    offsets = {'ARPIA': (0, 22), 'GATv2': (12, -30), 'HAN': (-12, 22), 'RGCN': (12, 22)}
    vas = {'ARPIA': 'bottom', 'GATv2': 'top', 'HAN': 'bottom', 'RGCN': 'bottom'}

    ax.annotate(
        f"{name}\n({mae:.2f}m | {params:.1f}M)",
        (params, mae), xytext=offsets[name], textcoords='offset points',
        fontsize=11, fontweight='bold', ha='center', va=vas[name],
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=COLORS[name], alpha=0.9, lw=1.5)
    )

ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
ax.set_ylabel('Mean Absolute Error (meters)', fontweight='bold')
ax.set_title(f'Performance vs. Model Complexity (MAE)\n{gpu_name}', fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True,
          framealpha=0.95, shadow=True, borderpad=1.2, labelspacing=1.5)
ax.set_xlim(0, 120)
ax.set_ylim(0, 200)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'performance_vs_parameters_mean_error.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: performance_vs_parameters_mean_error.png")

fig, ax = plt.subplots(figsize=(14, 9))
for name in MODELS:
    params = R[name]['params_M']
    rmse = T[name]['rmse']
    ax.scatter(params, rmse, s=350, color=COLORS[name], marker=MARKERS[name],
               alpha=0.9, edgecolors='black', linewidth=1.5, label=name, zorder=10)

    offsets = {'ARPIA': (0, 22), 'GATv2': (12, -30), 'HAN': (-12, 22), 'RGCN': (12, 22)}
    vas = {'ARPIA': 'bottom', 'GATv2': 'top', 'HAN': 'bottom', 'RGCN': 'bottom'}

    ax.annotate(
        f"{name}\n({rmse:.2f}m | {params:.1f}M)",
        (params, rmse), xytext=offsets[name], textcoords='offset points',
        fontsize=11, fontweight='bold', ha='center', va=vas[name],
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=COLORS[name], alpha=0.9, lw=1.5)
    )

ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
ax.set_ylabel('Root Mean Square Error (meters)', fontweight='bold')
ax.set_title(f'Performance vs. Model Complexity (RMSE)\n{gpu_name}', fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True,
          framealpha=0.95, shadow=True, borderpad=1.2, labelspacing=1.5)
ax.set_xlim(0, 120)
ax.set_ylim(0, 220)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'performance_vs_parameters_rmse.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: performance_vs_parameters_rmse.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
x = np.arange(len(MODELS))
bar_colors = [COLORS[n] for n in MODELS]

mae_per_M = [R[n]['mae_per_M'] for n in MODELS]
axes[0].bar(x, mae_per_M, color=bar_colors, edgecolor='white', width=0.6)
axes[0].set_xticks(x)
axes[0].set_xticklabels(MODELS)
axes[0].set_ylabel('MAE / Million Params (m/M)')
axes[0].set_title('Parameter Efficiency\n(lower is better)')
for i, v in enumerate(mae_per_M):
    axes[0].text(i, v + max(mae_per_M) * 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

inf_times = [R[n]['inference']['mean_ms'] / 1000 for n in MODELS]
axes[1].bar(x, inf_times, color=bar_colors, edgecolor='white', width=0.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels(MODELS)
axes[1].set_ylabel('Inference Time (seconds)')
axes[1].set_title(f'Inference Speed ({gpu_name})\n(lower is better)')
for i, v in enumerate(inf_times):
    axes[1].text(i, v + max(inf_times) * 0.02, f'{v:.1f}s', ha='center', fontsize=11, fontweight='bold')

train_times = [T[n]['training_time_min'] for n in MODELS]
axes[2].bar(x, train_times, color=bar_colors, edgecolor='white', width=0.6)
axes[2].set_xticks(x)
axes[2].set_xticklabels(MODELS)
axes[2].set_ylabel('Training Time (min)')
axes[2].set_title('Training Cost (L4 Colab)\n(lower is better)')
for i, v in enumerate(train_times):
    axes[2].text(i, v + max(train_times) * 0.02, f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')

fig.suptitle('Computational Efficiency Comparison', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'computational_efficiency_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: computational_efficiency_comparison.png")

fig, ax = plt.subplots(figsize=(14, 9))
for name in MODELS:
    params = R[name]['params_M']
    mae = T[name]['mae']
    inf_s = R[name]['inference']['mean_ms'] / 1000

    bubble_size = max(inf_s * 80, 150)
    ax.scatter(params, mae, s=bubble_size, color=COLORS[name], marker='o',
               alpha=0.5, edgecolors=COLORS[name], linewidth=2.5,
               label=f"{name} ({inf_s:.1f}s)", zorder=10)
    ax.annotate(
        f"{name}\n{mae:.1f}m | {params:.1f}M\n{inf_s:.1f}s inference",
        (params, mae),
        fontsize=10, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS[name], alpha=0.85)
    )

ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
ax.set_ylabel('Mean Absolute Error (meters)', fontweight='bold')
ax.set_title(f'Performance-Complexity-Speed Trade-off\n(bubble size = inference time, {gpu_name})',
             fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.95)
ax.set_xlim(0, 120)
ax.set_ylim(0, 200)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'performance_complexity_speed_bubble.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: performance_complexity_speed_bubble.png")

fig, ax = plt.subplots(figsize=(16, 5))
ax.axis('off')

col_labels = ['Architecture', 'Params (M)', 'MAE (m)', 'RMSE (m)', 'Inference (s)',
              'Peak Mem (GB)', 'Train Time (min)', 'MAE/M Params']
cell_data = []
for name in MODELS:
    cell_data.append([
        name,
        f"{R[name]['params_M']:.1f}",
        f"{T[name]['mae']:.2f}",
        f"{T[name]['rmse']:.2f}",
        f"{R[name]['inference']['mean_ms'] / 1000:.1f}",
        f"{R[name]['inference']['peak_memory_mb'] / 1024:.1f}",
        f"{T[name]['training_time_min']:.0f}",
        f"{R[name]['mae_per_M']:.4f}",
    ])

table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 2.0)

for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(len(MODELS)):
    row_color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[(i + 1, j)].set_facecolor(row_color)
    table[(i + 1, 0)].set_text_props(fontweight='bold', color=COLORS[MODELS[i]])

ax.set_title(f'Computational Cost Summary — {gpu_name}\n'
             f'PyTorch {data["metadata"]["pytorch"]} | Graph: {data["metadata"]["node_counts"]["PEA"]} PEAs, '
             f'{sum(data["metadata"]["node_counts"].values())} total nodes',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'computational_cost_summary_table.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: computational_cost_summary_table.png")

print("\nAll figures generated!")
