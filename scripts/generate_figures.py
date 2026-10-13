import os, sys, json, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/earc")
FIG_DIR = PROJECT_ROOT / "runs/rerun_default/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# -------------------------------------------------------------
# Figure 1: Fresh vs Stored Strict F1 across all 12 runs
# -------------------------------------------------------------
print("Generating Fig 1: Fresh vs Stored Strict F1...")
df_comp = pd.read_csv("runs/rerun_default/tables/fresh_vs_stored_summary.csv")

labels = [f"{r['dataset'].upper()}\n{r['generator']}\n({r['method'].split('_')[0]})" for _, r in df_comp.iterrows()]
stored_f1 = df_comp['stored_eval_f1'].tolist()
fresh_f1 = df_comp['fresh_eval_f1'].tolist()

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width/2, stored_f1, width, label='Stored Run (Lab PC)', color='#4C72B0', alpha=0.9)
rects2 = ax.bar(x + width/2, fresh_f1, width, label='Fresh Rerun (Windows)', color='#55A868', alpha=0.9)

ax.set_ylabel('Strict F1 Score (%)')
ax.set_title('Task 3: Stored vs Fresh Evaluation Results Across All 12 Runs')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add values on top of bars
for r in rects1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8)
for r in rects2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_fresh_vs_stored_f1.png", dpi=300)
plt.savefig(FIG_DIR / "fig1_fresh_vs_stored_f1.pdf")
plt.close()

# -------------------------------------------------------------
# Figure 2: Component Ablations (A1 - A5) Strict F1
# -------------------------------------------------------------
print("Generating Fig 2: Component Ablations...")
df_exp1 = pd.read_csv("runs/rerun_default/tables/exp1_summary_strict.csv")

# Filter configurations A1 through A5
abl_order = ['A1', 'A2', 'A3', 'A4', 'A5']
df_abl = df_exp1[df_exp1['config_id'].isin(abl_order)].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
datasets = ['nq', 'hotpotqa', 'triviaqa']
titles = ['Natural Questions (NQ)', 'HotpotQA', 'TriviaQA']

for idx, ds in enumerate(datasets):
    ax = axes[idx]
    sub_llama = df_abl[(df_abl['dataset'] == ds) & (df_abl['generator'].isin(['llama3.2_3b', 'llama3.2:3b']))].set_index('config_id')
    sub_mist = df_abl[(df_abl['dataset'] == ds) & (df_abl['generator'].isin(['ministral-8b', 'ministral_8b']))].set_index('config_id')
    
    y_llama = [sub_llama.loc[c]['f1'] if c in sub_llama.index else 0 for c in abl_order]
    y_mist = [sub_mist.loc[c]['f1'] if c in sub_mist.index else 0 for c in abl_order]
    
    pos = np.arange(len(abl_order))
    ax.bar(pos - width/2, y_llama, width, label='Llama 3.2 3B', color='#4C72B0')
    ax.bar(pos + width/2, y_mist, width, label='Ministral 8B', color='#C44E52')
    
    ax.set_title(titles[idx])
    ax.set_xticks(pos)
    ax.set_xticklabels(['A1\nSimOnly', 'A2\nEvidOnly', 'A3\nNoFilt', 'A4\nNoEvid', 'A5\nFullEARC'], fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if idx == 0:
        ax.set_ylabel('Strict Token F1 (%)')
        ax.legend(frameon=True)

fig.suptitle('Experiment 1: Architectural Component Ablations (Strict SQuAD F1)', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_component_ablations.png", dpi=300)
plt.savefig(FIG_DIR / "fig2_component_ablations.pdf")
plt.close()

# -------------------------------------------------------------
# Figure 3: Token Budget Sensitivity Curve
# -------------------------------------------------------------
print("Generating Fig 3: Token Budget Sensitivity Curve...")
df_exp2 = pd.read_csv("runs/rerun_default/tables/exp2_summary_strict.csv")

fig, ax = plt.subplots(figsize=(9, 5.5))
budgets = [150, 300, 450, 600]

for gen_match, gen_label, col, marker in [(['llama3.2_3b', 'llama3.2:3b'], 'Llama 3.2 3B', '#4C72B0', 'o'), (['ministral-8b', 'ministral_8b'], 'Ministral 8B', '#C44E52', 's')]:
    sub = df_exp2[(df_exp2['dataset'] == 'hotpotqa') & (df_exp2['generator'].isin(gen_match))]
    f1s = []
    for b in budgets:
        row = sub[sub['config_id'] == f'earc_b{b}']
        f1s.append(row['f1'].values[0] if len(row) > 0 else np.nan)
    ax.plot(budgets, f1s, marker=marker, linewidth=2.5, markersize=8, color=col, label=f'EARC ({gen_label})')

# Top-k k=5 reference lines
topk_llama = df_exp2[(df_exp2['dataset'] == 'hotpotqa') & (df_exp2['generator'].isin(['llama3.2_3b', 'llama3.2:3b'])) & (df_exp2['config_id'].isin(['topk_5', 'topk_k5']))]['f1'].values[0]
topk_mist = df_exp2[(df_exp2['dataset'] == 'hotpotqa') & (df_exp2['generator'].isin(['ministral-8b', 'ministral_8b'])) & (df_exp2['config_id'].isin(['topk_5', 'topk_k5']))]['f1'].values[0]

ax.axhline(topk_llama, color='#4C72B0', linestyle='--', alpha=0.7, label=f'Top-k k=5 (Llama 3.2 3B, 598 tok)')
ax.axhline(topk_mist, color='#C44E52', linestyle='--', alpha=0.7, label=f'Top-k k=5 (Ministral 8B, 598 tok)')

ax.set_xlabel('Token Budget (T)')
ax.set_ylabel('Strict Token F1 (%)')
ax.set_title('Experiment 2: HotpotQA Token Budget Scaling Curve (b150 - b600)')
ax.set_xticks(budgets)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, loc='lower right')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_token_budget_curve.png", dpi=300)
plt.savefig(FIG_DIR / "fig3_token_budget_curve.pdf")
plt.close()

# -------------------------------------------------------------
# Figure 4: Error Attribution Breakdown
# -------------------------------------------------------------
print("Generating Fig 4: Error Attribution Breakdown...")
df_ret = pd.read_csv("runs/rerun_default/tables/retention_attribution_strict.csv")

# Select Standard RAG vs EARC default for all 3 datasets on Llama
runs_to_show = [
    ('NQ Standard RAG', 'nq', 'pred_baseline_nq_standard_rag_llama3.2_3b.strict.jsonl'),
    ('NQ EARC Default', 'nq', 'pred_baseline_nq_earc_llama3.2_3b.strict.jsonl'),
    ('HotpotQA Std RAG', 'hotpotqa', 'pred_baseline_hotpotqa_standard_rag_llama3.2_3b.strict.jsonl'),
    ('HotpotQA EARC Def', 'hotpotqa', 'pred_baseline_hotpotqa_earc_llama3.2_3b.strict.jsonl'),
    ('TriviaQA Std RAG', 'triviaqa', 'pred_baseline_triviaqa_standard_rag_llama3.2_3b.strict.jsonl'),
    ('TriviaQA EARC Def', 'triviaqa', 'pred_baseline_triviaqa_earc_llama3.2_3b.strict.jsonl'),
]

labels_attr = []
corrects, misses, losses, errors = [], [], [], []

for lbl, ds, fname in runs_to_show:
    row = df_ret[df_ret['file'] == fname].iloc[0]
    labels_attr.append(lbl)
    corrects.append(row['correct'])
    misses.append(row['retrieval_miss'])
    losses.append(row['compression_loss'])
    errors.append(row['generation_error'])

fig, ax = plt.subplots(figsize=(11, 6))
ind = np.arange(len(labels_attr))
w = 0.55

p1 = ax.bar(ind, corrects, w, label='Correct Answer', color='#2ca02c')
p2 = ax.bar(ind, errors, w, bottom=corrects, label='Generation Error', color='#ff7f0e')
p3 = ax.bar(ind, losses, w, bottom=np.array(corrects) + np.array(errors), label='Compression Loss', color='#d62728')
p4 = ax.bar(ind, misses, w, bottom=np.array(corrects) + np.array(errors) + np.array(losses), label='Retrieval Miss', color='#7f7f7f')

ax.set_ylabel('Percentage of Evaluation Set (%)')
ax.set_title('Task 4: Evidence Retention and Error Attribution Breakdown')
ax.set_xticks(ind)
ax.set_xticklabels(labels_attr, rotation=30, ha='right')
ax.set_ylim(0, 105)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_error_attribution.png", dpi=300)
plt.savefig(FIG_DIR / "fig4_error_attribution.pdf")
plt.close()

# -------------------------------------------------------------
# Figure 5: Context Identity Between Stored and Fresh Runs
# -------------------------------------------------------------
print("Generating Fig 5: Context Identity...")
fig, ax = plt.subplots(figsize=(10, 5))
sub_identity = df_comp[['dataset', 'method', 'generator', 'identical_context_pct']].copy()
x_labels = [f"{r['dataset'].upper()} {r['method']}" for _, r in sub_identity.iterrows()][:6]
# Both generators have identical contexts
identities = sub_identity['identical_context_pct'].tolist()[:6]

bars = ax.bar(x_labels, identities, color='#1f77b4', width=0.5, alpha=0.85)
ax.set_ylabel('Identical Context (%)')
ax.set_ylim(90, 101)
ax.set_title('Task 3: Exact Byte-for-Byte Context Match Between Stored & Fresh Runs')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for b in bars:
    h = b.get_height()
    ax.annotate(f'{h:.2f}%', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_context_identity.png", dpi=300)
plt.savefig(FIG_DIR / "fig5_context_identity.pdf")
plt.close()

print("All 5 figures generated successfully in PNG and PDF formats!")
