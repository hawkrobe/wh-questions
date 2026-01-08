"""
Generate publication-ready figures for CogSci paper.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
FIND_COLOR = '#0077BB'  # Blue
AVOID_COLOR = '#EE7733'  # Orange

# =============================================================================
# Experiment 1: Goal × Base Rate (N=5, singleton)
# =============================================================================

exp1_data = {
    0.2: {'find': 0.368, 'avoid': 0.073},  # P(ask "Which uncontaminated?")
    0.5: {'find': 0.808, 'avoid': 0.192},
    0.8: {'find': 0.927, 'avoid': 0.632},
}

fig1, ax1 = plt.subplots(figsize=(5, 4))

base_rates = [0.2, 0.5, 0.8]
find_probs = [exp1_data[br]['find'] for br in base_rates]
avoid_probs = [exp1_data[br]['avoid'] for br in base_rates]

ax1.plot(base_rates, find_probs, 'o-', color=FIND_COLOR, linewidth=2,
         markersize=10, label='Find uncontaminated', markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(base_rates, avoid_probs, 's-', color=AVOID_COLOR, linewidth=2,
         markersize=10, label='Avoid contamination', markeredgecolor='white', markeredgewidth=1.5)

ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax1.set_xlabel('Contamination base rate')
ax1.set_ylabel('P(Ask "Which are uncontaminated?")')
ax1.set_xticks([0.2, 0.5, 0.8])
ax1.set_xticklabels(['20%', '50%', '80%'])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.legend(loc='center left', frameon=False)
ax1.set_title('Exp 1: Goal × Base Rate\n(Singleton selection, N=5)')

plt.tight_layout()
plt.savefig('exp1_predictions.pdf')
plt.savefig('exp1_predictions.png')
print("Saved exp1_predictions.pdf and exp1_predictions.png")

# =============================================================================
# Experiment 2: Decision Structure × Goal (N=4, 50% base rate)
# =============================================================================

exp2_data = {
    'singleton': {'find': 0.778, 'avoid': 0.222},  # P(ask "Which uncontaminated?")
    'set_id': {'find': 0.468, 'avoid': 0.532},
}

fig2, ax2 = plt.subplots(figsize=(5, 4))

x = np.array([0, 1])
width = 0.35

find_bars = ax2.bar(x - width/2, [exp2_data['singleton']['find'], exp2_data['set_id']['find']],
                     width, color=FIND_COLOR, label='Find uncontaminated', edgecolor='white', linewidth=1.5)
avoid_bars = ax2.bar(x + width/2, [exp2_data['singleton']['avoid'], exp2_data['set_id']['avoid']],
                      width, color=AVOID_COLOR, label='Avoid contamination', edgecolor='white', linewidth=1.5)

ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax2.set_ylabel('P(Ask "Which are uncontaminated?")')
ax2.set_xticks(x)
ax2.set_xticklabels(['Singleton\nselection', 'Set\nidentification'])
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.legend(loc='upper right', frameon=False)
ax2.set_title('Exp 2: Decision Structure × Goal\n(50% base rate, N=4)')

# Add effect size annotations
singleton_effect = exp2_data['singleton']['find'] - exp2_data['singleton']['avoid']
setid_effect = exp2_data['set_id']['find'] - exp2_data['set_id']['avoid']

ax2.annotate(f'Δ = {singleton_effect:.2f}', xy=(0, 0.95), ha='center', fontsize=9,
             color='black', fontweight='bold')
ax2.annotate(f'Δ = {setid_effect:.2f}', xy=(1, 0.95), ha='center', fontsize=9,
             color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('exp2_predictions.pdf')
plt.savefig('exp2_predictions.png')
print("Saved exp2_predictions.pdf and exp2_predictions.png")

# =============================================================================
# Combined figure for paper
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# Panel A: Exp 1
ax1.plot(base_rates, find_probs, 'o-', color=FIND_COLOR, linewidth=2,
         markersize=10, label='Find uncontaminated', markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(base_rates, avoid_probs, 's-', color=AVOID_COLOR, linewidth=2,
         markersize=10, label='Avoid contamination', markeredgecolor='white', markeredgewidth=1.5)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Contamination base rate')
ax1.set_ylabel('P(Ask "Which are uncontaminated?")')
ax1.set_xticks([0.2, 0.5, 0.8])
ax1.set_xticklabels(['20%', '50%', '80%'])
ax1.set_ylim(0, 1)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.legend(loc='center left', frameon=False)
ax1.set_title('A. Goal × Base Rate\n(Singleton, N=5)')
ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

# Panel B: Exp 2
x = np.array([0, 1])
width = 0.35
find_bars = ax2.bar(x - width/2, [exp2_data['singleton']['find'], exp2_data['set_id']['find']],
                     width, color=FIND_COLOR, label='Find uncontaminated', edgecolor='white', linewidth=1.5)
avoid_bars = ax2.bar(x + width/2, [exp2_data['singleton']['avoid'], exp2_data['set_id']['avoid']],
                      width, color=AVOID_COLOR, label='Avoid contamination', edgecolor='white', linewidth=1.5)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('P(Ask "Which are uncontaminated?")')
ax2.set_xticks(x)
ax2.set_xticklabels(['Singleton\nselection', 'Set\nidentification'])
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.legend(loc='upper right', frameon=False)
ax2.set_title('B. Decision Structure × Goal\n(50% base rate, N=4)')
ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

# Effect size annotations
ax2.annotate(f'Δ = {singleton_effect:.2f}', xy=(0, 0.95), ha='center', fontsize=9,
             color='black', fontweight='bold')
ax2.annotate(f'Δ = {setid_effect:.2f}', xy=(1, 0.95), ha='center', fontsize=9,
             color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('model_predictions_combined.pdf')
plt.savefig('model_predictions_combined.png')
print("Saved model_predictions_combined.pdf and model_predictions_combined.png")

# =============================================================================
# Print summary statistics
# =============================================================================

print("\n" + "="*60)
print("SUMMARY OF MODEL PREDICTIONS")
print("="*60)

print("\n--- Exp 1: Goal × Base Rate (singleton, N=5) ---")
for br in base_rates:
    goal_effect = exp1_data[br]['find'] - exp1_data[br]['avoid']
    print(f"Base rate {br*100:.0f}%: FIND={exp1_data[br]['find']:.3f}, "
          f"AVOID={exp1_data[br]['avoid']:.3f}, Δ={goal_effect:.3f}")

print("\n--- Exp 2: Decision Structure × Goal (50% base rate, N=4) ---")
for dt in ['singleton', 'set_id']:
    goal_effect = exp2_data[dt]['find'] - exp2_data[dt]['avoid']
    print(f"{dt}: FIND={exp2_data[dt]['find']:.3f}, "
          f"AVOID={exp2_data[dt]['avoid']:.3f}, Δ={goal_effect:.3f}")

print("\n--- Key Interaction ---")
print(f"Goal effect (singleton): {singleton_effect:.3f}")
print(f"Goal effect (set_id): {setid_effect:.3f}")
print(f"Interaction magnitude: {abs(singleton_effect - setid_effect):.3f}")

plt.show()
