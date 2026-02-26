import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy # To create deep copies of the network, so changes in one scenario don’t affect others.
import os

# ─────────────────────────────────────────────
# 0. LOAD NETWORK
# ─────────────────────────────────────────────
#  Get the directory where the script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Constructs the full path to the Excel file containing the network data.
file_path = os.path.join(script_dir, 'ieee9-wind.xlsx')

# Ensure all columns are displayed when printing DataFrames
pd.options.display.max_columns = None
# Load the network data from Excel into a Pandapower network object
net_base = pp.from_excel(file_path)


# ── Network layout (confirmed from debug) ─────────────────────────────────
#
#  pp bus | kV   | role
#  -------+------+-----------------------------------------------
#    0    | 16.5 | Gen 1 terminal (lv of trafo 0 → hv at bus 3)
#    1    | 18.0 | Gen 2 terminal (lv of trafo 2 → hv at bus 7)
#    2    | 13.8 | Gen 3 terminal (lv of trafo 1 → hv at bus 5)
#    3    | 230  | IEEE Bus 1  (swing / Gen-1 HV side)
#    4    | 230  | IEEE Bus 4  (load bus) ← current wind HV connection
#    5    | 230  | IEEE Bus 3  (Gen-3 HV side)
#    6    | 230  | IEEE Bus 6  (load bus)
#    7    | 230  | IEEE Bus 2  (Gen-2 HV side)
#    8    | 230  | IEEE Bus 8  (load bus)
#    9    | 110  | Wind sub-grid HV bus
#   10    | 110  | Wind sub-grid intermediate bus
#   11    | 33   | Wind generator (sgen) terminal
#
#  TRAFO 3: hv=4 (230 kV), lv=9 (110 kV)  → connects wind to main grid  ← we move this
#  TRAFO 4: hv=10, lv=11                   → internal wind plant trafo
#
#  To relocate the wind plant we change hv_bus of TRAFO 3 (not trafo 4).
# ──────────────────────────────────────────────────────────────────────────

# Index of the trafo that connects the wind sub-grid to the main grid. We will modify this trafo's hv_bus to move the wind connection point.
WIND_TRAFO_IDX = 3          # trafo that links wind sub-grid <-> main 9-bus ring

# Dictionary mapping wind plant connection points (bus 4, 7, or 5).
WIND_LOCATIONS = {
    'Bus 4 (baseline)': 4,  # original connection point
    'Bus 7':            7,  # Gen-2 HV bus  (IEEE Bus 2 side)
    'Bus 5':            5,  # Gen-3 HV bus  (IEEE Bus 3 side)
}

# Dictionary defining load scaling factors
LOAD_SCENARIOS = {
    'Baseline (100%)':   1.0,
    'Load +50% (150%)':  1.5,
    'Load +100% (200%)': 2.0,
}

# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
# reates a new, independent network for each scenario (wind location + load level)
def build_scenario_net(base_net, wind_hv_bus: int, load_scale: float):
    """Deep-copy base_net, move wind trafo hv_bus, scale loads."""
    net = copy.deepcopy(base_net)
    # change the high-voltage bus of the transformer at index WIND_TRAFO_IDX to wind_hv_bus
    net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = wind_hv_bus
    net.load['p_mw']   *= load_scale
    net.load['q_mvar'] *= load_scale
    # return the modified network
    return net

# function that runs the OPF for a given scenario and collects results
def run_scenario(base_net, wind_hv_bus, load_scale):
    net = build_scenario_net(base_net, wind_hv_bus, load_scale)
    try:
        # init='flat' (all buses at 1 pu / 0 deg) is robust to topology changes
        pp.runopp(net, init='flat', verbose=False)
        return {
            'converged':  True,
            'cost':       net.res_cost,
            'vm_pu':      net.res_bus.vm_pu.values.copy(),
            'line_load':  net.res_line.loading_percent.values.copy(),
            'trafo_load': net.res_trafo.loading_percent.values.copy(),
            'losses_mw':  net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum(),
        }
    except Exception as e:
        print(f"  !! OPF did not converge: {e}")
        return {'converged': False}


# ─────────────────────────────────────────────
# 2. RUN ALL SCENARIOS
# ─────────────────────────────────────────────
results = {}

for wind_label, wind_hv_bus in WIND_LOCATIONS.items():
    for load_label, load_scale in LOAD_SCENARIOS.items():
        key = (wind_label, load_label)
        print(f"\n{'─'*55}")
        print(f"  Wind @ {wind_label}  |  {load_label}")
        print(f"{'─'*55}")
        r = run_scenario(net_base, wind_hv_bus, load_scale)
        results[key] = r
        if r['converged']:
            print(f"  Cost              : {r['cost']:.2f}")
            print(f"  Losses            : {r['losses_mw']:.3f} MW")
            print(f"  Voltage (min/max) : {r['vm_pu'].min():.4f} / {r['vm_pu'].max():.4f} pu")
            print(f"  Max line loading  : {r['line_load'].max():.1f} %")


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
LOAD_COLORS  = ['#2196F3', '#FF9800', '#E91E63']
LOAD_HATCHES = ['', '///', 'xxx']

n_buses = len(net_base.bus)
n_lines = len(net_base.line)
bus_x   = np.arange(n_buses)
line_x  = np.arange(n_lines)
bar_w   = 0.25

wind_labels = list(WIND_LOCATIONS.keys())
load_labels = list(LOAD_SCENARIOS.keys())


# ── Figure 1: Voltage magnitudes (grouped scatter plot with guides) ─────────────────────────────────
plt.figure(figsize=(12, 6))
plt.title("Bus Voltage Magnitudes", fontsize=13, fontweight='bold')

# Define markers and colors for each scenario
markers = ['o', 's', '^']  # circle, square, triangle
colors = ['#2196F3', '#FF9800', '#E91E63']  # blue, orange, pink
labels = []
jitter = 0.1  # Small horizontal offset within each bus group

# Add light grey vertical guides for each bus
for bus in np.arange(n_buses):
    plt.axvline(x=bus, color='lightgrey', linestyle=':', linewidth=0.7, zorder=0)

# Plot each scenario
for i, (wind_label, wind_hv_bus) in enumerate(WIND_LOCATIONS.items()):
    for j, (load_label, load_scale) in enumerate(LOAD_SCENARIOS.items()):
        key = (wind_label, load_label)
        r = results[key]
        if r['converged']:
            label = f"{wind_label} | {load_label}"
            # Add small jitter to x positions within each bus group
            x_jittered = np.arange(n_buses) + np.random.uniform(-jitter, jitter, n_buses)
            plt.scatter(
                x_jittered, r['vm_pu'],
                marker=markers[j], color=colors[i], label=label, s=60, alpha=0.7,
                edgecolor='black', linewidth=0.5, zorder=1
            )
        else:
            label = f"{wind_label} | {load_label} (no conv.)"
            plt.scatter(
                np.arange(n_buses), [0.8]*n_buses,
                marker='x', color='lightgrey', label=label, s=60, alpha=0.7,
                edgecolor='black', linewidth=0.5, zorder=1
            )

# Add reference lines
plt.axhline(1.05, color='red', linestyle=':', linewidth=1.2, label='±5% limit')
plt.axhline(0.95, color='red', linestyle=':', linewidth=1.2)
plt.axhline(1.00, color='green', linestyle='--', linewidth=0.9, label='1.0 pu')

# Labels and legend
plt.xlabel("Bus Index")
plt.ylabel("Voltage (pu)")
plt.xticks(np.arange(n_buses))
plt.ylim(0.89, 1.1)
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.45)

plt.tight_layout()
""" plt.savefig(os.path.join(script_dir, 'task1_voltages_grouped_scatter_guides.png'), dpi=150, bbox_inches='tight')
print("\n[Saved] task1_voltages_grouped_scatter_guides.png")
"""  

# ── Figure 2: Line loading ─────────────────────────────────────────────────
# ── Figure 2: Line loading (grouped scatter plot with guides) ─────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5))
ax.set_title("Line Loading (%)", fontsize=13, fontweight='bold')

# Define markers and colors for each scenario
markers = ['o', 's', '^']  # circle, square, triangle
colors = ['#2196F3', '#FF9800', '#E91E63']  # blue, orange, pink
jitter = 0.1  # Small horizontal offset within each line group

# Add light grey vertical guides for each line
for line in np.arange(n_lines):
    ax.axvline(x=line, color='lightgrey', linestyle=':', linewidth=0.7, zorder=0)

# Plot each scenario
for i, (wind_label, wind_hv_bus) in enumerate(WIND_LOCATIONS.items()):
    for j, (load_label, load_scale) in enumerate(LOAD_SCENARIOS.items()):
        key = (wind_label, load_label)
        r = results[key]
        if r['converged']:
            label = f"{wind_label} | {load_label}"
            # Add small jitter to x positions within each line group
            x_jittered = np.arange(n_lines) + np.random.uniform(-jitter, jitter, n_lines)
            ax.scatter(
                x_jittered, r['line_load'],
                marker=markers[j], color=colors[i], label=label, s=60, alpha=0.7,
                edgecolor='black', linewidth=0.5, zorder=1
            )
        else:
            label = f"{wind_label} | {load_label} (no conv.)"
            ax.scatter(
                np.arange(n_lines), [-10]*n_lines,  # Plot below 0% to indicate no convergence
                marker='x', color='lightgrey', label=label, s=60, alpha=0.7,
                edgecolor='black', linewidth=0.5, zorder=1
            )

# Add reference line
ax.axhline(100, color='red', linestyle='--', linewidth=1.2, label='100% thermal limit')

# Labels and legend
ax.set_xlabel("Line Index")
ax.set_ylabel("Loading (%)")
ax.set_xticks(np.arange(n_lines))
ax.set_ylim(0, 120)  # Focus on relevant loading range
ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.45)

plt.tight_layout(rect=[0, 0, 0.9, 1])
""" plt.savefig(os.path.join(script_dir, 'task1_line_loading_final.png'), dpi=150, bbox_inches='tight')
print("[Saved] task1_line_loading_final.png") """

# ── Figure 3: Summary – cost & losses ─────────────────────────────────────
# ── Figure 3: Summary – cost & losses (bar plot) ───────────────────────────────────────────────────────────────────────────────────────────────────────
fig3, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Economic Dispatch Summary", fontsize=13, fontweight='bold')

labels, costs, losses = [], [], []
palette = []
for i, (wind_label, _) in enumerate(WIND_LOCATIONS.items()):
    for j, (load_label, _) in enumerate(LOAD_SCENARIOS.items()):
        palette.append(colors[i])
        key = (wind_label, load_label)
        r = results[key]
        labels.append(f"{wind_label}\n{load_label.split('(')[0].strip()}")
        costs.append(r['cost'] if r['converged'] else 0)
        losses.append(r['losses_mw'] if r['converged'] else 0)

x = np.arange(len(labels))

# Plot cost
ax_c.bar(x, costs, color=palette, edgecolor='black', width=0.7)
ax_c.set_title("Objective Function Value")
ax_c.set_ylabel("Cost (€/h)")
ax_c.set_xticks(x)
ax_c.set_xticklabels(labels, fontsize=8)
ax_c.grid(axis='y', linestyle='--', alpha=0.45)

# Plot losses
ax_l.bar(x, losses, color=palette, edgecolor='black', width=0.7)
ax_l.set_title("Total Network Losses")
ax_l.set_ylabel("Losses (MW)")
ax_l.set_xticks(x)
ax_l.set_xticklabels(labels, fontsize=8)
ax_l.grid(axis='y', linestyle='--', alpha=0.45)

plt.tight_layout(rect=[0, 0, 1, 1])
""" plt.savefig(os.path.join(script_dir, 'task1_summary_final.png'), dpi=150, bbox_inches='tight')
print("[Saved] task1_summary_final.png")
 """

# ── Figure 4: Active Power Losses (grouped scatter plot with guides) ─────────────────────────────────
# ── Figure 4: Active Power Losses (grouped scatter plot, lines and transformers separated) ─────────────────────────────────
fig4, ax = plt.subplots(figsize=(14, 6))
ax.set_title("Active Power Losses", fontsize=14, fontweight='bold')

# Define markers and colors for each scenario
markers = ['o', 's', '^']  # circle, square, triangle
colors = ['#2196F3', '#FF9800', '#E91E63']  # blue, orange, pink
jitter = 0.2  # Slightly larger jitter to avoid overlap

# Calculate max loss for y-axis scaling
max_loss = max([r['losses_mw'] for r in results.values() if r['converged']], default=1.0)

# Number of lines and transformers
n_lines = len(net_base.line)
n_trafos = len(net_base.trafo)

# Plot each scenario
for i, (wind_label, wind_hv_bus) in enumerate(WIND_LOCATIONS.items()):
    for j, (load_label, load_scale) in enumerate(LOAD_SCENARIOS.items()):
        key = (wind_label, load_label)
        r = results[key]
        if r['converged']:
            label = f"{wind_label} | {load_label}"
            # Get the network for this scenario
            net = build_scenario_net(net_base, wind_hv_bus, load_scale)
            pp.runopp(net, init='flat', verbose=False)
            # Combine line and trafo losses
            line_losses = net.res_line.pl_mw
            trafo_losses = net.res_trafo.pl_mw
            losses_mw = np.concatenate([line_losses, trafo_losses])

            # X positions: lines 0 to n_lines-1, transformers 0 to n_trafos-1
            x_line = np.arange(n_lines) + np.random.uniform(-jitter, jitter, n_lines)
            x_trafo = np.arange(n_trafos) + n_lines + np.random.uniform(-jitter, jitter, n_trafos)
            x_positions = np.concatenate([x_line, x_trafo])

            ax.scatter(
                x_positions, losses_mw,
                marker=markers[j], color=colors[i], label=label, s=70, alpha=0.8,
                edgecolor='black', linewidth=0.5, zorder=1
            )
        else:
            label = f"{wind_label} | {load_label} (no conv.)"
            ax.scatter(
                [], [],
                marker='x', color='lightgrey', label=label, s=70, alpha=0.8,
                edgecolor='black', linewidth=0.5, zorder=1
            )

# Add light grey vertical guides for each branch
for i in np.arange(n_lines):
    ax.axvline(x=i, color='lightgrey', linestyle=':', linewidth=0.7, zorder=0)
for i in np.arange(n_trafos):
    ax.axvline(x=n_lines + i, color='lightgrey', linestyle=':', linewidth=0.7, zorder=0)


# Add labels for lines and transformers
ax.text(n_lines/2, -0.15 * max_loss, 'Lines', ha='center', va='top', fontsize=11, fontweight='bold')
ax.text(n_lines + n_trafos/2, -0.15 * max_loss, 'Transformers', ha='center', va='top', fontsize=11, fontweight='bold')

# Add reference line at 0 MW
ax.axhline(0, color='black', linestyle='-', linewidth=0.7)

# Labels and legend
ax.set_ylabel("Active Power Losses (MW)")
ax.set_xticks(list(np.arange(n_lines)) + list(n_lines + np.arange(n_trafos)))
ax.set_xticklabels(list(np.arange(n_lines)) + list(np.arange(n_trafos)))
ax.set_ylim(0, 7.5)
ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0, 0.9, 1])

plt.show()
print("\n[DONE] All scenarios complete.")