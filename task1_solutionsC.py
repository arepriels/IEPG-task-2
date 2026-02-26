import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# ─────────────────────────────────────────────
# 0. LOAD NETWORK
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'ieee9-wind.xlsx')

pd.options.display.max_columns = None
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

WIND_TRAFO_IDX = 3          # trafo that links wind sub-grid <-> main 9-bus ring

WIND_LOCATIONS = {
    'Bus 4 (baseline)': 4,  # original connection point
    'Bus 7':            7,  # Gen-2 HV bus  (IEEE Bus 2 side)
    'Bus 5':            5,  # Gen-3 HV bus  (IEEE Bus 3 side)
}

LOAD_SCENARIOS = {
    'Baseline (100%)':   1.0,
    'Load +50% (150%)':  1.5,
    'Load +100% (200%)': 2.0,
}

# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
def build_scenario_net(base_net, wind_hv_bus: int, load_scale: float):
    """Deep-copy base_net, move wind trafo hv_bus, scale loads."""
    net = copy.deepcopy(base_net)
    net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = wind_hv_bus
    net.load['p_mw']   *= load_scale
    net.load['q_mvar'] *= load_scale
    return net


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

# ── Figure 1: Voltage magnitudes ──────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig1.suptitle("Bus Voltage Magnitudes (pu) – Economic Dispatch",
              fontsize=13, fontweight='bold')

for ax, wind_label in zip(axes1, wind_labels):
    ax.set_title(f"Wind @ {wind_label}", fontsize=10)
    for i, (load_label, color, hatch) in enumerate(
            zip(load_labels, LOAD_COLORS, LOAD_HATCHES)):
        r = results[(wind_label, load_label)]
        offset = (i - 1) * bar_w
        if r['converged']:
            ax.bar(bus_x + offset, r['vm_pu'], width=bar_w,
                   label=load_label, color=color, hatch=hatch,
                   edgecolor='black', linewidth=0.5)
        else:
            ax.bar(bus_x + offset, [0]*n_buses, width=bar_w,
                   label=f"{load_label} (no conv.)",
                   color='lightgrey', hatch='\\\\', edgecolor='black')

    ax.axhline(1.05, color='red',   linestyle=':', linewidth=1.2,
               label='±5 % limit')
    ax.axhline(0.95, color='red',   linestyle=':', linewidth=1.2)
    ax.axhline(1.00, color='green', linestyle='--', linewidth=0.9,
               label='1.0 pu')
    ax.set_xlabel("Bus Index")
    ax.set_ylabel("Voltage (pu)")
    ax.set_xticks(bus_x)
    ax.set_ylim(0.85, 1.12)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.45)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task1_voltages.png'),
            dpi=150, bbox_inches='tight')
print("\n[Saved] task1_voltages.png")

# ── Figure 2: Line loading ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig2.suptitle("Line Loading (%) – Economic Dispatch",
              fontsize=13, fontweight='bold')

for ax, wind_label in zip(axes2, wind_labels):
    ax.set_title(f"Wind @ {wind_label}", fontsize=10)
    for i, (load_label, color, hatch) in enumerate(
            zip(load_labels, LOAD_COLORS, LOAD_HATCHES)):
        r = results[(wind_label, load_label)]
        offset = (i - 1) * bar_w
        if r['converged']:
            ax.bar(line_x + offset, r['line_load'], width=bar_w,
                   label=load_label, color=color, hatch=hatch,
                   edgecolor='black', linewidth=0.5)

    ax.axhline(100, color='red', linestyle='--', linewidth=1.2,
               label='100 % thermal limit')
    ax.set_xlabel("Line Index")
    ax.set_ylabel("Loading (%)")
    ax.set_xticks(line_x)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.45)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task1_line_loading.png'),
            dpi=150, bbox_inches='tight')
print("[Saved] task1_line_loading.png")

# ── Figure 3: Summary – cost & losses ─────────────────────────────────────
fig3, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Economic Dispatch Summary", fontsize=13, fontweight='bold')

labels, costs, losses = [], [], []
palette = LOAD_COLORS * len(wind_labels)

for wind_label in wind_labels:
    for load_label in load_labels:
        r = results[(wind_label, load_label)]
        labels.append(f"{wind_label}\n{load_label.split('(')[0].strip()}")
        costs.append( r['cost']       if r['converged'] else 0)
        losses.append(r['losses_mw']  if r['converged'] else 0)

x = np.arange(len(labels))

for ax, vals, ylabel, title in [
        (ax_c, costs,  "Cost (€/h)",  "Objective Function Value"),
        (ax_l, losses, "Losses (MW)", "Total Network Losses")]:
    bars = ax.bar(x, vals, color=palette[:len(labels)], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    vmax = max(vals) if any(vals) else 1
    for bar, val in zip(bars, vals):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + vmax*0.01,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task1_summary.png'),
            dpi=150, bbox_inches='tight')
print("[Saved] task1_summary.png")

plt.show()
print("\n[DONE] All scenarios complete.")