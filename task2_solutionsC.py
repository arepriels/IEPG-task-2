import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import copy
import os

pd.options.display.max_columns = None

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'ieee9-wind.xlsx')
net_base = pp.from_excel(file_path)

# ─────────────────────────────────────────────
# NETWORK CONSTANTS  (confirmed from debug)
# ─────────────────────────────────────────────
WIND_TRAFO_IDX = 3   # trafo connecting wind sub-grid (lv=9) to main 230-kV ring (hv=?)

WIND_LOCATIONS = {
    'Bus 4\n(baseline)': 4,
    'Bus 7':              7,
    'Bus 5':              5,
}

# Reactive power modes for the wind plant
# Each entry: (label, q_mode)
#   q_mode = 'unity'       -> Q = 0
#   q_mode = 'overexcited' -> Q > 0  (inject reactive, pf = 0.95 cap)
#   q_mode = 'underexcited'-> Q < 0  (absorb reactive, pf = 0.95 ind)
PF_TARGET = 0.95

Q_MODES = {
    'Unity PF\n(Q=0)':          'unity',
    'Overexcited\n(pf=0.95)':   'overexcited',
    'Underexcited\n(pf=0.95)':  'underexcited',
}

# ─────────────────────────────────────────────
# WIND / LOAD PARAMETERS
# ─────────────────────────────────────────────
k        = 2.02
lambda_  = 11
Pwpp     = 180
wsin     = 3
wsr      = 12
wsout    = 20
N        = 100

np.random.seed(5489)
random.seed(5489)

# Pre-generate wind speeds & powers (same for every scenario)
ws_rand   = np.asarray([random.weibullvariate(lambda_, k) for _ in range(N)])
Pwpp_act  = []
for ws in ws_rand:
    if   ws < wsin:              Pwpp_act.append(0)
    elif ws < wsr:               Pwpp_act.append(Pwpp * (ws**3 - wsin**3) / (wsr**3 - wsin**3))
    elif ws < wsout:             Pwpp_act.append(Pwpp)
    else:                        Pwpp_act.append(0)
Pwpp_act = np.asarray(Pwpp_act)

# Load distribution parameters (same seed → same draws per scenario)
means    = np.asarray(net_base.load.p_mw)
sd       = np.random.uniform(1, 30, 3)
pq_ratio = np.asarray(net_base.load.q_mvar / net_base.load.p_mw)


# ─────────────────────────────────────────────
# HELPER: compute Q limit from P and power factor
# ─────────────────────────────────────────────
def q_from_pf(p_mw, pf, sign):
    """sign=+1 → overexcited (inject Q), sign=-1 → underexcited (absorb Q)."""
    if pf >= 1.0 or p_mw == 0:
        return 0.0
    return sign * p_mw * np.tan(np.arccos(pf))


# ─────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────
# results[wind_label][q_label] = {
#     'vm_pu':        list of arrays (one per converged sample), shape (n_buses,)
#     'line_load':    list of arrays, shape (n_lines,)
#     'pl_mw':        list of arrays, shape (n_lines,)
#     'n_conv':       int
# }
results = {}

for wind_label, wind_hv_bus in WIND_LOCATIONS.items():
    results[wind_label] = {}
    for q_label, q_mode in Q_MODES.items():

        vm_list, ll_list, pl_list = [], [], []
        n_conv = 0

        # Reset load seed so every scenario sees identical load samples
        np.random.seed(5489)
        random.seed(5489)
        # Re-draw wind (same as above)
        _ws = np.asarray([random.weibullvariate(lambda_, k) for _ in range(N)])
        _sd = np.random.uniform(1, 30, 3)

        for p in Pwpp_act:
            net = copy.deepcopy(net_base)
            # Relocate wind trafo
            net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = wind_hv_bus

            # Random load sample
            net.load.p_mw    = np.random.normal(means, _sd)
            net.load.q_mvar  = np.asarray(net.load.p_mw) * pq_ratio

            # Wind active power
            net.sgen.p_mw    = p
            net.sgen.max_p_mw = p
            net.sgen.min_p_mw = p

            # Reactive power mode
            if q_mode == 'unity':
                q_set          = 0.0
                net.sgen.max_q_mvar = 0.0
                net.sgen.min_q_mvar = 0.0
            elif q_mode == 'overexcited':
                q_set               = q_from_pf(p, PF_TARGET, +1)
                net.sgen.max_q_mvar = q_set
                net.sgen.min_q_mvar = q_set
            else:  # underexcited
                q_set               = q_from_pf(p, PF_TARGET, -1)
                net.sgen.max_q_mvar = q_set
                net.sgen.min_q_mvar = q_set

            net.sgen.q_mvar = q_set

            try:
                pp.runpp(net, init='flat', verbose=False)
                pp.runopp(net, init='pf', verbose=False)
                vm_list.append(net.res_bus.vm_pu.values.copy())
                ll_list.append(net.res_line.loading_percent.values.copy())
                pl_list.append(net.res_line.pl_mw.values.copy())
                n_conv += 1
            except:
                continue

        results[wind_label][q_label] = {
            'vm_pu':     np.array(vm_list)   if vm_list else None,   # (n_conv, n_bus)
            'line_load': np.array(ll_list)   if ll_list else None,   # (n_conv, n_line)
            'pl_mw':     np.array(pl_list)   if pl_list else None,   # (n_conv, n_line)
            'n_conv':    n_conv,
        }
        print(f"  {wind_label.replace(chr(10),' ')} | {q_label.replace(chr(10),' ')}"
              f" → {n_conv}/{N} converged")


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
WIND_LABELS = list(WIND_LOCATIONS.keys())
Q_LABELS    = list(Q_MODES.keys())
N_BUS       = len(net_base.bus)
N_LINE      = len(net_base.line)

# Colors per Q mode
Q_COLORS = {
    'Unity PF\n(Q=0)':         '#2196F3',   # blue
    'Overexcited\n(pf=0.95)':  '#E91E63',   # red/pink
    'Underexcited\n(pf=0.95)': '#FF9800',   # orange
}


def make_boxplot(ax, data_matrix, positions, color, width=0.18):
    """
    data_matrix: (n_samples, n_buses_or_lines)
    positions:   x positions for each box
    """
    if data_matrix is None or len(data_matrix) == 0:
        return
    bp = ax.boxplot(
        [data_matrix[:, i] for i in range(data_matrix.shape[1])],
        positions=positions,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.55),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1),
        capprops=dict(color=color, linewidth=1),
        flierprops=dict(marker='.', markersize=2, color=color, alpha=0.4),
        manage_ticks=False,
    )
    return bp


# ── FIGURE 1: Voltage magnitude variability ───────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig1.suptitle("Bus Voltage Magnitude Variability – PPF (N=100)",
              fontsize=13, fontweight='bold')

n_q     = len(Q_LABELS)
spacing = 0.22
bus_x   = np.arange(N_BUS)

for ax, wind_label in zip(axes1, WIND_LABELS):
    ax.set_title(wind_label.replace('\n', ' '), fontsize=10)
    legend_patches = []
    for qi, (q_label, color) in enumerate(Q_COLORS.items()):
        r = results[wind_label][q_label]
        offset = (qi - (n_q - 1) / 2) * spacing
        positions = bus_x + offset
        make_boxplot(ax, r['vm_pu'], positions, color, width=spacing * 0.85)
        legend_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.65,
                           label=f"{q_label.replace(chr(10),' ')} (n={r['n_conv']})"))

    ax.axhline(1.05, color='red', linestyle=':', linewidth=1.3, label='±5% limit')
    ax.axhline(0.95, color='red', linestyle=':', linewidth=1.3)
    ax.axhline(1.00, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Bus Index")
    ax.set_ylabel("Voltage (pu)")
    ax.set_xticks(bus_x)
    ax.set_ylim(0.87, 1.10)
    ax.legend(handles=legend_patches + [
        mpatches.Patch(facecolor='white', edgecolor='red',
                       linestyle=':', label='±5% limit')],
        fontsize=7, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task2_voltages.png'), dpi=150, bbox_inches='tight')
print("[Saved] task2_voltages.png")


# ── FIGURE 2: Branch loading variability ─────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig2.suptitle("Line Loading Variability (%) – PPF (N=100)",
              fontsize=13, fontweight='bold')

line_x = np.arange(N_LINE)

for ax, wind_label in zip(axes2, WIND_LABELS):
    ax.set_title(wind_label.replace('\n', ' '), fontsize=10)
    legend_patches = []
    for qi, (q_label, color) in enumerate(Q_COLORS.items()):
        r = results[wind_label][q_label]
        offset = (qi - (n_q - 1) / 2) * spacing
        positions = line_x + offset
        make_boxplot(ax, r['line_load'], positions, color, width=spacing * 0.85)
        legend_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.65,
                           label=f"{q_label.replace(chr(10),' ')} (n={r['n_conv']})"))

    ax.axhline(100, color='red', linestyle='--', linewidth=1.3, label='100% limit')
    ax.set_xlabel("Line Index")
    ax.set_ylabel("Loading (%)")
    ax.set_xticks(line_x)
    ax.legend(handles=legend_patches, fontsize=7, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task2_line_loading.png'), dpi=150, bbox_inches='tight')
print("[Saved] task2_line_loading.png")


# ── FIGURE 3: Active power losses variability ─────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig3.suptitle("Active Power Losses per Line (MW) – PPF (N=100)",
              fontsize=13, fontweight='bold')

for ax, wind_label in zip(axes3, WIND_LABELS):
    ax.set_title(wind_label.replace('\n', ' '), fontsize=10)
    legend_patches = []
    for qi, (q_label, color) in enumerate(Q_COLORS.items()):
        r = results[wind_label][q_label]
        offset = (qi - (n_q - 1) / 2) * spacing
        positions = line_x + offset
        make_boxplot(ax, r['pl_mw'], positions, color, width=spacing * 0.85)
        legend_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.65,
                           label=f"{q_label.replace(chr(10),' ')} (n={r['n_conv']})"))

    ax.set_xlabel("Line Index")
    ax.set_ylabel("Active Power Loss (MW)")
    ax.set_xticks(line_x)
    ax.legend(handles=legend_patches, fontsize=7, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'task2_losses.png'), dpi=150, bbox_inches='tight')
print("[Saved] task2_losses.png")

plt.show()

# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Wind':20s} {'Q mode':22s} {'n_conv':>6} {'V_min med':>10} "
      f"{'V_max med':>10} {'Max load med':>12} {'Loss med':>10}")
print("=" * 75)
for wind_label in WIND_LABELS:
    for q_label in Q_LABELS:
        r = results[wind_label][q_label]
        if r['vm_pu'] is not None and r['n_conv'] > 0:
            vm_med  = np.median(r['vm_pu'],     axis=0)
            ll_med  = np.median(r['line_load'], axis=0)
            pl_med  = np.median(r['pl_mw'],     axis=0)
            print(f"{wind_label.replace(chr(10),' '):20s} "
                  f"{q_label.replace(chr(10),' '):22s} "
                  f"{r['n_conv']:>6d} "
                  f"{vm_med.min():>10.4f} "
                  f"{vm_med.max():>10.4f} "
                  f"{ll_med.max():>12.1f} "
                  f"{pl_med.sum():>10.3f}")
        else:
            print(f"{wind_label.replace(chr(10),' '):20s} "
                  f"{q_label.replace(chr(10),' '):22s} "
                  f"{'0':>6} {'---':>10} {'---':>10} {'---':>12} {'---':>10}")
print("=" * 75)