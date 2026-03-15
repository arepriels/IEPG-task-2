import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import copy
import os
script_dir = os.path.dirname(os.path.abspath(__file__))


output_folder = os.path.join(script_dir, 'plots-heatmaps')  # Creates a 'plots' folder in the same directory as your script
os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist


# ─────────────────────────────────────────────
# 0. LOAD NETWORK
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'ieee9-wind.xlsx')

net_base = pp.from_excel(file_path)

WIND_TRAFO_IDX = 3
WIND_LOCATIONS = {
    'Bus 4': 3,
    'Bus 7':            6,
    'Bus 5':            4,
}
LOAD_SCENARIOS = {
    'Baseline\n(100%)':    1.0,
    '+50% load\n(150%)':   1.5,
    '+100% load\n(200%)':  2.0,
}

# ─────────────────────────────────────────────
# 1. RUN ALL SCENARIOS
# ─────────────────────────────────────────────
def run_scenario(base_net, wind_hv_bus, load_scale):
    net = copy.deepcopy(base_net)
    net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = wind_hv_bus
    net.load['p_mw']   *= load_scale
    net.load['q_mvar'] *= load_scale
    try:
        pp.runopp(net, init='flat', verbose=False)
        return {
            'converged':  True,
            'cost':       net.res_cost,
            'vm_pu':      net.res_bus.vm_pu.values.copy(),
            'line_load':  net.res_line.loading_percent.values.copy(),
            'losses_mw':  net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum(),
        }
    except:
        return {'converged': False}

results = {}
for wl, wb in WIND_LOCATIONS.items():
    for ll, ls in LOAD_SCENARIOS.items():
        results[(wl, ll)] = run_scenario(net_base, wb, ls)

wind_labels = list(WIND_LOCATIONS.keys())
load_labels = list(LOAD_SCENARIOS.keys())
n_buses     = len(net_base.bus)
n_lines     = len(net_base.line)

# Row order for heatmap: wind location is the outer group
row_labels = [f"{wl}  |  {ll.replace(chr(10), ' ')}"
              for wl in wind_labels for ll in load_labels]
n_rows = len(row_labels)

# ─────────────────────────────────────────────
# BUILD MATRICES
# ─────────────────────────────────────────────
vm_matrix   = np.full((n_rows, n_buses), np.nan)
ll_matrix   = np.full((n_rows, n_lines), np.nan)
loss_matrix = np.full((n_rows, n_lines), np.nan)
cost_vec    = np.full(n_rows, np.nan)

for i, wl in enumerate(wind_labels):
    for j, ll in enumerate(load_labels):
        row = i * len(load_labels) + j
        r   = results[(wl, ll)]
        if r['converged']:
            vm_matrix[row]   = r['vm_pu']
            ll_matrix[row]   = r['line_load']
            loss_matrix[row] = r['res_line_pl'] if 'res_line_pl' in r else np.zeros(n_lines)
            cost_vec[row]    = r['cost']

# Re-collect line losses properly
for i, wl in enumerate(wind_labels):
    for j, ll in enumerate(load_labels):
        row = i * len(load_labels) + j
        r   = results[(wl, ll)]
        if r['converged']:
            # Re-run to get per-line losses (stored in full run above)
            net = copy.deepcopy(net_base)
            net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = WIND_LOCATIONS[wl]
            net.load['p_mw']   *= LOAD_SCENARIOS[ll]
            net.load['q_mvar'] *= LOAD_SCENARIOS[ll]
            try:
                pp.runopp(net, init='flat', verbose=False)
                loss_matrix[row] = net.res_line.pl_mw.values
            except:
                pass


# ─────────────────────────────────────────────
# FIGURE 1 — VOLTAGE HEATMAP
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 5))

# Custom diverging colormap centred on 1.0 pu
# green = near 1.0, red = near limits
cmap_v = plt.cm.RdYlGn
norm_v = mcolors.TwoSlopeNorm(vmin=0.93, vcenter=1.00, vmax=1.06)

# Mask non-converged rows (NaN)
masked_vm = np.ma.masked_invalid(vm_matrix)
im1 = ax1.imshow(masked_vm, aspect='auto', cmap=cmap_v, norm=norm_v)

# Annotate each cell with the value or "n/c"
for row in range(n_rows):
    for col in range(n_buses):
        if np.isnan(vm_matrix[row, col]):
            ax1.text(col, row, 'n/c', ha='center', va='center',
                     fontsize=7, color='grey')
        else:
            val = vm_matrix[row, col]
            color = 'white' if (val < 0.96 or val > 1.04) else 'black'
            ax1.text(col, row, f'{val:.3f}', ha='center', va='center',
                     fontsize=7, color=color, fontweight='bold')

# Draw horizontal separators between wind location groups
for sep in [3, 6]:
    ax1.axhline(sep - 0.5, color='black', linewidth=2)

# Axes labels
ax1.set_xticks(range(n_buses))
ax1.set_xticklabels([f'Bus {i}' for i in range(n_buses)], fontsize=8)
ax1.set_yticks(range(n_rows))
ax1.set_yticklabels(row_labels, fontsize=8)
ax1.set_xlabel("Bus", fontsize=10)
ax1.set_title("Bus Voltage Magnitudes",
              fontsize=11, fontweight='bold')

cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.025, pad=0.02)
cbar1.set_label('Voltage (pu)', fontsize=9)
cbar1.ax.axhline(y=1.05, color='white', linewidth=1.5, linestyle='--')
cbar1.ax.axhline(y=0.95, color='white', linewidth=1.5, linestyle='--')

plt.tight_layout()
plt.savefig(f"voltage-heatmap.png")
plt.savefig(
    os.path.join(output_folder, "voltage-heatmap.png"),
    dpi=1500,
    bbox_inches='tight'
)

# ─────────────────────────────────────────────
# MAKE MATRICES AGAIN WITH LINES + TRANSFORMERS
# ─────────────────────────────────────────────
# Number of transformers
n_trafos = len(net_base.trafo)

# Initialize matrices with lines + transformers
ll_matrix   = np.full((n_rows, n_lines + n_trafos), np.nan)
loss_matrix = np.full((n_rows, n_lines + n_trafos), np.nan)

# Re-run scenarios and collect line + transformer data
for i, wl in enumerate(wind_labels):
    for j, ll in enumerate(load_labels):
        row = i * len(load_labels) + j
        r   = results[(wl, ll)]
        if r['converged']:
            # Re-run to get per-line and per-trafo results
            net = copy.deepcopy(net_base)
            net.trafo.at[WIND_TRAFO_IDX, 'hv_bus'] = WIND_LOCATIONS[wl]
            net.load['p_mw']   *= LOAD_SCENARIOS[ll]
            net.load['q_mvar'] *= LOAD_SCENARIOS[ll]
            try:
                pp.runopp(net, init='flat', verbose=False)
                # Line loading
                ll_matrix[row, :n_lines] = net.res_line.loading_percent.values
                # Transformer loading
                ll_matrix[row, n_lines:] = net.res_trafo.loading_percent.values
                # Line losses
                loss_matrix[row, :n_lines] = net.res_line.pl_mw.values
                # Transformer losses
                loss_matrix[row, n_lines:] = net.res_trafo.pl_mw.values
            except:
                pass

# ─────────────────────────────────────────────
# FIGURE 2 — LINE LOADING HEATMAP
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 5))

cmap_l = plt.cm.YlOrRd
norm_l = mcolors.Normalize(vmin=0, vmax=100)

masked_ll = np.ma.masked_invalid(ll_matrix)
im2 = ax2.imshow(masked_ll, aspect='auto', cmap=cmap_l, norm=norm_l)

# Annotate cells
for row in range(n_rows):
    for col in range(n_lines + n_trafos):
        if np.isnan(ll_matrix[row, col]):
            ax2.text(col, row, 'n/c', ha='center', va='center', fontsize=8, color='grey')
        else:
            val = ll_matrix[row, col]
            color = 'white' if val > 75 else 'black'
            ax2.text(col, row, f'{val:.1f}%', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

# Separators
for sep in [3, 6]:
    ax2.axhline(sep - 0.5, color='black', linewidth=2)

# X-axis labels: lines first, then transformers
xtick_labels = [f'Line {i}' for i in range(n_lines)] + [f'Trafo {i}' for i in range(n_trafos)]
ax2.set_xticks(range(n_lines + n_trafos))
ax2.set_xticklabels(xtick_labels, fontsize=8, rotation=45)
ax2.set_yticks(range(n_rows))
ax2.set_yticklabels(row_labels, fontsize=8)
ax2.set_xlabel("Branch (Line/Transformer)", fontsize=10)
ax2.set_title("Line & Transformer Loading ", fontsize=11, fontweight='bold')

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
cbar2.set_label('Loading (%)', fontsize=9)

plt.tight_layout()
plt.savefig(f"lineloading-heatmap.png")
plt.savefig(os.path.join(output_folder, 'lineloading-heatmap.png'), dpi=1500, bbox_inches='tight')

# ─────────────────────────────────────────────
# FIGURE 3 — LOSS HEATMAP
# ─────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))

cmap_p = plt.cm.Blues
norm_p = mcolors.Normalize(vmin=0, vmax=np.nanmax(loss_matrix) * 1.1 or 1)

masked_pl = np.ma.masked_invalid(loss_matrix)
im3 = ax3.imshow(masked_pl, aspect='auto', cmap=cmap_p, norm=norm_p)

# Annotate cells
for row in range(n_rows):
    for col in range(n_lines + n_trafos):
        if np.isnan(loss_matrix[row, col]):
            ax3.text(col, row, 'n/c', ha='center', va='center', fontsize=8, color='grey')
        else:
            val = loss_matrix[row, col]
            color = 'white' if val > np.nanmax(loss_matrix) * 0.6 else 'black'
            ax3.text(col, row, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

# Separators
for sep in [3, 6]:
    ax3.axhline(sep - 0.5, color='black', linewidth=2)

# X-axis labels: lines first, then transformers
ax3.set_xticks(range(n_lines + n_trafos))
ax3.set_xticklabels(xtick_labels, fontsize=8, rotation=45)
ax3.set_yticks(range(n_rows))
ax3.set_yticklabels(row_labels, fontsize=8)
ax3.set_xlabel("Branch (Line/Transformer)", fontsize=10)
ax3.set_title("Active Power Losses per Branch", fontsize=11, fontweight='bold')

cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
cbar3.set_label('Active Power Loss (MW)', fontsize=9)

plt.tight_layout()
plt.savefig(f"loss-heatmap.png")
plt.savefig(os.path.join(output_folder, 'loss-heatmap.png'), dpi=1500, bbox_inches='tight')
