import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# SCARA/grid parameters
# -----------------------
k = 20                    # grid size
cell_mm = 0.4             # mm per grid cell
L1 = 1.8                  # mm
L2 = 1.8                  # mm
r_min = abs(L1 - L2) / cell_mm          # min radius in cells
r_max = (L1 + L2) / cell_mm             # max radius in cells

# Desired base separation (mm) and conversion to cells
sep_mm = 6.2
sep_cells = sep_mm / cell_mm            # e.g., 6.2/0.4 = 15.5 cells

# Center reference (same as original)
icenter, jcenter = k // 2 - 1, k // 2 - 1   # (9,9) for k=20

# Choose integer base indices approximating the desired separation, clipped to grid
jbase1 = int(round(jcenter - sep_cells / 2))
jbase2 = int(round(jcenter + sep_cells / 2))
ibase1 = icenter
ibase2 = icenter
jbase1 = max(0, min(k-1, jbase1))
jbase2 = max(0, min(k-1, jbase2))

# Report actual achieved separation in mm (due to integer grid)
actual_sep_cells = abs(jbase2 - jbase1)
actual_sep_mm = actual_sep_cells * cell_mm
print(f"Actual base sep ~ {actual_sep_mm:.2f} mm (requested {sep_mm} mm)")

# Single target (grid index)
target = (6,9)

# -----------------------
# Reachability masks for each agent
# -----------------------
Y, X = np.ogrid[:k, :k]

dist1 = np.sqrt((Y - ibase1)**2 + (X - jbase1)**2)
reach1 = (dist1 >= r_min - 1e-9) & (dist1 <= r_max + 1e-9)

dist2 = np.sqrt((Y - ibase2)**2 + (X - jbase2)**2)
reach2 = (dist2 >= r_min - 1e-9) & (dist2 <= r_max + 1e-9)

# At least one agent must be able to reach the target
if not (reach1[target] or reach2[target]):
    raise ValueError(
        f"Target {target} not reachable by either agent with L1=L2={L1}mm and cell_mm={cell_mm}mm."
    )

# -----------------------
# Costs & actions (centralized DP on joint state)
# actions: 0=cont1, 1=cont2, 2=stop
# -----------------------
CONT_COST = 1.0
HUGE_PENALTY = 1e6
TARGET_REWARD = -150.0

# -----------------------
# Joint DP state: (i1, j1, i2, j2)
# Only states where agent1 in reach1 AND agent2 in reach2 are valid.
# -----------------------
valid4 = reach1[:, :, None, None] & reach2[None, None, :, :]

# -----------------------
# Precompute neighbor connectivity (degrees & adjacency) per agent
# -----------------------
def neighbor_degrees(mask: np.ndarray):
    up = np.zeros_like(mask, dtype=int);    up[1: , :] = (mask[1: , :] & mask[:-1, :])
    dn = np.zeros_like(mask, dtype=int);    dn[:-1, :] = (mask[:-1, :] & mask[1: , :])
    lf = np.zeros_like(mask, dtype=int);    lf[:, 1:] = (mask[:, 1:] & mask[:, :-1])
    rt = np.zeros_like(mask, dtype=int);    rt[:, :-1] = (mask[:, :-1] & mask[:, 1:])
    deg = up + dn + lf + rt
    return (up, dn, lf, rt, deg)

up1, dn1, lf1, rt1, deg1 = neighbor_degrees(reach1)
up2, dn2, lf2, rt2, deg2 = neighbor_degrees(reach2)

# Broadcast-friendly degrees
deg1_4 = deg1.astype(np.float64)[..., None, None]  # (k,k,1,1)
deg2_4 = deg2.astype(np.float64)[None, None, :, :]  # (1,1,k,k)

# Stop cost tensor: reward only if either agent is exactly on target
stop_cost = np.full((k, k, k, k), HUGE_PENALTY, dtype=np.float64)
Tmask = np.zeros((k, k), dtype=bool); Tmask[target] = True
stop_cost[(Tmask[..., None, None]) | (Tmask[None, None, ...])] = TARGET_REWARD

# -----------------------
# Vectorized Jacobi value iteration
# -----------------------
epsilon = 1e-8
max_iters = 20000

# Initialize value and policy
J = np.zeros((k, k, k, k), dtype=np.float64)
pistar = -np.ones((k, k, k, k), dtype=int)

# random init inside valid region
Jplus = np.random.rand(k, k, k, k).astype(np.float64)
Jplus[~valid4] = np.nan

t = 0
while True:
    t += 1
    if t > max_iters:
        print("Reached max_iters without full convergence.")
        break

    J = Jplus
    # Replace NaNs with zero only for intermediate neighbor-sum math
    J0 = np.nan_to_num(J, nan=0.0)

    # --- Agent 1 neighbor sums across axes (0,1) ---
    J_up1 = np.zeros_like(J0);  J_up1[1:, :, :, :] = J0[:-1, :, :, :]
    J_dn1 = np.zeros_like(J0);  J_dn1[:-1, :, :, :] = J0[1:, :, :, :]
    J_lf1 = np.zeros_like(J0);  J_lf1[:, 1:, :, :] = J0[:, :-1, :, :]
    J_rt1 = np.zeros_like(J0);  J_rt1[:, :-1, :, :] = J0[:, 1:, :, :]

    S1 = (J_up1 * up1[..., None, None] +
          J_dn1 * dn1[..., None, None] +
          J_lf1 * lf1[..., None, None] +
          J_rt1 * rt1[..., None, None])

    # Safe division for cont1 (no warnings)
    mask1b = np.broadcast_to(deg1_4 > 0, J0.shape)
    tmp1 = np.zeros_like(J0)
    np.divide(S1, np.broadcast_to(deg1_4, J0.shape), out=tmp1, where=mask1b)
    cont1 = np.full_like(J0, np.inf)
    cont1[mask1b] = CONT_COST + tmp1[mask1b]

    # --- Agent 2 neighbor sums across axes (2,3) ---
    J_up2 = np.zeros_like(J0);  J_up2[:, :, 1:, :] = J0[:, :, :-1, :]
    J_dn2 = np.zeros_like(J0);  J_dn2[:, :, :-1, :] = J0[:, :, 1:, :]
    J_lf2 = np.zeros_like(J0);  J_lf2[:, :, :, 1:] = J0[:, :, :, :-1]
    J_rt2 = np.zeros_like(J0);  J_rt2[:, :, :, :-1] = J0[:, :, :, 1:]

    # NOTE: correct broadcasting for agent-2 adjacency (two leading axes only)
    S2 = (J_up2 * up2[None, None, :, :] +
          J_dn2 * dn2[None, None, :, :] +
          J_lf2 * lf2[None, None, :, :] +
          J_rt2 * rt2[None, None, :, :])

    # Safe division for cont2 (no warnings)
    mask2b = np.broadcast_to(deg2_4 > 0, J0.shape)
    tmp2 = np.zeros_like(J0)
    np.divide(S2, np.broadcast_to(deg2_4, J0.shape), out=tmp2, where=mask2b)
    cont2 = np.full_like(J0, np.inf)
    cont2[mask2b] = CONT_COST + tmp2[mask2b]

    # --- Min over actions (0=cont1, 1=cont2, 2=stop) ---
    Cstack = np.stack([cont1, cont2, stop_cost], axis=-1)  # (..., 3)
    Jnew = np.min(Cstack, axis=-1)
    pi = np.argmin(Cstack, axis=-1).astype(int)

    # Keep invalid as NaN / -1
    Jnew[~valid4] = np.nan
    pi[~valid4] = -1

    Jplus = Jnew
    pistar = pi

    # Convergence over valid entries
    diff = np.nan_to_num(Jplus - J, nan=0.0)
    if np.linalg.norm(diff) <= epsilon:
        print(f"Converged in {t} iterations")
        break

# -----------------------
# Visualization (2D slices of the 4D value/policy)
# Slice A: agent2 fixed at its base (i2=ibase2, j2=jbase2), vary agent1 (i1,j1)
# Slice B: agent1 fixed at its base (i1=ibase1, j1=jbase1), vary agent2 (i2,j2)
# -----------------------
valid_A = valid4[:, :, ibase2, jbase2]
J_A = np.ma.array(J[:, :, ibase2, jbase2], mask=~valid_A)
pi_A = np.ma.array(pistar[:, :, ibase2, jbase2], mask=~valid_A)

valid_B = valid4[ibase1, jbase1, :, :]
J_B = np.ma.array(J[ibase1, jbase1, :, :], mask=~valid_B)
pi_B = np.ma.array(pistar[ibase1, jbase1, :, :], mask=~valid_B)

plt.figure(figsize=(12, 10))

# Policy slice when agent 2 is at base
plt.subplot(2, 2, 1)
im1 = plt.imshow(pi_A, cmap='hot', extent=[0, k-1, 0, k-1], origin='lower')
plt.scatter([jbase1], [ibase1], c='white', s=40, marker='x', label='Base 1')
plt.scatter([jbase2], [ibase2], c='white', s=40, marker='+', label='Base 2')
plt.scatter([target[1]], [target[0]], c='cyan', s=60, marker='o', label='Target')
plt.title('Policy slice (agent2 fixed at base)\n0=cont1, 1=cont2, 2=stop')
plt.legend(loc='upper right', fontsize=8)
plt.colorbar(im1, fraction=0.046, pad=0.04)

# Value slice when agent 2 is at base
plt.subplot(2, 2, 2)
im2 = plt.imshow(J_A, cmap='copper', extent=[0, k-1, 0, k-1], origin='lower')
plt.scatter([jbase1], [ibase1], c='white', s=40, marker='x')
plt.scatter([jbase2], [ibase2], c='white', s=40, marker='+')
plt.scatter([target[1]], [target[0]], c='cyan', s=60, marker='o')
plt.title('Value slice (agent2 fixed at base)')
plt.colorbar(im2, fraction=0.046, pad=0.04)

# Policy slice when agent 1 is at base
plt.subplot(2, 2, 3)
im3 = plt.imshow(pi_B, cmap='hot', extent=[0, k-1, 0, k-1], origin='lower')
plt.scatter([jbase1], [ibase1], c='white', s=40, marker='x', label='Base 1')
plt.scatter([jbase2], [ibase2], c='white', s=40, marker='+', label='Base 2')
plt.scatter([target[1]], [target[0]], c='cyan', s=60, marker='o', label='Target')
plt.title('Policy slice (agent1 fixed at base)\n0=cont1, 1=cont2, 2=stop')
plt.legend(loc='upper right', fontsize=8)
plt.colorbar(im3, fraction=0.046, pad=0.04)

# Value slice when agent 1 is at base
plt.subplot(2, 2, 4)
im4 = plt.imshow(J_B, cmap='copper', extent=[0, k-1, 0, k-1], origin='lower')
plt.scatter([jbase1], [ibase1], c='white', s=40, marker='x')
plt.scatter([jbase2], [ibase2], c='white', s=40, marker='+')
plt.scatter([target[1]], [target[0]], c='cyan', s=60, marker='o')
plt.title('Value slice (agent1 fixed at base)')
plt.colorbar(im4, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
