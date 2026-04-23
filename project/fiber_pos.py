import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters (Tuning)
# =========================
CELL = 0.2
SPACING_UNITS = 6.2
S_CELLS = int(round(SPACING_UNITS / CELL))

# SCARA geometry & motion
R = 1.8                         # link length (alpha and beta)
ALPHA_STEP = np.deg2rad(2)      # α step magnitude
BETA_STEP  = np.deg2rad(2)      # β step magnitude
REACH_TOL  = 0.10               # tip-to-target tolerance (units)

# Movement freedom
ALLOW_ALPHA_CW = True           # allow clockwise α
ALLOW_BETA_DECREASE = True      # allow β decrease (still clamped to [0,π])

# Optional soft alpha-first bias
ALPHA_ALIGN_TOL = np.deg2rad(10)
W_ALPHA_FIRST   = 0.05          # set 0 to disable

# Joint limits (β uses: 0=retracted, π=extended)
ALPHA_MIN, ALPHA_MAX = 0.0, 2*np.pi
BETA_MIN,  BETA_MAX  = 0.0, np.pi

# Collision geometry (capsule approx)
LINK_RAD   = 0.25
SAFETY_PAD = 0.05
CAPSULE_R  = LINK_RAD + SAFETY_PAD

# Potential-field (DP-style soft costs)
SIGMA_AGENTS_UNITS  = 1.2
SIGMA_TARGETS_UNITS = 1.6
LAMBDA_AGENTS       = 3.0
LAMBDA_REACHED      = 8.0
W_ATTRACT           = 1.0       # attraction to own target (distance)

# --- Anti-oscillation weights ---
W_STEP        = 0.01            # tiny cost to move at all
W_STEP_NORM   = 0.01            # cost per step magnitude (normalized by step sizes)
W_REVERSE_A   = 0.20            # penalty for reversing α direction
W_REVERSE_B   = 0.20            # penalty for reversing β direction
W_PROGRESS    = 0.50            # penalize lack of progress: adds (d1 - d0) * W_PROGRESS
W_HOLD        = 0.02            # small base cost for HOLD; lower lets it wait more

# Action enumeration
INCLUDE_HOLD = True             # include (0,0) so it can wait to avoid ping-pong

# Display / sim
PAUSE = 0.04
NUM_STEPS = 900
RNG_SEED = 7

# =========================
# Base layout (63 bases in a centered pyramid)
# =========================
COL_COUNTS = [9, 10, 9, 8, 7, 6, 5, 4, 3, 2]
C = len(COL_COUNTS); H = max(COL_COUNTS)
MARGIN = 2 * S_CELLS
k = int(max((C - 1)*S_CELLS + 1 + 2*MARGIN, (H - 1)*S_CELLS + 1 + 2*MARGIN))
if k % 2 == 0: k += 1
BOARD_SIZE_UNITS = k * CELL

def build_bases_pyramid(col_counts, k, s_cells, cell_size):
    cx = (k - 1)//2; cy = (k - 1)//2
    x0 = cx - (len(col_counts) - 1)*s_cells//2
    pts = []
    for c, count in enumerate(col_counts):
        x = x0 + c*s_cells
        y0 = cy - (count - 1)*s_cells//2
        for r in range(count):
            y = y0 + r*s_cells
            pts.append([x*cell_size, y*cell_size])
    return np.array(pts, dtype=float)

bases = build_bases_pyramid(COL_COUNTS, k, S_CELLS, CELL)
N = len(bases)

# =========================
# Targets (reachable annulus)
# =========================
rng = np.random.default_rng(RNG_SEED)
def random_target_near_base(base, r=R, margin=0.2):
    rho = rng.uniform(0.5*r, 2.0*r - margin)  # ensure reachable
    ang = rng.uniform(0, 2*np.pi)
    return base + rho * np.array([np.cos(ang), np.sin(ang)])
targets = np.array([random_target_near_base(b) for b in bases])

# =========================
# Kinematics (α, β with β=0=retracted, β=π=extended)
# =========================
def fk_alpha_beta(base, alpha, beta, r=R):
    """Forward kinematics; returns elbow, tip."""
    theta2_rel = beta - np.pi
    bx, by = base
    ex = bx + r * np.cos(alpha)
    ey = by + r * np.sin(alpha)
    tx = ex + r * np.cos(alpha + theta2_rel)
    ty = ey + r * np.sin(alpha + theta2_rel)
    return np.array([ex, ey]), np.array([tx, ty])

def ik_alpha_beta(base, target, r=R):
    """
    IK branches, used only to get a desired α azimuth for alpha-first bias.
    """
    bx, by = base; tx, ty = target
    dx, dy = tx - bx, ty - by
    rho2 = dx*dx + dy*dy
    rho = np.sqrt(rho2) + 1e-12
    rho = np.clip(rho, 1e-9, 2*r - 1e-9)
    c2 = (rho2 - 2*r*r) / (2*r*r)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2*c2))
    theta2_rel_a = np.arctan2(+s2, c2)
    theta2_rel_b = np.arctan2(-s2, c2)
    phi = np.arctan2(dy, dx)
    k1 = r * (1 + c2); k2 = r * s2
    alpha_a = (phi - np.arctan2(k2, k1)) % (2*np.pi)
    alpha_b = (phi - np.arctan2(-k2, k1)) % (2*np.pi)
    beta_a = np.clip(theta2_rel_a + np.pi, BETA_MIN, BETA_MAX)
    beta_b = np.clip(theta2_rel_b + np.pi, BETA_MIN, BETA_MAX)
    return (alpha_a, beta_a), (alpha_b, beta_b)

def ang_diff_signed(a, b):
    """Smallest signed angle from a -> b in (-π, π]."""
    d = (b - a + np.pi) % (2*np.pi) - np.pi
    return d

# =========================
# Collision utilities
# =========================
def dist_point_segment(p, a, b):
    ap = p - a; ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def step_collides(i, elbow_i, tip_i, bases, elbows, tips, capsule_r=CAPSULE_R, board=BOARD_SIZE_UNITS):
    if not (0 <= tip_i[0] <= board and 0 <= tip_i[1] <= board): return True
    if not (0 <= elbow_i[0] <= board and 0 <= elbow_i[1] <= board): return True
    for j in range(len(bases)):
        if j == i: 
            continue
        if np.linalg.norm(tip_i - tips[j]) < 2*capsule_r:      return True
        if np.linalg.norm(tip_i - elbows[j]) < 2*capsule_r:    return True
        if dist_point_segment(tip_i, bases[j],  elbows[j]) < 2*capsule_r: return True
        if dist_point_segment(tip_i, elbows[j], tips[j])  < 2*capsule_r:  return True
    return False

# =========================
# Potential fields (Gaussian via FFT on cell grid)
# =========================
sigma_agents_cells  = SIGMA_AGENTS_UNITS  / CELL
sigma_targets_cells = SIGMA_TARGETS_UNITS / CELL
xf = np.fft.fftfreq(k) * k
Xf, Yf = np.meshgrid(xf, xf, indexing='ij')
GAUSS_AGENTS_F  = np.fft.fft2(np.exp(-(Xf**2 + Yf**2) / (2 * sigma_agents_cells**2)))
GAUSS_TARGETS_F = np.fft.fft2(np.exp(-(Xf**2 + Yf**2) / (2 * sigma_targets_cells**2)))

def points_to_occ_grid(points_xy):
    if len(points_xy) == 0:
        return np.zeros((k, k), dtype=float)
    ii = np.clip((points_xy[:,0] / CELL).astype(int), 0, k-1)
    jj = np.clip((points_xy[:,1] / CELL).astype(int), 0, k-1)
    occ = np.zeros((k, k), dtype=float)
    occ[ii, jj] += 1.0
    return occ

def convolve_gaussian_fft(occ, GAUSS_F):
    if occ is None or occ.size == 0:
        return np.zeros((k, k), dtype=float)
    return np.fft.ifft2(np.fft.fft2(occ) * GAUSS_F).real

def sample_field(field, p_xy):
    i = int(np.clip(p_xy[0] / CELL, 0, k - 1))
    j = int(np.clip(p_xy[1] / CELL, 0, k - 1))
    return field[i, j]

# =========================
# State init
# =========================
alphabeta = np.zeros((N, 2), dtype=float)  # [alpha, beta]
elbows = np.zeros_like(bases)
tips   = np.zeros_like(bases)
for i in range(N):
    elbows[i], tips[i] = fk_alpha_beta(bases[i], alphabeta[i,0], alphabeta[i,1], r=R)
reached = np.zeros(N, dtype=bool)

# Remember last chosen action for momentum/anti-reversal
last_action = np.zeros((N, 2), dtype=float)  # [Δα_last, Δβ_last]

# Build action set (Δα, Δβ)
alpha_steps = [-ALPHA_STEP, 0.0, +ALPHA_STEP] if ALLOW_ALPHA_CW else [0.0, +ALPHA_STEP]
beta_steps  = [-BETA_STEP,  0.0, +BETA_STEP]  if ALLOW_BETA_DECREASE else [0.0, +BETA_STEP]
ACTIONS = [(da, db) for da in alpha_steps for db in beta_steps
           if INCLUDE_HOLD or abs(da) > 1e-12 or abs(db) > 1e-12]

# =========================
# Simulation loop (tip-centric with anti-oscillation)
# =========================
fig, ax = plt.subplots(figsize=(8, 8))

for step in range(NUM_STEPS):
    # Potentials
    occ_agents  = points_to_occ_grid(tips)
    phi_agents  = LAMBDA_AGENTS  * convolve_gaussian_fft(occ_agents, GAUSS_AGENTS_F)
    reached_targets = targets[reached]
    if len(reached_targets) > 0:
        occ_reached = points_to_occ_grid(reached_targets)
        phi_reached = LAMBDA_REACHED * convolve_gaussian_fft(occ_reached, GAUSS_TARGETS_F)
    else:
        phi_reached = np.zeros((k, k), dtype=float)

    def instantaneous_cost(i, tip_old, tip_new, a_cand, da, db, target):
        # potentials at new tip
        c = sample_field(phi_agents, tip_new) + sample_field(phi_reached, tip_new)
        # attraction (distance to own target)
        d_new = np.linalg.norm(tip_new - target)
        d_old = np.linalg.norm(tip_old - target)
        c += W_ATTRACT * d_new
        # progress bonus (prefer d_new < d_old)
        c += W_PROGRESS * (d_new - d_old)  # negative if progress
        # soft alpha-first bias if α far from desired azimuth and we're changing β
        if W_ALPHA_FIRST > 0 and abs(db) > 1e-12:
            sols = ik_alpha_beta(bases[i], target, r=R)
            a_des = sols[0][0]  # just an azimuth reference
            mis = abs(ang_diff_signed(a_cand, a_des))
            if mis > ALPHA_ALIGN_TOL:
                c += W_ALPHA_FIRST
        # small step cost + normalized magnitude (discourages dithering)
        c += W_STEP + W_STEP_NORM * ((abs(da)/ALPHA_STEP) + (abs(db)/BETA_STEP))
        # anti-reversal
        da_last, db_last = last_action[i]
        if da_last != 0 and np.sign(da_last) == -np.sign(da): c += W_REVERSE_A
        if db_last != 0 and np.sign(db_last) == -np.sign(db): c += W_REVERSE_B
        # HOLD bias (if hold chosen)
        if abs(da) < 1e-12 and abs(db) < 1e-12:
            c += W_HOLD
        return c

    # viz
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, BOARD_SIZE_UNITS)
    ax.set_ylim(0, BOARD_SIZE_UNITS)
    ax.set_title(f"SCARA (tip-as-agent, anti-oscillation) — Step {step+1}")
    ax.scatter(targets[:,0], targets[:,1], c='red', marker='*', s=90, label='Targets')

    order = np.arange(N)
    rng.shuffle(order)

    for i in order:
        if reached[i]:
            continue
        base   = bases[i]
        target = targets[i]
        a0, b0 = alphabeta[i]
        elbow0, tip0 = elbows[i], tips[i]

        best = None
        for (da, db) in ACTIONS:
            a_cand = (a0 + da) % (2*np.pi)
            b_cand = np.clip(b0 + db, BETA_MIN, BETA_MAX)
            elbow_c, tip_c = fk_alpha_beta(base, a_cand, b_cand, r=R)

            # skip colliding actions
            if step_collides(i, elbow_c, tip_c, bases, elbows, tips, capsule_r=CAPSULE_R, board=BOARD_SIZE_UNITS):
                continue

            c = instantaneous_cost(i, tip0, tip_c, a_cand, da, db, target)
            if (best is None) or (c < best[0]):
                best = (c, da, db, a_cand, b_cand, elbow_c, tip_c)

        if best is not None:
            _, da_sel, db_sel, a_sel, b_sel, elbow_sel, tip_sel = best
            alphabeta[i,0], alphabeta[i,1] = a_sel, b_sel
            elbows[i], tips[i] = elbow_sel, tip_sel
            last_action[i] = np.array([da_sel, db_sel])
            if np.linalg.norm(tip_sel - target) < REACH_TOL:
                reached[i] = True
        # else: no safe action this frame → wait (keeps last_action as-is)

    # draw robots
    for i in range(N):
        base = bases[i]; elbow = elbows[i]; tip = tips[i]
        color = 'g-' if reached[i] else 'c-'
        lw = 1.6 if reached[i] else 1.2
        ax.plot([base[0], elbow[0]], [base[1], elbow[1]], color, lw=lw)
        ax.plot([elbow[0], tip[0]],  [elbow[1], tip[1]],  color, lw=lw)
        ax.add_patch(plt.Circle(base,  LINK_RAD, color='gray'))
        ax.add_patch(plt.Circle(elbow, LINK_RAD, color='skyblue'))
        ax.add_patch(plt.Circle(tip,   LINK_RAD, color='cyan' if not reached[i] else 'lime'))

    if np.any(reached):
        ax.scatter(tips[reached,0], tips[reached,1], c='lime', s=28, label='Reached')

    ax.legend(loc='upper right')
    plt.pause(PAUSE)
    if np.all(reached):
        print(f"✅ All targets reached by step {step+1}")
        break

plt.show()
