import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters (Tuning)
# =========================
CELL = 0.2
SPACING_UNITS = 6.2
S_CELLS = int(round(SPACING_UNITS / CELL))

R = 1.8  # link length (alpha and beta)
ALPHA_STEP = np.deg2rad(2)  # α step (CCW)
BETA_STEP = np.deg2rad(2)   # β step (CCW = increase-only)
REACH_TOL = 0.10            # tip-to-target tolerance (units)

# Joint limits and conventions
ALPHA_MIN, ALPHA_MAX = 0.0, 2*np.pi
BETA_MIN, BETA_MAX = 0.0, np.pi  # β=0 retracted, β=π extended

# Collision geometry (capsule approx)
LINK_RAD = 0.25  # thickness radius for links & joints
SAFETY_PAD = 0.05  # extra padding
CAPSULE_R = LINK_RAD + SAFETY_PAD  # effective collision radius
MAX_TRIES = 10  # detour attempts when a move collides
ALPHA_DETOUR = np.deg2rad(6)  # extra CCW on α when blocked
BETA_DETOUR = np.deg2rad(2)   # small extra CCW on β when blocked

# Display / sim
PAUSE = 0.04
NUM_STEPS = 900

# =========================
# Base layout (63 bases in a centered pyramid)
# =========================
COL_COUNTS = [9, 10, 9, 8, 7, 6, 5, 4, 3, 2]
C = len(COL_COUNTS); H = max(COL_COUNTS)
MARGIN = 2 * S_CELLS
k = int(max((C - 1)*S_CELLS + 1 + 2*MARGIN, (H - 1)*S_CELLS + 1 + 2*MARGIN))
if k % 2 == 0:
    k += 1
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
rng = np.random.default_rng(7)

def random_target_near_base(base, r=R, margin=0.2):
    rho = rng.uniform(0.5*r, 2.0*r - margin)  # ensure reachable
    ang = rng.uniform(0, 2*np.pi)
    return base + rho * np.array([np.cos(ang), np.sin(ang)])

targets = np.array([random_target_near_base(b) for b in bases])

# =========================
# Kinematics (α, β with β=0 retracted, β=π extended)
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
    """ Two classic 2-link IK branches -> converted to (α, β) with β in [0, π]. """
    bx, by = base; tx, ty = target
    dx, dy = tx - bx, ty - by
    rho2 = dx*dx + dy*dy
    rho = np.sqrt(rho2) + 1e-12
    rho = np.clip(rho, 1e-9, 2*r - 1e-9)
    c2 = (rho2 - 2*r*r) / (2*r*r)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2*c2))
    theta2_rel_a = np.arctan2(+s2, c2)  # elbow-up
    theta2_rel_b = np.arctan2(-s2, c2)  # elbow-down
    phi = np.arctan2(dy, dx)
    k = r * (1 + c2); k2 = r * s2
    alpha_a = (phi - np.arctan2(k2, k)) % (2*np.pi)
    alpha_b = (phi - np.arctan2(-k2, k)) % (2*np.pi)
    beta_a = np.clip(theta2_rel_a + np.pi, BETA_MIN, BETA_MAX)
    beta_b = np.clip(theta2_rel_b + np.pi, BETA_MIN, BETA_MAX)
    return (alpha_a, beta_a), (alpha_b, beta_b)

# =========================
# CCW-only stepping & choice
# =========================
def ccw_delta_wrap(cur, des):
    """CCW distance on circle for α."""
    return (des - cur) % (2*np.pi)

def ccw_delta_beta(cur, des):
    """CCW-only distance on [0, π] for β (increase only)."""
    d = des - cur
    return d if d >= 0 else np.inf

def choose_solution_ccw_pref(alpha_cur, beta_cur, sols):
    (a1, b1), (a2, b2) = sols
    s1 = ccw_delta_wrap(alpha_cur, a1) + ccw_delta_beta(beta_cur, b1)
    s2 = ccw_delta_wrap(alpha_cur, a2) + ccw_delta_beta(beta_cur, b2)
    if np.isinf(s1) and np.isinf(s2):
        # both want β to decrease; pick closer β so α can rotate until β becomes feasible
        return (a1, b1) if abs(b1 - beta_cur) <= abs(b2 - beta_cur) else (a2, b2)
    return (a1, b1) if s1 <= s2 else (a2, b2)

def step_alpha_ccw(alpha_cur, alpha_des, step=ALPHA_STEP):
    return (alpha_cur + min(ccw_delta_wrap(alpha_cur, alpha_des), step)) % (2*np.pi)

def step_beta_ccw(beta_cur, beta_des, step=BETA_STEP):
    if beta_des <= beta_cur:  # cannot decrease
        return beta_cur
    return min(beta_cur + step, beta_des, BETA_MAX)

# =========================
# Collision utilities
# =========================
def dist_point_segment(p, a, b):
    """Distance from point p to segment ab."""
    ap = p - a; ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def step_collides(i, elbow_i, tip_i, bases, elbows, tips, capsule_r=CAPSULE_R):
    """
    Check if agent i's new elbow/tip collide with any other agent's links/joints.
    We check:
    - tip_i vs other tips (circles)
    - tip_i vs other elbows (circles)
    - tip_i vs other segments (base->elbow and elbow->tip) (capsules)
    """
    # own base for segments
    base_i = bases[i]

    # Early: keep within board
    if not (0 <= tip_i[0] <= BOARD_SIZE_UNITS and 0 <= tip_i[1] <= BOARD_SIZE_UNITS):
        return True

    for j in range(len(bases)):
        if j == i:
            continue
        # circles
        if np.linalg.norm(tip_i - tips[j]) < 2*capsule_r:  # tip-tip
            return True
        if np.linalg.norm(tip_i - elbows[j]) < 2*capsule_r:  # tip-elbow
            return True
        # capsules: point-to-segment distance
        if dist_point_segment(tip_i, bases[j], elbows[j]) < 2*capsule_r:
            return True
        if dist_point_segment(tip_i, elbows[j], tips[j]) < 2*capsule_r:
            return True

    # Also ensure our own segments don't go outside board (soft check)
    if not (0 <= elbow_i[0] <= BOARD_SIZE_UNITS and 0 <= elbow_i[1] <= BOARD_SIZE_UNITS):
        return True

    return False

# =========================
# State init
# =========================
alphabeta = np.zeros((N, 2), dtype=float)  # [alpha, beta]
elbows = np.zeros_like(bases)
tips = np.zeros_like(bases)
for i in range(N):
    elbows[i], tips[i] = fk_alpha_beta(bases[i], alphabeta[i,0], alphabeta[i,1], r=R)
reached = np.zeros(N, dtype=bool)

# =========================
# Simulation loop
# =========================
fig, ax = plt.subplots(figsize=(8, 8))
for step in range(NUM_STEPS):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, BOARD_SIZE_UNITS)
    ax.set_ylim(0, BOARD_SIZE_UNITS)
    ax.set_title(f"SCARA (collision-aware) — Step {step+1}")

    # Draw targets
    ax.scatter(targets[:,0], targets[:,1], c='red', marker='*', s=90, label='Targets')

    # Randomize update order slightly to reduce symmetry deadlocks
    order = np.arange(N)
    rng.shuffle(order)

    for i in order:
        base = bases[i]
        if reached[i]:
            continue

        # Desired IK (CCW-preferred branch)
        sols = ik_alpha_beta(base, targets[i], r=R)
        a_des, b_des = choose_solution_ccw_pref(alphabeta[i,0], alphabeta[i,1], sols)

        # Propose nominal step
        a_try = step_alpha_ccw(alphabeta[i,0], a_des, ALPHA_STEP)
        b_try = step_beta_ccw (alphabeta[i,1], b_des, BETA_STEP)

        # Try detours if collision predicted
        success = False
        for attempt in range(MAX_TRIES+1):
            # apply extra CCW detour with attempts
            add_a = ALPHA_DETOUR * attempt
            add_b = min(BETA_DETOUR * attempt, BETA_MAX - b_try)
            a_candidate = (a_try + add_a) % (2*np.pi)
            b_candidate = min(b_try + add_b, BETA_MAX)

            elbow_c, tip_c = fk_alpha_beta(base, a_candidate, b_candidate, r=R)
            if not step_collides(i, elbow_c, tip_c, bases, elbows, tips, capsule_r=CAPSULE_R):
                # commit
                alphabeta[i,0], alphabeta[i,1] = a_candidate, b_candidate
                elbows[i], tips[i] = elbow_c, tip_c
                if np.linalg.norm(tip_c - targets[i]) < REACH_TOL:
                    reached[i] = True
                success = True
                break

        if not success:
            # Skip moving this step (stay put) to avoid collision
            pass

    # Draw all robots
    for i in range(N):
        base = bases[i]; elbow = elbows[i]; tip = tips[i]
        color = 'g-' if reached[i] else 'c-'
        lw = 1.6 if reached[i] else 1.2
        ax.plot([base[0], elbow[0]], [base[1], elbow[1]], color, lw=lw)
        ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color, lw=lw)
        ax.add_patch(plt.Circle(base, LINK_RAD, color='gray'))
        ax.add_patch(plt.Circle(elbow, LINK_RAD, color='skyblue'))
        ax.add_patch(plt.Circle(tip, LINK_RAD, color='cyan' if not reached[i] else 'lime'))

    if np.any(reached):
        ax.scatter(tips[reached,0], tips[reached,1], c='lime', s=28, label='Reached')
    ax.legend(loc='upper right')
    plt.pause(PAUSE)

    if np.all(reached):
        print(f"✅ All targets reached by step {step+1}")
        break

plt.show()
