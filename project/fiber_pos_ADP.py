import numpy as np
import matplotlib.pyplot as plt

# ========== Parameters ==========
CELL = 0.2
SPACING_UNITS = 6.2
S_CELLS = int(round(SPACING_UNITS / CELL))
R = 1.8
ALPHA_STEP = np.deg2rad(2)
BETA_STEP = np.deg2rad(2)
REACH_TOL = 0.10
ALPHA_MIN, ALPHA_MAX = 0.0, 2*np.pi
BETA_MIN, BETA_MAX = 0.0, np.pi
LINK_RAD = 0.25
SAFETY_PAD = 0.05
CAPSULE_R = LINK_RAD + SAFETY_PAD
MAX_TRIES = 10
ALPHA_DETOUR = np.deg2rad(6)
BETA_DETOUR = np.deg2rad(2)
PAUSE = 0.04
NUM_STEPS = 900
W_DIST = 1.0
W_DENSITY = 0.6
W_BETA = 0.4
W_STALL = 0.5
STEP_MAG_WEIGHT = 0.05
NOISE_ALPHA_STD = np.deg2rad(0.5)
NOISE_BETA_STD = np.deg2rad(0.5)

# ========== Grid Layout ==========
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
rng = np.random.default_rng(7)

def random_target_near_base(base, r=R, margin=0.2):
    rho = rng.uniform(0.5*r, 2.0*r - margin)
    ang = rng.uniform(0, 2*np.pi)
    return base + rho * np.array([np.cos(ang), np.sin(ang)])

targets = np.array([random_target_near_base(b) for b in bases])

# ========== Kinematics ==========
def fk_alpha_beta(base, alpha, beta, r=R):
    theta2_rel = beta - np.pi
    bx, by = base
    ex = bx + r * np.cos(alpha)
    ey = by + r * np.sin(alpha)
    tx = ex + r * np.cos(alpha + theta2_rel)
    ty = ey + r * np.sin(alpha + theta2_rel)
    return np.array([ex, ey]), np.array([tx, ty])

def ik_alpha_beta(base, target, r=R):
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

def ccw_delta_wrap(cur, des): return (des - cur) % (2*np.pi)
def ccw_delta_beta(cur, des): return des - cur if des >= cur else np.inf

def choose_solution_ccw_pref(alpha_cur, beta_cur, sols):
    (a1, b1), (a2, b2) = sols
    s1 = ccw_delta_wrap(alpha_cur, a1) + ccw_delta_beta(beta_cur, b1)
    s2 = ccw_delta_wrap(alpha_cur, a2) + ccw_delta_beta(beta_cur, b2)
    if np.isinf(s1) and np.isinf(s2):
        return (a1, b1) if abs(b1 - beta_cur) <= abs(b2 - beta_cur) else (a2, b2)
    return (a1, b1) if s1 <= s2 else (a2, b2)

def step_alpha_ccw(alpha_cur, alpha_des, step=ALPHA_STEP):
    return (alpha_cur + min(ccw_delta_wrap(alpha_cur, alpha_des), step)) % (2*np.pi)

def step_beta_ccw(beta_cur, beta_des, step=BETA_STEP):
    return beta_cur if beta_des <= beta_cur else min(beta_cur + step, beta_des, BETA_MAX)

def dist_point_segment(p, a, b):
    ap = p - a; ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def step_collides(i, elbow_i, tip_i, bases, elbows, tips, capsule_r=CAPSULE_R):
    base_i = bases[i]
    if not (0 <= tip_i[0] <= BOARD_SIZE_UNITS and 0 <= tip_i[1] <= BOARD_SIZE_UNITS):
        return True
    for j in range(len(bases)):
        if j == i: continue
        if np.linalg.norm(tip_i - tips[j]) < 2*capsule_r: return True
        if np.linalg.norm(tip_i - elbows[j]) < 2*capsule_r: return True
        if dist_point_segment(tip_i, bases[j], elbows[j]) < 2*capsule_r: return True
        if dist_point_segment(tip_i, elbows[j], tips[j]) < 2*capsule_r: return True
    if not (0 <= elbow_i[0] <= BOARD_SIZE_UNITS and 0 <= elbow_i[1] <= BOARD_SIZE_UNITS):
        return True
    return False

alphabeta = np.zeros((N, 2))
elbows = np.zeros_like(bases)
tips = np.zeros_like(bases)
for i in range(N):
    elbows[i], tips[i] = fk_alpha_beta(bases[i], 0, 0)
reached = np.zeros(N, dtype=bool)
prev_distances = np.full(N, np.inf)

def estimate_local_density(i, tips, radius=1.5):
    p = tips[i]
    others = np.delete(tips, i, axis=0)
    return np.sum(np.linalg.norm(others - p, axis=1) < radius)

fig, ax = plt.subplots(figsize=(8, 8))
for step in range(NUM_STEPS):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, BOARD_SIZE_UNITS)
    ax.set_ylim(0, BOARD_SIZE_UNITS)
    ax.set_title(f"ADP — Step {step+1}")
    ax.scatter(targets[:,0], targets[:,1], c='red', marker='*', s=90, label='Targets')

    order = np.arange(N)
    rng.shuffle(order)
    for i in order:
        if reached[i]: continue
        base = bases[i]
        sols = ik_alpha_beta(base, targets[i], r=R)
        a_des, b_des = choose_solution_ccw_pref(alphabeta[i,0], alphabeta[i,1], sols)
        a_try = step_alpha_ccw(alphabeta[i,0], a_des)
        b_try = step_beta_ccw (alphabeta[i,1], b_des)

        best_cost = np.inf
        best_config = None

        for attempt in range(MAX_TRIES+1):
            a_cand = (a_try + ALPHA_DETOUR * attempt + rng.normal(0, NOISE_ALPHA_STD)) % (2*np.pi)
            b_cand = min(b_try + BETA_DETOUR * attempt + rng.normal(0, NOISE_BETA_STD), BETA_MAX)
            elbow_c, tip_c = fk_alpha_beta(base, a_cand, b_cand, r=R)
            if step_collides(i, elbow_c, tip_c, bases, elbows, tips, capsule_r=CAPSULE_R): continue
            dist = np.linalg.norm(tip_c - targets[i])
            density = estimate_local_density(i, np.concatenate([tips[:i], [tip_c], tips[i+1:]]))
            beta_effort = b_cand / np.pi
            delta_dist = dist - prev_distances[i]
            step_size = np.linalg.norm(tip_c - tips[i])
            cost = W_DIST*dist + W_DENSITY*density + W_BETA*beta_effort + W_STALL*delta_dist - STEP_MAG_WEIGHT*step_size
            if cost < best_cost:
                best_cost = cost
                best_config = (a_cand, b_cand, elbow_c, tip_c, dist)

        if best_config:
            a_sel, b_sel, elbow_sel, tip_sel, new_dist = best_config
            alphabeta[i] = [a_sel, b_sel]
            elbows[i], tips[i] = elbow_sel, tip_sel
            prev_distances[i] = new_dist
            if new_dist < REACH_TOL:
                reached[i] = True

    for i in range(N):
        base, elbow, tip = bases[i], elbows[i], tips[i]
        color = 'g-' if reached[i] else 'c-'
        ax.plot([base[0], elbow[0]], [base[1], elbow[1]], color, lw=1.6)
        ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color, lw=1.6)
        ax.add_patch(plt.Circle(base, LINK_RAD, color='gray'))
        ax.add_patch(plt.Circle(elbow, LINK_RAD, color='skyblue'))
        ax.add_patch(plt.Circle(tip, LINK_RAD, color='lime' if reached[i] else 'cyan'))

    if np.any(reached):
        ax.scatter(tips[reached,0], tips[reached,1], c='lime', s=28, label='Reached')
    ax.legend(loc='upper right')
    plt.pause(PAUSE)
    if np.all(reached):
        print(f"✅ All targets reached by step {step+1}")
        break

plt.show()
