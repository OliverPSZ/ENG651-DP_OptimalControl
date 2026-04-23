#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# =============================================================================
# Parameters (Tuning) – matches your original conventions
# =============================================================================
CELL = 0.2
SPACING_UNITS = 6.2
S_CELLS = int(round(SPACING_UNITS / CELL))

R = 1.8  # link length (alpha and beta)
ALPHA_STEP = np.deg2rad(2)  # α step (CCW)
BETA_STEP  = np.deg2rad(2)  # β step (CCW = increase-only)
REACH_TOL  = 0.10           # tip-to-target tolerance (units)

# Joint limits and conventions
ALPHA_MIN, ALPHA_MAX = 0.0, 2*np.pi
BETA_MIN,  BETA_MAX  = 0.0, np.pi  # β=0 retracted, β=π extended

# Collision geometry (capsule approx)
LINK_RAD   = 0.25  # thickness radius for links & joints
SAFETY_PAD = 0.05  # extra padding
CAPSULE_R  = LINK_RAD + SAFETY_PAD  # effective collision radius

# =============================================================================
# Base layout (63 bases in a centered pyramid)
# =============================================================================
COL_COUNTS = [9, 10, 9, 8, 7, 6, 5, 4, 3, 2]
C = len(COL_COUNTS); H = max(COL_COUNTS)
MARGIN = 2 * S_CELLS
k_side = int(max((C - 1)*S_CELLS + 1 + 2*MARGIN, (H - 1)*S_CELLS + 1 + 2*MARGIN))
if k_side % 2 == 0:
    k_side += 1
BOARD_SIZE_UNITS = k_side * CELL  # for bounds checks

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

bases = build_bases_pyramid(COL_COUNTS, k_side, S_CELLS, CELL)
N = len(bases)  # should be 63

# =============================================================================
# Targets (reachable annulus around each base)
# =============================================================================
rng = np.random.default_rng(7)

def random_target_near_base(base, r=R, margin=0.2):
    rho = rng.uniform(0.5*r, 2.0*r - margin)  # ensure reachable
    ang = rng.uniform(0, 2*np.pi)
    return base + rho * np.array([np.cos(ang), np.sin(ang)])

targets = np.array([random_target_near_base(b) for b in bases])

# =============================================================================
# Kinematics (α, β with β=0 retracted, β=π extended)
# =============================================================================
def fk_alpha_beta(base, alpha, beta, r=R):
    """Forward kinematics; returns (elbow, tip)."""
    theta2_rel = beta - np.pi
    bx, by = base
    ex = bx + r * np.cos(alpha)
    ey = by + r * np.sin(alpha)
    tx = ex + r * np.cos(alpha + theta2_rel)
    ty = ey + r * np.sin(alpha + theta2_rel)
    return np.array([ex, ey]), np.array([tx, ty])

# =============================================================================
# Collision utilities
# =============================================================================
def dist_point_segment(p, a, b):
    """Distance from point p to segment ab."""
    ap = p - a; ab = b - a
    denom = np.dot(ab, ab) + 1e-12
    t = np.dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# =============================================================================
# DP discretization (indices ↔ angles)
# =============================================================================
K_ALPHA = int(round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP))  # ~ 2π / 2° = 180
L_BETA  = int(round((BETA_MAX  - BETA_MIN ) / BETA_STEP ))  # ~  π / 2° =  90
assert K_ALPHA >= 1 and L_BETA >= 1

def alpha_idx_to_val(k):
    return (k % K_ALPHA) * ALPHA_STEP + ALPHA_MIN

def beta_idx_to_val(l):
    return l * BETA_STEP + BETA_MIN

def clamp_beta_idx(l):
    return max(0, min(L_BETA, l))

START_A_IDX = 0
START_B_IDX = 0

# =============================================================================
# Reservation-aware collision check (against already planned agents at time t)
# =============================================================================
def collides_with_reservation(elbow_i, tip_i, others, capsule_r=CAPSULE_R):
    """
    others: iterable of dicts with keys 'base','elbow','tip' (2D vectors)
    We conservatively check the moving tip_i against:
    - other tips (disks)
    - other elbows (disks)
    - other links (capsules) base->elbow and elbow->tip
    Also checks board bounds and own elbow bounds.
    """
    # bounds
    if not (0 <= tip_i[0]  <= BOARD_SIZE_UNITS and 0 <= tip_i[1]  <= BOARD_SIZE_UNITS):
        return True
    if not (0 <= elbow_i[0] <= BOARD_SIZE_UNITS and 0 <= elbow_i[1] <= BOARD_SIZE_UNITS):
        return True

    for o in others:
        # tip-tip
        if np.linalg.norm(tip_i - o['tip']) < 2*capsule_r:
            return True
        # tip-elbow
        if np.linalg.norm(tip_i - o['elbow']) < 2*capsule_r:
            return True
        # tip vs base->elbow capsule
        if dist_point_segment(tip_i, o['base'],  o['elbow']) < 2*capsule_r:
            return True
        # tip vs elbow->tip capsule
        if dist_point_segment(tip_i, o['elbow'], o['tip'])   < 2*capsule_r:
            return True
    return False

# =============================================================================
# Goal tests and helpers on the discretized grid
# =============================================================================
def is_goal(alpha_idx, beta_idx, base, target, r=R):
    _, tip = fk_alpha_beta(base, alpha_idx_to_val(alpha_idx), beta_idx_to_val(beta_idx), r=r)
    return np.linalg.norm(tip - target) <= REACH_TOL

def elbow_tip(alpha_idx, beta_idx, base, r=R):
    return fk_alpha_beta(base, alpha_idx_to_val(alpha_idx), beta_idx_to_val(beta_idx), r=r)

# =============================================================================
# DP for one agent: BFS on time-expanded DAG (unit costs)
# =============================================================================
def plan_agent_dp(base, target, reservations, T_max,
                  allow_wait=True, verbose=True, progress_every=20000):
    """
    reservations[t] is a list of dicts {'base','elbow','tip'} for already-planned agents at time t.
    Returns:
        path_idx: list of (alpha_idx, beta_idx) for t=0..T_reach
        elbows: (T_reach+1, 2)
        tips:   (T_reach+1, 2)
    Raises:
        RuntimeError if no plan found within horizon.
    """
    start = (START_A_IDX, START_B_IDX, 0)
    MOVES = [(1,0), (0,1), (1,1)]
    if allow_wait:
        MOVES.append((0,0))  # wait

    # start pose check vs reservations at t=0
    e0, x0 = elbow_tip(START_A_IDX, START_B_IDX, base, r=R)
    if collides_with_reservation(e0, x0, reservations[0]):
        if verbose: print("Start state collides with reservations at t=0.", flush=True)
        raise RuntimeError("start-collision")

    if is_goal(START_A_IDX, START_B_IDX, base, target, r=R):
        if verbose: print("Already at goal at t=0.", flush=True)
        return [(START_A_IDX, START_B_IDX)], np.vstack([e0]), np.vstack([x0])

    Q = deque([start])
    seen = {start}
    parent = {}
    goal_node = None
    expansions = 0

    while Q and goal_node is None:
        k, l, t = Q.popleft()
        expansions += 1
        if verbose and (expansions % progress_every == 0):
            print(f"  expanded {expansions:,} nodes (time up to {t})…", flush=True)

        if t >= T_max:
            continue

        for da, db in MOVES:
            k2 = (k + da) % K_ALPHA      # α CCW with wrap
            l2 = clamp_beta_idx(l + db)  # β nondecreasing
            t2 = t + 1
            state2 = (k2, l2, t2)
            if state2 in seen:
                continue

            e2, x2 = elbow_tip(k2, l2, base, r=R)
            if collides_with_reservation(e2, x2, reservations[t2]):
                continue

            parent[state2] = (k, l, t)
            seen.add(state2)
            Q.append(state2)

            if is_goal(k2, l2, base, target, r=R):
                goal_node = state2
                break

    if goal_node is None:
        if verbose: print("No path within T_max.", flush=True)
        raise RuntimeError("no-path")

    # Reconstruct path
    path_rev = []
    node = goal_node
    while node != start:
        path_rev.append((node[0], node[1], node[2]))
        node = parent[node]
    path_rev.append((start[0], start[1], start[2]))
    path_rev.reverse()

    path_idx = [(k, l) for (k, l, t) in path_rev]
    elbows, tips = [], []
    for (k, l) in path_idx:
        e, x = elbow_tip(k, l, base, r=R)
        elbows.append(e); tips.append(x)
    return path_idx, np.vstack(elbows), np.vstack(tips)

# =============================================================================
# All-agents planner: prioritized DP with reservation table
# =============================================================================
def plan_all_agents_dp(bases, targets, T_max=300, rng_seed=7,
                       verbose=True, progress_every=20000):
    """
    Plans all agents sequentially with reservations.
    Returns:
        plans_idx, plans_el, plans_tip, reservations
    """
    N = len(bases)
    rng_local = np.random.default_rng(rng_seed)
    order = np.arange(N)
    rng_local.shuffle(order)

    reservations = [[] for _ in range(T_max + 1)]
    plans_idx  = [None]*N
    plans_el   = [None]*N
    plans_tip  = [None]*N

    if verbose:
        print(f"Planning {N} agents with horizon T_max={T_max}…", flush=True)
        print(f"Priority order: {order.tolist()}", flush=True)

    for count, i in enumerate(order, 1):
        if verbose:
            print(f"\n[{count}/{N}] Planning agent {i} …", flush=True)

        path_idx, elbows, tips = plan_agent_dp(
            bases[i], targets[i], reservations, T_max,
            allow_wait=True, verbose=verbose, progress_every=progress_every
        )

        plans_idx[i] = path_idx
        plans_el[i]  = elbows
        plans_tip[i] = tips

        T_reach = len(path_idx) - 1
        e_last, x_last = elbows[-1], tips[-1]

        # reserve the path geometry
        for t in range(len(path_idx)):
            reservations[t].append({'base': bases[i], 'elbow': elbows[t], 'tip': tips[t]})
        # hold the final pose for the rest of the horizon
        for t in range(len(path_idx), T_max + 1):
            reservations[t].append({'base': bases[i], 'elbow': e_last, 'tip': x_last})

        if verbose:
            print(f"  ✓ agent {i} planned in {T_reach} steps; reservations updated.", flush=True)

    if verbose:
        print("\nAll agents planned successfully.", flush=True)
    return plans_idx, plans_el, plans_tip, reservations

# =============================================================================
# Optional: simple viewer that plays back reservations over time
# =============================================================================
def playback_reservations(reservations, bases, reached_color=True, pause=0.03):
    if not HAS_MPL:
        print("matplotlib not available; skipping viewer.")
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    T_max = len(reservations) - 1
    for t in range(T_max + 1):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(0, BOARD_SIZE_UNITS)
        ax.set_ylim(0, BOARD_SIZE_UNITS)
        ax.set_title(f"DP Planned Trajectories — time {t}")
        # draw agents
        for occ in reservations[t]:
            base = occ['base']; elbow = occ['elbow']; tip = occ['tip']
            ax.plot([base[0], elbow[0]], [base[1], elbow[1]], '-', lw=1.2)
            ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], '-', lw=1.2)
            ax.add_patch(plt.Circle(base,  LINK_RAD, color='gray'))
            ax.add_patch(plt.Circle(elbow, LINK_RAD, color='skyblue'))
            ax.add_patch(plt.Circle(tip,   LINK_RAD, color='lime' if reached_color else 'cyan'))
        plt.pause(pause)
    plt.show()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    try:
        # You can tweak these knobs:
        T_MAX      = 300        # planning horizon (time steps)
        RNG_SEED   = 7          # priority randomization seed
        VERBOSE    = True       # print progress
        HEARTBEAT  = 10000      # BFS "expanded N nodes..." heartbeat
        RUN_VIEWER = False      # set True to playback trajectories (needs matplotlib)

        plans_idx, plans_el, plans_tip, reservations = plan_all_agents_dp(
            bases, targets, T_max=T_MAX, rng_seed=RNG_SEED,
            verbose=VERBOSE, progress_every=HEARTBEAT
        )

        total_steps = sum(len(p)-1 for p in plans_idx if p is not None)
        max_steps   = max((len(p)-1 for p in plans_idx if p is not None), default=0)
        print(f"\nSummary: planned {len([p for p in plans_idx if p is not None])} agents "
              f"(N={N}), total steps={total_steps}, critical path={max_steps} steps.", flush=True)

        if RUN_VIEWER:
            playback_reservations(reservations, bases)

    except Exception as e:
        print("Planner failed with exception:", repr(e), flush=True)
