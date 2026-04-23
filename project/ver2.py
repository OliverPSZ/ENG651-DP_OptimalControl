import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

"""
Adaptation of your simple grid DP into an alpha/beta-grid value-iteration planner
that matches the 63-positioner layout and the discrete joint stepping you use
(ALPHA_STEP, BETA_STEP). It plans in joint-space (α, β) and converts to tip paths.

Key features:
- 63 bases arranged in the same centered pyramid layout you used.
- Joint limits and discrete steps: α wraps in [0, 2π), β ∈ [0, π].
- Actions: ±Δα, ±Δβ, and optionally HOLD.
- Terminal set: all (α,β) whose tip is within REACH_TOL (units) of the target.
- Value Iteration solves a deterministic shortest-path in (α,β)-grid to the terminal set.
- Extracts a greedy path from (α0,β0) to target (returns a sequence of (α,β)).

Hooks to extend:
- Collision penalties/forbidden moves (add large cost or mask transitions).
- Multi-robot coupling (iterate best responses, plan batches, etc.).

Usage:
- See the __main__ block for a demo on all 63 robots with random reachable targets.
- Integrate the returned alpha/beta sequences with your existing animation/export.
"""

# =========================
# Parameters (tuning & geometry)
# =========================
CELL = 0.2
SPACING_UNITS = 6.2
S_CELLS = int(round(SPACING_UNITS / CELL))

# SCARA geometry & motion
R = 1.8  # link length (alpha and beta)
ALPHA_STEP = np.deg2rad(2)  # α step magnitude
BETA_STEP = np.deg2rad(2)   # β step magnitude
REACH_TOL = 0.10            # tip-to-target tolerance (units)

# Movement freedom
ALLOW_ALPHA_CW = True       # allow clockwise α (negative step)
ALLOW_BETA_DECREASE = True  # allow β decrease (still clamped to [0,π])

# Joint limits (β uses: 0=retracted, π=extended)
ALPHA_MIN, ALPHA_MAX = 0.0, 2*np.pi
BETA_MIN, BETA_MAX = 0.0, np.pi

# Display/sim convenience
RNG_SEED = 7

# =========================
# 63-base centered pyramid layout (same as your previous code)
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
N = len(bases)  # should be 63

# =========================
# Kinematics (α, β with β=0=retracted, β=π=extended)
# =========================

def fk_alpha_beta(base: np.ndarray, alpha: float, beta: float, r: float = R) -> Tuple[np.ndarray, np.ndarray]:
    """Forward kinematics; returns (elbow, tip)."""
    theta2_rel = beta - np.pi
    bx, by = base
    ex = bx + r * np.cos(alpha)
    ey = by + r * np.sin(alpha)
    tx = ex + r * np.cos(alpha + theta2_rel)
    ty = ey + r * np.sin(alpha + theta2_rel)
    return np.array([ex, ey]), np.array([tx, ty])


# =========================
# Alpha/Beta grid and actions
# =========================

def make_alpha_beta_axes(alpha_step=ALPHA_STEP, beta_step=BETA_STEP):
    n_alpha = int(np.round((ALPHA_MAX - ALPHA_MIN) / alpha_step))
    n_beta  = int(np.round((BETA_MAX - BETA_MIN) / beta_step)) + 1  # include both ends
    alpha_vals = (ALPHA_MIN + np.arange(n_alpha) * alpha_step) % (2*np.pi)
    beta_vals  = BETA_MIN + np.arange(n_beta) * beta_step
    beta_vals[-1] = BETA_MAX  # ensure exact π
    return alpha_vals, beta_vals


def build_actions(include_hold: bool = True,
                  allow_alpha_cw: bool = ALLOW_ALPHA_CW,
                  allow_beta_decr: bool = ALLOW_BETA_DECREASE):
    a_steps = [0, +1]
    b_steps = [0, +1]
    if allow_alpha_cw:
        a_steps = [-1, 0, +1]
    if allow_beta_decr:
        b_steps = [-1, 0, +1]
    actions = []  # (dai, dbi) index steps
    for dai in a_steps:
        for dbi in b_steps:
            if not include_hold and (dai == 0 and dbi == 0):
                continue
            actions.append((dai, dbi))
    return actions


# =========================
# Planner
# =========================
@dataclass
class PlanOptions:
    include_hold: bool = True
    move_cost: float = 1.0
    hold_cost: float = 1.0  # only applies if include_hold=True
    max_iters: int = 2000
    eps: float = 1e-8


class AlphaBetaPlanner:
    def __init__(self,
                 alpha_step: float = ALPHA_STEP,
                 beta_step: float = BETA_STEP,
                 reach_tol: float = REACH_TOL):
        self.alpha_step = alpha_step
        self.beta_step = beta_step
        self.reach_tol = reach_tol
        self.alpha_vals, self.beta_vals = make_alpha_beta_axes(alpha_step, beta_step)
        self.Na = len(self.alpha_vals)
        self.Nb = len(self.beta_vals)

    def _neighbors_indices(self, dai: int, dbi: int, a_idx_grid: np.ndarray, b_idx_grid: np.ndarray):
        # α wraps, β clamps
        a_next = (a_idx_grid + dai) % self.Na
        b_next = np.clip(b_idx_grid + dbi, 0, self.Nb - 1)
        return a_next, b_next

    def plan_single(self,
                    base: np.ndarray,
                    target: np.ndarray,
                    opts: PlanOptions = PlanOptions(),
                    start_alpha: float = 0.0,
                    start_beta: float = 0.0,
                    forbid_mask: Optional[np.ndarray] = None,
                    extra_step_cost: Optional[np.ndarray] = None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        """
        Value-iteration on (α,β)-grid for one robot.

        Returns:
            J:        (Na,Nb) optimal cost-to-go
            policy:   (Na,Nb,2) best (dai, dbi) at each state (int32)
            tips:     (Na,Nb,2) tip positions for visualization
            terminal: (Na,Nb) boolean mask of terminal states
            path_ab:  list of (alpha, beta) from (start_alpha, start_beta) to terminal
        """
        actions = build_actions(include_hold=opts.include_hold)
        # Grids of indices
        a_idx = np.arange(self.Na)[:, None]
        b_idx = np.arange(self.Nb)[None, :]

        # Precompute tips for all (α,β)
        tips = np.zeros((self.Na, self.Nb, 2), dtype=float)
        for ia, a in enumerate(self.alpha_vals):
            # vectorized over beta
            beta_vec = self.beta_vals  # (Nb,)
            _, tip_vec = fk_alpha_beta(base, a, beta_vec[:, None])  # fk for array beta? We'll loop to be safe
        # safer explicit loop over β for clarity
        for ia, a in enumerate(self.alpha_vals):
            for ib, b in enumerate(self.beta_vals):
                _, t = fk_alpha_beta(base, a, b)
                tips[ia, ib] = t

        # Terminal set: within reach tolerance
        term = np.linalg.norm(tips - target[None, None, :], axis=-1) <= self.reach_tol

        # Forbidden mask (e.g., collisions); False = free, True = forbidden
        if forbid_mask is None:
            forbid_mask = np.zeros((self.Na, self.Nb), dtype=bool)

        # Initial cost J
        BIG = 1e9
        J = np.full((self.Na, self.Nb), BIG, dtype=float)
        J[term] = 0.0
        J[forbid_mask] = BIG

        # Optional per-state step surcharge (e.g., potentials), must be >=0
        if extra_step_cost is None:
            extra_step_cost = np.zeros_like(J)

        # Value Iteration
        policy = np.zeros((self.Na, self.Nb, 2), dtype=np.int32)
        for it in range(opts.max_iters):
            J_old = J.copy()
            # Compute candidate costs for each action
            best_cost = np.full_like(J, BIG)
            best_act = np.zeros((self.Na, self.Nb, 2), dtype=np.int32)

            for (dai, dbi) in actions:
                a_next, b_next = self._neighbors_indices(dai, dbi, a_idx, b_idx)
                step_c = opts.move_cost if (dai != 0 or dbi != 0) else opts.hold_cost
                cand = step_c + extra_step_cost + J_old[a_next, b_next]
                # Keep terminals at 0 and forbid masked as BIG
                cand = np.where(term, 0.0, cand)
                cand = np.where(forbid_mask, BIG, cand)
                improve = cand < best_cost
                best_act[improve, 0] = dai
                best_act[improve, 1] = dbi
                best_cost = np.where(improve, cand, best_cost)

            J = best_cost
            policy = best_act

            # Convergence check (excluding terminals and forbids)
            mask = ~(term | forbid_mask)
            if np.linalg.norm(J[mask] - J_old[mask]) <= opts.eps:
                break

        # Extract path from start (snap start to nearest grid indices)
        a0 = int(np.round((start_alpha - ALPHA_MIN) / self.alpha_step)) % self.Na
        b0 = int(np.round((start_beta  - BETA_MIN ) / self.beta_step))
        b0 = int(np.clip(b0, 0, self.Nb - 1))

        path = [(self.alpha_vals[a0], self.beta_vals[b0])]
        MAX_PATH = 20000  # safeguard
        a_i, b_i = a0, b0
        for _ in range(MAX_PATH):
            if term[a_i, b_i]:
                break
            dai, dbi = policy[a_i, b_i]
            a_i = (a_i + int(dai)) % self.Na
            b_i = int(np.clip(b_i + int(dbi), 0, self.Nb - 1))
            path.append((self.alpha_vals[a_i], self.beta_vals[b_i]))
            if len(path) > 3 and path[-1] == path[-3]:
                # tiny cycle guard (shouldn't happen with positive costs)
                break

        return J, policy, tips, term, path


# =========================
# Helpers
# =========================

def random_target_near_base(base, r=R, rng=None, r_min=0.5, margin=0.2):
    """Sample a reachable random target in the annulus [r_min*r, 2*r - margin]."""
    if rng is None:
        rng = np.random.default_rng()
    rho = rng.uniform(r_min*r, 2.0*r - margin)
    ang = rng.uniform(0, 2*np.pi)
    return base + rho * np.array([np.cos(ang), np.sin(ang)])


def path_to_arrays(path: List[Tuple[float, float]]):
    a = np.array([p[0] for p in path], dtype=float)
    b = np.array([p[1] for p in path], dtype=float)
    return a, b


# =========================
# Demo (plans for all 63 robots)
# =========================
if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    planner = AlphaBetaPlanner(ALPHA_STEP, BETA_STEP, REACH_TOL)
    opts = PlanOptions(include_hold=True, move_cost=1.0, hold_cost=1.0, max_iters=2000, eps=1e-7)

    # Build random but reachable targets per base (replace with your real targets)
    targets = np.array([random_target_near_base(b, r=R, rng=rng) for b in bases])

    # Plan for each robot independently (no coupling/collisions yet)
    all_paths_ab: List[List[Tuple[float, float]]] = []
    for i in range(N):
        J, pi, tips, term, path = planner.plan_single(
            base=bases[i],
            target=targets[i],
            opts=opts,
            start_alpha=0.0,
            start_beta=0.0,
            forbid_mask=None,          # <-- plug collision mask here if desired
            extra_step_cost=None,      # <-- plug potentials/penalties here if desired
        )
        all_paths_ab.append(path)
        print(f"Robot {i+1:02d}: path length {len(path)} states; terminal reached: {term[int(np.round(0.0/ALPHA_STEP))%planner.Na, 0] or (len(path)>0 and term[np.argmin(np.abs(planner.alpha_vals-path[-1][0])), np.argmin(np.abs(planner.beta_vals-path[-1][1]))])}")

    # Convert one example path to arrays for your animator
    a_seq, b_seq = path_to_arrays(all_paths_ab[0])
    print(f"Example robot 1: {len(a_seq)} alpha steps, {len(b_seq)} beta steps.")

    # NOTE:
    # - For coupling/collision-aware planning, compute a forbid_mask per robot per iteration.
    #   For example, rasterize other robots' elbow/tip capsules at their currently planned steps
    #   into (α,β) states whose tips would collide, and set forbid_mask=True there.
    # - You can add soft penalties via `extra_step_cost` to bias flows (e.g., keep-away fields).
