import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
try:
    import osqp
    import scipy.sparse as sp
    HAS_OSQP = True
except Exception:
    HAS_OSQP = False

# ============================================================
# How to run:
#   Minimal pipeline (CBF_Altering is optional):
#   1) mjpython JointSpace_Altering.py \
#          --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#          --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#          --out_nominal_csv /Users/salamahalmazrouei/Desktop/test_joint_nominal.csv \
#          --ik-seed 0.413164 -0.105627 -0.306975 -2.46688 -0.0954591 2.39035 -2.16711 \
#          --seed_is_initial_pose \
#          --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_altering_plots
#   2) mjpython JointSpace_Check.py \
#          --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#          --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_check_plots
#   3) mjpython JointSpace_Check.py \
#          --csv /Users/salamahalmazrouei/Desktop/test_joint_nominal.csv \
#          --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_check_nominal_plots
#
#   Optional pre-filter (if your reference is noisy/aggressive):
#      python3 CBF_Altering.py \
#          --in_csv /Users/salamahalmazrouei/Desktop/test.csv \
#          --out_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv
#
#   Joint-space alter only (no interactive windows):
#   mjpython JointSpace_Altering.py \
#       --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#       --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#       --ik-seed 0.413164 -0.105627 -0.306975 -2.46688 -0.0954591 2.39035 -2.16711 \
#       --no_plot
#
#   Optional full safety check in task-space:
#   python3 Check_Safety.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv
#
#   With seed and explicit bounds/limits:
#   mjpython JointSpace_Altering.py \
#       --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#       --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#       --out_nominal_csv /Users/salamahalmazrouei/Desktop/test_joint_nominal.csv \
#       --orientation_mode hold_initial \
#       --ik-seed 0.413164 -0.105627 -0.306975 -2.46688 -0.0954591 2.39035 -2.16711 \
#       --seed_is_initial_pose \
#       --center 0.36 -0.27 0.46 --x_m 0.45 --gamma_p 10.0 \
#       --qdot_max 2.175 2.175 2.175 2.175 2.610 2.610 2.610 \
#       --qddot_max 15 7.5 10 12.5 15 20 20 \
#       --q-min -2.8 -1.7 -2.8 -3.0 -2.8 0.5 -2.8 \
#       --q-max  2.8  1.7  2.8 -0.1  2.8 3.8  2.8 \
#       --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_altering_plots
# ============================================================


SCENE_WITH_GRIPPER = "scene.xml"
SCENE_NO_GRIPPER = "scene_nohand.xml"
ARM_DOFS = 7
FRANKA_QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=float)
FRANKA_QDDOT_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)


def damped_pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))


def solve_qdot_optimized(J, v_task, qdot_prev, qdot_sec, lam, w_smooth, w_null):
    """
    Solve:
      min ||J qdot - v_task||^2 + lam^2||qdot||^2
          + w_smooth||qdot - qdot_prev||^2 + w_null||qdot - qdot_sec||^2
    """
    I = np.eye(ARM_DOFS)
    A = J.T @ J + (lam * lam + w_smooth + w_null) * I
    b = J.T @ v_task + w_smooth * qdot_prev + w_null * qdot_sec
    return np.linalg.solve(A, b)


def solve_qdot_qp_projected(H, g, lb, ub, A_ineq=None, b_ineq=None, x0=None, iters=60):
    """
    Solve a small convex QP approximately:
      minimize 0.5 x^T H x - g^T x
      subject to lb <= x <= ub,  A_ineq x <= b_ineq
    using projected gradient with half-space projections.
    """
    if x0 is None:
        x = np.clip(np.zeros_like(lb), lb, ub)
    else:
        x = np.clip(x0.copy(), lb, ub)

    # Conservative step size from Hessian spectral norm.
    L = float(np.linalg.norm(H, ord=2))
    alpha = 1.0 / max(L, 1e-6)

    if A_ineq is None or b_ineq is None or len(A_ineq) == 0:
        A_rows = []
        b_vals = []
    else:
        A_rows = [np.asarray(r, dtype=float).reshape(-1) for r in A_ineq]
        b_vals = [float(v) for v in b_ineq]

    for _ in range(iters):
        grad = H @ x - g
        x = x - alpha * grad
        x = np.clip(x, lb, ub)

        # Project onto each half-space.
        for a, b in zip(A_rows, b_vals):
            viol = float(a @ x - b)
            if viol > 0.0:
                a2 = float(a @ a)
                if a2 > 1e-12:
                    x = x - (viol / a2) * a
                    x = np.clip(x, lb, ub)

    return x


def solve_qdot_qp_osqp(H, g, lb, ub, A_ineq, b_ineq, x_warm=None):
    n = H.shape[0]
    m = A_ineq.shape[0] if A_ineq is not None else 0
    P = sp.csc_matrix(2.0 * H)
    q = -2.0 * g
    if m > 0:
        A = sp.vstack([sp.eye(n, format="csc"), sp.csc_matrix(A_ineq)], format="csc")
        l = np.hstack([lb, -np.inf * np.ones(m)])
        u = np.hstack([ub, b_ineq])
    else:
        A = sp.eye(n, format="csc")
        l = lb
        u = ub
    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        verbose=False,
        warm_start=True,
        polish=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=4000,
    )
    if x_warm is not None:
        solver.warm_start(x=x_warm)
    res = solver.solve()
    if res.info.status_val not in (1, 2):  # solved / solved inaccurate
        raise RuntimeError(f"OSQP failed with status: {res.info.status}")
    return np.asarray(res.x, dtype=float).reshape(-1)


def enforce_position_cbf_velocity(v, x, c, x_m, gamma_p):
    """
    Enforce task-space CBF for position sphere:
      h = X_m^2 - ||x-c||^2
      CBF: h_dot + gamma_p h >= 0
      with h_dot = -2 (x-c)^T v
      => (x-c)^T v <= (gamma_p/2) h
    Projection onto half-space in v-space.
    """
    a = (x - c).reshape(3)
    a2 = float(a @ a)
    if a2 < 1e-12:
        return v
    h = x_m * x_m - a2
    b = 0.5 * gamma_p * h
    av = float(a @ v)
    if av <= b:
        return v
    return v - ((av - b) / a2) * a


def _is_float_token(token):
    try:
        float(token)
        return True
    except Exception:
        return False


def quat_xyzw_to_rot(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def rot_log_vee(R):
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    s = np.sin(theta)
    if abs(s) < 1e-8:
        return 0.5 * w
    return (theta / (2.0 * s)) * w


def load_trajectory_csv(path, dt_fallback):
    times = []
    points = []
    quat_ref = []
    R_ref = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) > 0]

    if len(rows) < 2:
        raise ValueError("CSV trajectory must contain at least 2 rows.")

    first = [c.strip() for c in rows[0]]
    has_header = not all(_is_float_token(c) for c in first)

    if has_header:
        header = [h.strip().lower() for h in first]
        data_rows = rows[1:]

        def find_col(candidates):
            for cand in candidates:
                if cand in header:
                    return header.index(cand)
            return None

        ix = find_col(["x", "px", "pos_x", "position_x", "o_t_ee[12]"])
        iy = find_col(["y", "py", "pos_y", "position_y", "o_t_ee[13]"])
        iz = find_col(["z", "pz", "pos_z", "position_z", "o_t_ee[14]"])
        it = find_col(["t", "time", "timestamp"])

        if ix is None or iy is None or iz is None:
            raise ValueError("Header must contain x,y,z-like columns.")

        iqx = find_col(["qx", "quat_x", "orientation_x"])
        iqy = find_col(["qy", "quat_y", "orientation_y"])
        iqz = find_col(["qz", "quat_z", "orientation_z"])
        iqw = find_col(["qw", "quat_w", "orientation_w", "w"])
        has_quat = (iqx is not None and iqy is not None and iqz is not None and iqw is not None)

        iR11 = find_col(["r11", "r_11"])
        iR12 = find_col(["r12", "r_12"])
        iR13 = find_col(["r13", "r_13"])
        iR21 = find_col(["r21", "r_21"])
        iR22 = find_col(["r22", "r_22"])
        iR23 = find_col(["r23", "r_23"])
        iR31 = find_col(["r31", "r_31"])
        iR32 = find_col(["r32", "r_32"])
        iR33 = find_col(["r33", "r_33"])
        has_R = all(idx is not None for idx in [iR11, iR12, iR13, iR21, iR22, iR23, iR31, iR32, iR33])

        for i, row in enumerate(data_rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+2} does not have enough columns.")
            points.append([float(row[ix]), float(row[iy]), float(row[iz])])
            if has_quat:
                q = np.array([float(row[iqx]), float(row[iqy]), float(row[iqz]), float(row[iqw])], dtype=float)
                quat_ref.append(q)
                R_ref.append(quat_xyzw_to_rot(q))
            elif has_R:
                R = np.array(
                    [
                        [float(row[iR11]), float(row[iR12]), float(row[iR13])],
                        [float(row[iR21]), float(row[iR22]), float(row[iR23])],
                        [float(row[iR31]), float(row[iR32]), float(row[iR33])],
                    ],
                    dtype=float,
                )
                R_ref.append(project_to_so3(R))
            if it is not None and it < len(row):
                times.append(float(row[it]))
            else:
                times.append(i * dt_fallback)
    else:
        ncols = len(first)
        if ncols >= 4:
            it, ix, iy, iz = 0, 1, 2, 3
        elif ncols >= 3:
            it, ix, iy, iz = None, 0, 1, 2
        else:
            raise ValueError("CSV without header must have at least 3 columns.")

        for i, row in enumerate(rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+1} does not have enough columns.")
            points.append([float(row[ix]), float(row[iy]), float(row[iz])])
            if it is not None:
                times.append(float(row[it]))
            else:
                times.append(i * dt_fallback)

    t = np.asarray(times, dtype=float)
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        raise ValueError("Need at least 2 points.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time must be strictly increasing.")

    q_ref = np.asarray(quat_ref, dtype=float) if len(quat_ref) > 0 else None
    R_ref = np.asarray(R_ref, dtype=float) if len(R_ref) > 0 else None
    return t, p, R_ref, q_ref


def get_ee_body_id(model):
    for name in ("hand", "attachment", "link7"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return bid
    raise RuntimeError("No EE body found (hand/attachment/link7).")


def get_joint_limits(model, dofs):
    q_min = -np.inf * np.ones(dofs)
    q_max = np.inf * np.ones(dofs)
    for j in range(dofs):
        if model.jnt_limited[j]:
            q_min[j] = model.jnt_range[j, 0]
            q_max[j] = model.jnt_range[j, 1]
    return q_min, q_max


def limit_avoidance_gradient(q, q_min, q_max, eps=1e-6):
    # Barrier-style gradient pushing joints away from hard limits.
    d_lo = np.maximum(q - q_min, eps)
    d_hi = np.maximum(q_max - q, eps)
    return (1.0 / d_lo) - (1.0 / d_hi)


def clip_qdot_to_joint_bounds(q, qdot, dt, q_min, q_max, margin):
    lo = np.where(np.isfinite(q_min), q_min + margin, -np.inf)
    hi = np.where(np.isfinite(q_max), q_max - margin, np.inf)
    bad = lo > hi
    lo = np.where(bad, q_min, lo)
    hi = np.where(bad, q_max, hi)
    qdot_lo = (lo - q) / dt
    qdot_hi = (hi - q) / dt
    return np.clip(qdot, qdot_lo, qdot_hi)


def parse_scalar_or_7(vals, name):
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr.item(), ARM_DOFS)
    if arr.size == ARM_DOFS:
        return arr
    raise ValueError(f"{name} must be 1 value or {ARM_DOFS} values.")


def initialize_to_first_reference(
    model,
    data,
    ee_body_id,
    q_init,
    q_min,
    q_max,
    x_ref0,
    R_ref0,
    lam,
    max_iters=2000,
    pos_tol=1e-5,
    ori_tol=1e-4,
):
    q = np.clip(np.asarray(q_init, dtype=float).copy(), q_min, q_max)
    data.qpos[:ARM_DOFS] = q
    mujoco.mj_forward(model, data)
    ori_err_norm = np.nan

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        x = data.xpos[ee_body_id].copy()
        R = data.xmat[ee_body_id].reshape(3, 3).copy()
        e_p = x_ref0 - x
        pos_err_norm = float(np.linalg.norm(e_p))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        Jp = jacp[:, :ARM_DOFS]
        Jr = jacr[:, :ARM_DOFS]

        if R_ref0 is not None:
            R_err = R_ref0 @ R.T
            e_w = 0.5 * np.array(
                [
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ],
                dtype=float,
            )
            ori_err_norm = float(np.linalg.norm(e_w))
            if pos_err_norm < pos_tol and ori_err_norm < ori_tol:
                break
            J = np.vstack([Jp, Jr])
            e = np.hstack([4.0 * e_p, 3.0 * e_w])
        else:
            if pos_err_norm < pos_tol:
                break
            J = Jp
            e = 4.0 * e_p

        dq = damped_pinv(J, lam) @ e
        dq = np.clip(dq, -0.05, 0.05)
        q = np.clip(q + dq, q_min, q_max)
        data.qpos[:ARM_DOFS] = q

    mujoco.mj_forward(model, data)
    x = data.xpos[ee_body_id].copy()
    R = data.xmat[ee_body_id].reshape(3, 3).copy()
    pos_err_norm = float(np.linalg.norm(x_ref0 - x))
    if R_ref0 is not None:
        R_err = R_ref0 @ R.T
        e_w = 0.5 * np.array(
            [
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ],
            dtype=float,
        )
        ori_err_norm = float(np.linalg.norm(e_w))
    return q.copy(), pos_err_norm, ori_err_norm


def main():
    parser = argparse.ArgumentParser(
        description="Alter task-space trajectory using joint-space-feasible IK + nullspace."
    )
    parser.add_argument("--in_csv", type=Path, required=True, help="Input task-space CSV.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output altered CSV.")
    parser.add_argument(
        "--out_nominal_csv",
        type=Path,
        default=None,
        help="Output nominal joint-space CSV (default: <out_csv stem>_nominal.csv).",
    )
    parser.add_argument(
        "--seed_is_initial_pose",
        action="store_true",
        help="Use ik-seed exactly as initial joint pose (skip start IK alignment to first reference).",
    )
    parser.add_argument("--no-gripper", action="store_true", help="Use scene_nohand.xml.")
    parser.add_argument("--dt_fallback", type=float, default=0.001, help="Fallback dt for CSV without time.")
    parser.add_argument("--k_track", type=float, default=1.0, help="Task tracking gain.")
    parser.add_argument("--k_ori", type=float, default=2.0, help="Orientation tracking gain (if orientation columns exist).")
    parser.add_argument("--w_max", type=float, default=1.0, help="Angular velocity clamp for orientation tracking.")
    parser.add_argument("--k_home", type=float, default=0.05, help="Nullspace home gain.")
    parser.add_argument("--k_limit", type=float, default=0.02, help="Nullspace limit-avoid gain.")
    parser.add_argument("--w_smooth", type=float, default=5e-2, help="Weight for qdot smoothness (closer to previous qdot).")
    parser.add_argument("--w_null", type=float, default=1e-2, help="Weight for secondary/nullspace objective.")
    parser.add_argument("--qp-iters", type=int, default=60, help="Iterations for internal projected QP solve.")
    parser.add_argument("--lambda_dls", type=float, default=2e-2, help="Damped pseudo-inverse lambda.")
    parser.add_argument(
        "--qdot_max",
        nargs="+",
        type=float,
        default=None,
        help="Joint speed limits. Provide 1 value or 7 values. Default: Franka per-joint limits.",
    )
    parser.add_argument(
        "--qddot_max",
        nargs="+",
        type=float,
        default=None,
        help="Joint accel limits. Provide 1 value or 7 values. Default: Franka per-joint limits.",
    )
    parser.add_argument("--q-min", nargs="+", type=float, default=None, help="Joint lower bounds override (1 or 7 values).")
    parser.add_argument("--q-max", nargs="+", type=float, default=None, help="Joint upper bounds override (1 or 7 values).")
    parser.add_argument("--joint_margin", type=float, default=0.02, help="Joint-limit margin [rad].")
    parser.add_argument("--center", nargs=3, type=float, default=[0.36, -0.27, 0.46], metavar=("CX", "CY", "CZ"))
    parser.add_argument("--x_m", type=float, default=0.45, help="Task-space position bound radius.")
    parser.add_argument("--v_m", type=float, default=1.15, help="Task-space speed bound (via linearized constraints).")
    parser.add_argument("--a_m", type=float, default=13.0, help="Task-space accel bound (via linearized constraints).")
    parser.add_argument(
        "--tol_plot_outside",
        type=float,
        default=1e-3,
        help="Plot-only tolerance for outside highlighting (abs margin added to bounds).",
    )
    parser.add_argument("--gamma_p", type=float, default=10.0, help="Task-space position CBF gain.")
    parser.add_argument("--gamma_v", type=float, default=10.0, help="Task-space velocity CBF gain.")
    parser.add_argument(
        "--orientation_mode",
        type=str,
        choices=["track_ref", "hold_initial", "ignore"],
        default="track_ref",
        help="Orientation behavior: track_ref (default), hold_initial (constant target), or ignore.",
    )
    parser.add_argument("--ik-seed", nargs=7, type=float, default=None, help="Initial q seed q0..q6.")
    parser.add_argument(
        "--q-home",
        nargs=7,
        type=float,
        default=[0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8],
        help="Home posture for nullspace.",
    )
    parser.add_argument("--no_plot", action="store_true", help="Disable interactive plotting.")
    parser.add_argument("--save_plots_dir", type=Path, default=None, help="Save plots to this directory (PNG).")
    args = parser.parse_args()
    if args.no_plot or args.save_plots_dir is not None:
        plt.switch_backend("Agg")
        plt.rcParams["figure.max_open_warning"] = 0

    xml_path = SCENE_NO_GRIPPER if args.no_gripper else SCENE_WITH_GRIPPER
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ee_body_id = get_ee_body_id(model)
    q_min, q_max = get_joint_limits(model, ARM_DOFS)
    q_min_override = parse_scalar_or_7(args.q_min, "--q-min")
    q_max_override = parse_scalar_or_7(args.q_max, "--q-max")
    if q_min_override is not None:
        q_min = q_min_override
    if q_max_override is not None:
        q_max = q_max_override
    if np.any(q_min > q_max):
        raise ValueError("Invalid joint bounds: each q-min must be <= q-max.")
    if args.qdot_max is None:
        qdot_max = FRANKA_QDOT_MAX.copy()
    else:
        qdot_max = parse_scalar_or_7(args.qdot_max, "--qdot_max")
    if args.qddot_max is None:
        qddot_max = FRANKA_QDDOT_MAX.copy()
    else:
        qddot_max = parse_scalar_or_7(args.qddot_max, "--qddot_max")
    q_home = np.asarray(args.q_home, dtype=float)
    c = np.asarray(args.center, dtype=float)
    out_nominal_csv = (
        args.out_nominal_csv if args.out_nominal_csv is not None else args.out_csv.with_name(f"{args.out_csv.stem}_nominal.csv")
    )

    t_ref, x_ref, R_ref, q_ref = load_trajectory_csv(args.in_csv, args.dt_fallback)
    has_orientation_ref = R_ref is not None and len(R_ref) == len(x_ref)
    n = len(t_ref)

    data_nom = mujoco.MjData(model)
    q_seed = data.qpos[:ARM_DOFS].copy()
    if args.ik_seed is not None:
        q_seed = np.asarray(args.ik_seed, dtype=float)
    q_seed = np.clip(q_seed, q_min, q_max)
    if args.seed_is_initial_pose:
        q = q_seed.copy()
        q_nom = q_seed.copy()
        data.qpos[:ARM_DOFS] = q
        data.ctrl[:ARM_DOFS] = q
        data_nom.qpos[:ARM_DOFS] = q_nom
        data_nom.ctrl[:ARM_DOFS] = q_nom
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(model, data_nom)
        x0 = data.xpos[ee_body_id].copy()
        x0_nom = data_nom.xpos[ee_body_id].copy()
        start_pos_err = float(np.linalg.norm(x_ref[0] - x0))
        start_pos_err_nom = float(np.linalg.norm(x_ref[0] - x0_nom))
        if has_orientation_ref:
            R0 = data.xmat[ee_body_id].reshape(3, 3).copy()
            R0_nom = data_nom.xmat[ee_body_id].reshape(3, 3).copy()
            R_err0 = R_ref[0] @ R0.T
            R_err0_nom = R_ref[0] @ R0_nom.T
            e_w0 = 0.5 * np.array(
                [R_err0[2, 1] - R_err0[1, 2], R_err0[0, 2] - R_err0[2, 0], R_err0[1, 0] - R_err0[0, 1]],
                dtype=float,
            )
            e_w0_nom = 0.5 * np.array(
                [
                    R_err0_nom[2, 1] - R_err0_nom[1, 2],
                    R_err0_nom[0, 2] - R_err0_nom[2, 0],
                    R_err0_nom[1, 0] - R_err0_nom[0, 1],
                ],
                dtype=float,
            )
            start_ori_err = float(np.linalg.norm(e_w0))
            start_ori_err_nom = float(np.linalg.norm(e_w0_nom))
        else:
            start_ori_err = np.nan
            start_ori_err_nom = np.nan
    else:
        q = q_seed.copy()
        q, start_pos_err, start_ori_err = initialize_to_first_reference(
            model=model,
            data=data,
            ee_body_id=ee_body_id,
            q_init=q,
            q_min=q_min,
            q_max=q_max,
            x_ref0=x_ref[0],
            R_ref0=(R_ref[0] if has_orientation_ref else None),
            lam=args.lambda_dls,
        )
        q_nom, start_pos_err_nom, start_ori_err_nom = initialize_to_first_reference(
            model=model,
            data=data_nom,
            ee_body_id=ee_body_id,
            q_init=q_seed,
            q_min=q_min,
            q_max=q_max,
            x_ref0=x_ref[0],
            R_ref0=(R_ref[0] if has_orientation_ref else None),
            lam=args.lambda_dls,
        )
    data.qpos[:ARM_DOFS] = q
    data.ctrl[:ARM_DOFS] = q
    data_nom.qpos[:ARM_DOFS] = q_nom
    data_nom.ctrl[:ARM_DOFS] = q_nom
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, data_nom)

    orientation_track = args.orientation_mode != "ignore"
    orientation_mode_effective = args.orientation_mode
    if args.orientation_mode == "track_ref" and not has_orientation_ref:
        orientation_mode_effective = "hold_initial"
        print("Orientation mode fallback: no orientation reference in CSV -> using hold_initial.")
    if orientation_track:
        if orientation_mode_effective == "track_ref":
            R_target = R_ref.copy()
        else:
            # Constant orientation target equal to initial EE orientation.
            R0 = data.xmat[ee_body_id].reshape(3, 3).copy()
            R_target = np.repeat(R0[None, :, :], n, axis=0)
    else:
        R_target = None

    if has_orientation_ref:
        print(f"Start alignment | pos_err={start_pos_err:.6e} m, ori_err={start_ori_err:.6e} rad")
        print(
            f"Nominal start alignment | pos_err={start_pos_err_nom:.6e} m, ori_err={start_ori_err_nom:.6e} rad"
        )
    else:
        print(f"Start alignment | pos_err={start_pos_err:.6e} m")
        print(f"Nominal start alignment | pos_err={start_pos_err_nom:.6e} m")
    print(f"QP backend: {'OSQP' if HAS_OSQP else 'Projected fallback'}")
    print(f"Orientation mode: {orientation_mode_effective}")
    print(f"Initial pose mode: {'seed exact' if args.seed_is_initial_pose else 'aligned to first reference'}")

    q_hist = np.zeros((n, ARM_DOFS))
    qdot_hist = np.zeros((n, ARM_DOFS))
    q_nom_hist = np.zeros((n, ARM_DOFS))
    qdot_nom_hist = np.zeros((n, ARM_DOFS))
    x_exec = np.zeros((n, 3))
    x_nom_exec = np.zeros((n, 3))
    R_exec = np.zeros((n, 3, 3))
    R_nom_exec = np.zeros((n, 3, 3))
    err = np.zeros(n)
    err_nom = np.zeros(n)
    ori_err = np.full(n, np.nan)
    ori_err_nom = np.full(n, np.nan)
    cbf_p_res = np.zeros(n)
    cbf_p_res_nom = np.zeros(n)
    qdot_prev = np.zeros(ARM_DOFS)
    qdot_prev_nom = np.zeros(ARM_DOFS)

    for i in range(n):
        dt = t_ref[i] - t_ref[i - 1] if i > 0 else (t_ref[1] - t_ref[0])
        dt = max(dt, 1e-6)
        if i < n - 1:
            v_ff = (x_ref[i + 1] - x_ref[i]) / max(t_ref[i + 1] - t_ref[i], 1e-6)
        else:
            v_ff = np.zeros(3)
        if orientation_track:
            if i < n - 1:
                dt_ff = max(t_ref[i + 1] - t_ref[i], 1e-6)
                R_rel = R_target[i].T @ R_target[i + 1]
                w_ff = rot_log_vee(project_to_so3(R_rel)) / dt_ff
            else:
                w_ff = np.zeros(3)

        # -------------------- Nominal rollout (unconstrained in task-space) --------------------
        mujoco.mj_forward(model, data_nom)
        x_nom = data_nom.xpos[ee_body_id].copy()
        R_nom = data_nom.xmat[ee_body_id].reshape(3, 3).copy()
        q_nom_hist[i] = q_nom.copy()
        x_nom_exec[i] = x_nom
        R_nom_exec[i] = R_nom
        err_nom[i] = np.linalg.norm(x_ref[i] - x_nom)

        v_des_nom = v_ff + args.k_track * (x_ref[i] - x_nom)
        h_nom = args.x_m * args.x_m - float((x_nom - c) @ (x_nom - c))

        jacp_nom = np.zeros((3, model.nv))
        jacr_nom = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data_nom, jacp_nom, jacr_nom, ee_body_id)
        Jp_nom = jacp_nom[:, :ARM_DOFS]
        Jr_nom = jacr_nom[:, :ARM_DOFS]
        if orientation_track:
            R_err_nom = R_target[i] @ R_nom.T
            e_w_nom = 0.5 * np.array(
                [
                    R_err_nom[2, 1] - R_err_nom[1, 2],
                    R_err_nom[0, 2] - R_err_nom[2, 0],
                    R_err_nom[1, 0] - R_err_nom[0, 1],
                ],
                dtype=float,
            )
            ori_err_nom[i] = float(np.linalg.norm(e_w_nom))
            w_des_nom = w_ff + args.k_ori * e_w_nom
            nw_nom = np.linalg.norm(w_des_nom)
            if nw_nom > args.w_max and nw_nom > 1e-12:
                w_des_nom = w_des_nom * (args.w_max / nw_nom)
            v_task_nom = np.hstack([v_des_nom, w_des_nom])
            J_nom = np.vstack([Jp_nom, Jr_nom])
        else:
            v_task_nom = v_des_nom
            J_nom = Jp_nom

        grad_lim_nom = limit_avoidance_gradient(q_nom, q_min, q_max)
        qdot_sec_nom = args.k_home * (q_home - q_nom) - args.k_limit * grad_lim_nom
        qdot_nom = solve_qdot_optimized(
            J=J_nom,
            v_task=v_task_nom,
            qdot_prev=qdot_prev_nom,
            qdot_sec=qdot_sec_nom,
            lam=args.lambda_dls,
            w_smooth=args.w_smooth,
            w_null=args.w_null,
        )
        # Keep nominal rollout dynamically comparable to altered rollout:
        # enforce joint speed, accel-step, and position-rate bounds, but no task-space CBF constraints.
        lo_nom = np.where(np.isfinite(q_min), q_min + args.joint_margin, -np.inf)
        hi_nom = np.where(np.isfinite(q_max), q_max - args.joint_margin, np.inf)
        bad_nom = lo_nom > hi_nom
        lo_nom = np.where(bad_nom, q_min, lo_nom)
        hi_nom = np.where(bad_nom, q_max, hi_nom)
        lb_pos_nom = (lo_nom - q_nom) / dt
        ub_pos_nom = (hi_nom - q_nom) / dt
        lb_nom = np.maximum.reduce(
            [
                -qdot_max,
                qdot_prev_nom - qddot_max * dt,
                lb_pos_nom,
            ]
        )
        ub_nom = np.minimum.reduce(
            [
                qdot_max,
                qdot_prev_nom + qddot_max * dt,
                ub_pos_nom,
            ]
        )
        infeas_nom = lb_nom > ub_nom
        if np.any(infeas_nom):
            mid_nom = 0.5 * (lb_nom + ub_nom)
            lb_nom[infeas_nom] = mid_nom[infeas_nom]
            ub_nom[infeas_nom] = mid_nom[infeas_nom]
        qdot_nom = np.clip(qdot_nom, lb_nom, ub_nom)
        qdot_nom = clip_qdot_to_joint_bounds(q_nom, qdot_nom, dt, q_min, q_max, args.joint_margin)
        cbf_p_res_nom[i] = -2.0 * float((x_nom - c) @ (Jp_nom @ qdot_nom)) + args.gamma_p * h_nom
        if i < n - 1:
            q_nom = q_nom + qdot_nom * dt
            q_nom = np.clip(q_nom, q_min, q_max)
            data_nom.qpos[:ARM_DOFS] = q_nom
            data_nom.ctrl[:ARM_DOFS] = q_nom
            qdot_prev_nom = qdot_nom.copy()
        else:
            qdot_nom = np.zeros(ARM_DOFS)
            data_nom.ctrl[:ARM_DOFS] = q_nom
        qdot_nom_hist[i] = qdot_nom

        # -------------------- Altered rollout (QP constrained) --------------------
        mujoco.mj_forward(model, data)
        x = data.xpos[ee_body_id].copy()
        R = data.xmat[ee_body_id].reshape(3, 3).copy()
        q_hist[i] = q.copy()
        x_exec[i] = x
        R_exec[i] = R
        err[i] = np.linalg.norm(x_ref[i] - x)

        v_des = v_ff + args.k_track * (x_ref[i] - x)
        h = args.x_m * args.x_m - float((x - c) @ (x - c))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        Jp = jacp[:, :ARM_DOFS]
        Jr = jacr[:, :ARM_DOFS]
        if orientation_track:
            R_err = R_target[i] @ R.T
            e_w = 0.5 * np.array(
                [
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ],
                dtype=float,
            )
            ori_err[i] = float(np.linalg.norm(e_w))
            w_des = w_ff + args.k_ori * e_w
            nw = np.linalg.norm(w_des)
            if nw > args.w_max and nw > 1e-12:
                w_des = w_des * (args.w_max / nw)
            v_task = np.hstack([v_des, w_des])
            J = np.vstack([Jp, Jr])
        else:
            v_task = v_des
            J = Jp

        grad_lim = limit_avoidance_gradient(q, q_min, q_max)
        qdot_sec = args.k_home * (q_home - q) - args.k_limit * grad_lim

        # QP objective: task tracking + damping + smoothness + nullspace.
        I = np.eye(ARM_DOFS)
        H = J.T @ J + (args.lambda_dls * args.lambda_dls + args.w_smooth + args.w_null) * I
        g = J.T @ v_task + args.w_smooth * qdot_prev + args.w_null * qdot_sec

        # Hard joint-space bounds.
        lo = np.where(np.isfinite(q_min), q_min + args.joint_margin, -np.inf)
        hi = np.where(np.isfinite(q_max), q_max - args.joint_margin, np.inf)
        bad = lo > hi
        lo = np.where(bad, q_min, lo)
        hi = np.where(bad, q_max, hi)
        lb_pos = (lo - q) / dt
        ub_pos = (hi - q) / dt
        lb = np.maximum.reduce(
            [
                -qdot_max,
                qdot_prev - qddot_max * dt,
                lb_pos,
            ]
        )
        ub = np.minimum.reduce(
            [
                qdot_max,
                qdot_prev + qddot_max * dt,
                ub_pos,
            ]
        )
        infeas = lb > ub
        if np.any(infeas):
            mid = 0.5 * (lb + ub)
            lb[infeas] = mid[infeas]
            ub[infeas] = mid[infeas]

        # Task-space linear constraints.
        A_rows = []
        b_rows = []

        # Position CBF: -2(x-c)^T Jp qdot + gamma h >= 0  <=>  a qdot <= b
        a_cbf = 2.0 * ((x - c) @ Jp)
        b_cbf = args.gamma_p * h
        A_rows.append(a_cbf)
        b_rows.append(b_cbf)

        # Velocity CBF:
        # h_v = V_m^2 - ||v||^2,  h_v_dot + gamma_v h_v >= 0
        # with v ≈ v_prev_task and v_dot ≈ (Jp qdot - v_prev_task)/dt
        # => v_prev_task^T Jp qdot <= ||v_prev_task||^2 + 0.5*gamma_v*dt*(V_m^2-||v_prev_task||^2)
        v_prev_task = Jp @ qdot_prev
        v_prev_n2 = float(v_prev_task @ v_prev_task)
        a_vcbf = v_prev_task @ Jp
        b_vcbf = v_prev_n2 + 0.5 * args.gamma_v * dt * (args.v_m * args.v_m - v_prev_n2)
        A_rows.append(a_vcbf)
        b_rows.append(b_vcbf)

        # Conservative task speed bound: ||v||2 <= V_m via ||v||inf <= V_m/sqrt(3)
        v_inf = args.v_m / np.sqrt(3.0)
        for rj in range(3):
            a = Jp[rj, :]
            A_rows.append(a)
            b_rows.append(v_inf)
            A_rows.append(-a)
            b_rows.append(v_inf)

        # Conservative task accel bound: ||a||2 <= A_m via ||a||inf <= A_m/sqrt(3)
        # Approx: a_task ≈ (Jp qdot - v_prev_task)/dt
        a_inf = args.a_m / np.sqrt(3.0)
        # Reuse v_prev_task from velocity-CBF construction.
        for rj in range(3):
            a = Jp[rj, :]
            A_rows.append(a)
            b_rows.append(v_prev_task[rj] + a_inf * dt)
            A_rows.append(-a)
            b_rows.append(-v_prev_task[rj] + a_inf * dt)

        A_ineq = np.asarray(A_rows, dtype=float)
        b_ineq = np.asarray(b_rows, dtype=float)

        if HAS_OSQP:
            try:
                qdot = solve_qdot_qp_osqp(H, g, lb, ub, A_ineq, b_ineq, x_warm=qdot_prev)
            except Exception:
                qdot = solve_qdot_qp_projected(
                    H=H, g=g, lb=lb, ub=ub, A_ineq=A_ineq, b_ineq=b_ineq, x0=qdot_prev, iters=args.qp_iters
                )
        else:
            qdot = solve_qdot_qp_projected(
                H=H, g=g, lb=lb, ub=ub, A_ineq=A_ineq, b_ineq=b_ineq, x0=qdot_prev, iters=args.qp_iters
            )

        # Final clipping for numerical safety.
        qdot = np.clip(qdot, lb, ub)
        qdot = clip_qdot_to_joint_bounds(q, qdot, dt, q_min, q_max, args.joint_margin)
        v_exec_now = Jp @ qdot
        cbf_p_res[i] = -2.0 * float((x - c) @ v_exec_now) + args.gamma_p * h

        if i < n - 1:
            q = q + qdot * dt
            q = np.clip(q, q_min, q_max)
            data.qpos[:ARM_DOFS] = q
            data.ctrl[:ARM_DOFS] = q
            qdot_prev = qdot.copy()
        else:
            qdot = np.zeros(ARM_DOFS)
            data.ctrl[:ARM_DOFS] = q
        qdot_hist[i] = qdot

    v_exec = np.gradient(x_exec, t_ref, axis=0)
    a_exec = np.gradient(v_exec, t_ref, axis=0)
    v_nom_exec = np.gradient(x_nom_exec, t_ref, axis=0)
    a_nom_exec = np.gradient(v_nom_exec, t_ref, axis=0)

    header = (
        [
            "t",
            "x_ref",
            "y_ref",
            "z_ref",
            "x_exec",
            "y_exec",
            "z_exec",
            "vx_exec",
            "vy_exec",
            "vz_exec",
            "ax_exec",
            "ay_exec",
            "az_exec",
            "err",
            "cbf_p_res",
        ]
        + [f"q{i}" for i in range(ARM_DOFS)]
        + [f"qdot{i}" for i in range(ARM_DOFS)]
    )
    arrays = [t_ref, x_ref, x_exec, v_exec, a_exec, err, cbf_p_res, q_hist, qdot_hist]
    if q_ref is not None:
        header += ["qx_ref", "qy_ref", "qz_ref", "qw_ref"]
        arrays.append(q_ref)
    if orientation_track:
        header += [f"r{i}{j}_ref" for i in (1, 2, 3) for j in (1, 2, 3)]
        header += [f"r{i}{j}_exec" for i in (1, 2, 3) for j in (1, 2, 3)]
        header += ["ori_err"]
        arrays.append(R_target.reshape(n, 9))
        arrays.append(R_exec.reshape(n, 9))
        arrays.append(ori_err.reshape(n, 1))
    out = np.column_stack(arrays)
    np.savetxt(args.out_csv, out, delimiter=",", header=",".join(header), comments="")

    header_nom = (
        [
            "t",
            "x_ref",
            "y_ref",
            "z_ref",
            "x_exec",
            "y_exec",
            "z_exec",
            "vx_exec",
            "vy_exec",
            "vz_exec",
            "ax_exec",
            "ay_exec",
            "az_exec",
            "err",
            "cbf_p_res",
        ]
        + [f"q{i}" for i in range(ARM_DOFS)]
        + [f"qdot{i}" for i in range(ARM_DOFS)]
    )
    arrays_nom = [t_ref, x_ref, x_nom_exec, v_nom_exec, a_nom_exec, err_nom, cbf_p_res_nom, q_nom_hist, qdot_nom_hist]
    if q_ref is not None:
        header_nom += ["qx_ref", "qy_ref", "qz_ref", "qw_ref"]
        arrays_nom.append(q_ref)
    if orientation_track:
        header_nom += [f"r{i}{j}_ref" for i in (1, 2, 3) for j in (1, 2, 3)]
        header_nom += [f"r{i}{j}_exec" for i in (1, 2, 3) for j in (1, 2, 3)]
        header_nom += ["ori_err"]
        arrays_nom.append(R_target.reshape(n, 9))
        arrays_nom.append(R_nom_exec.reshape(n, 9))
        arrays_nom.append(ori_err_nom.reshape(n, 1))
    out_nom = np.column_stack(arrays_nom)
    np.savetxt(out_nominal_csv, out_nom, delimiter=",", header=",".join(header_nom), comments="")

    print(f"Saved altered trajectory to: {args.out_csv}")
    print(f"Saved nominal trajectory to: {out_nominal_csv}")
    print(f"Samples: {n}")
    print(f"Tracking error mean/max: {np.mean(err):.6f} / {np.max(err):.6f} m")
    print(f"Nominal tracking error mean/max: {np.mean(err_nom):.6f} / {np.max(err_nom):.6f} m")
    print(f"Task-space CBF residual min: {np.min(cbf_p_res):.6e}")
    print(f"Nominal task-space CBF residual min: {np.min(cbf_p_res_nom):.6e}")
    if orientation_track:
        valid = ~np.isnan(ori_err)
        print(f"Orientation err mean/max: {np.mean(ori_err[valid]):.6f} / {np.max(ori_err[valid]):.6f} rad")
        valid_nom = ~np.isnan(ori_err_nom)
        print(
            "Nominal orientation err mean/max: "
            f"{np.mean(ori_err_nom[valid_nom]):.6f} / {np.max(ori_err_nom[valid_nom]):.6f} rad"
        )

    # ==================== PLOTS ====================
    r_ref = np.linalg.norm(x_ref - c[None, :], axis=1)
    r_exec = np.linalg.norm(x_exec - c[None, :], axis=1)
    h_p_ref = args.x_m * args.x_m - r_ref * r_ref
    h_p_exec = args.x_m * args.x_m - r_exec * r_exec
    vn_exec = np.linalg.norm(v_exec, axis=1)
    an_exec = np.linalg.norm(a_exec, axis=1)
    h_v_exec = args.v_m * args.v_m - vn_exec * vn_exec
    h_a_exec = args.a_m * args.a_m - an_exec * an_exec
    outside_p = r_exec > (args.x_m + args.tol_plot_outside)
    outside_v = vn_exec > (args.v_m + args.tol_plot_outside)
    outside_a = an_exec > (args.a_m + args.tol_plot_outside)
    idxBp = np.where(np.abs(h_p_exec) <= -0.1)[0]
    idxBv = np.where(np.abs(h_v_exec) <= -0.1)[0]
    idxBa = np.where(np.abs(h_a_exec) <= -0.1)[0]
    qddot_hist = np.gradient(qdot_hist, t_ref, axis=0)
    qdot_usage = np.abs(qdot_hist) / np.maximum(qdot_max[None, :], 1e-12)
    qddot_usage = np.abs(qddot_hist) / np.maximum(qddot_max[None, :], 1e-12)
    max_qdot_usage = np.max(qdot_usage, axis=1)
    max_qddot_usage = np.max(qddot_usage, axis=1)
    min_margin_t = np.min(np.minimum(q_hist - q_min[None, :], q_max[None, :] - q_hist), axis=1)

    # 3D reference and executed trajectory with safety sphere.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Reference vs executed trajectory with safety sphere")
    u = np.linspace(0.0, 2.0 * np.pi, 61)
    v = np.linspace(0.0, np.pi, 61)
    xs = c[0] + args.x_m * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + args.x_m * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + args.x_m * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.20, linewidth=0.0, color=(0.85, 0.90, 1.00))
    ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], "k--", linewidth=1.2, label="Reference")
    ax.plot(x_exec[:, 0], x_exec[:, 1], x_exec[:, 2], "b", linewidth=1.8, label="Executed")
    x_out = x_exec.copy()
    x_out[~outside_p, :] = np.nan
    ax.plot(x_out[:, 0], x_out[:, 1], x_out[:, 2], "r", linewidth=2.2, label="Outside")
    if idxBp.size > 0:
        ax.plot(x_exec[idxBp, 0], x_exec[idxBp, 1], x_exec[idxBp, 2], "ro", markersize=5, label="Boundary-near")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, h_p_ref, "k--", linewidth=1.2, label="h_p reference")
    plt.plot(t_ref, h_p_exec, "b", linewidth=1.8, label="h_p executed")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBp.size > 0:
        plt.plot(t_ref[idxBp], h_p_exec[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_p")
    plt.title("Position barrier h_p(t)")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, r_ref, "k--", linewidth=1.2, label="||x_ref-c||")
    plt.plot(t_ref, r_exec, "b", linewidth=1.8, label="||x_exec-c||")
    plt.axhline(args.x_m, color="k", linestyle="--", linewidth=1.2, label="X_m")
    r_out = r_exec.copy()
    r_out[~outside_p] = np.nan
    plt.plot(t_ref, r_out, "r", linewidth=2.2, label="outside")
    if idxBp.size > 0:
        plt.plot(t_ref[idxBp], r_exec[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("distance")
    plt.title("Distance to sphere center")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, vn_exec, linewidth=1.8, label="||v_exec||")
    plt.axhline(args.v_m, color="k", linestyle="--", linewidth=1.2, label="V_m")
    vn_out = vn_exec.copy()
    vn_out[~outside_v] = np.nan
    plt.plot(t_ref, vn_out, "r", linewidth=2.2, label="outside")
    if idxBv.size > 0:
        plt.plot(t_ref[idxBv], vn_exec[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("speed")
    plt.title("Task-space speed norm")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, h_v_exec, linewidth=1.8, label="h_v executed")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBv.size > 0:
        plt.plot(t_ref[idxBv], h_v_exec[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_v")
    plt.title("Velocity barrier h_v(t)")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, an_exec, linewidth=1.8, label="||a_exec||")
    plt.axhline(args.a_m, color="k", linestyle="--", linewidth=1.2, label="A_m")
    an_out = an_exec.copy()
    an_out[~outside_a] = np.nan
    plt.plot(t_ref, an_out, "r", linewidth=2.2, label="outside")
    if idxBa.size > 0:
        plt.plot(t_ref[idxBa], an_exec[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("acceleration")
    plt.title("Task-space acceleration norm")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, h_a_exec, linewidth=1.8, label="h_a executed")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBa.size > 0:
        plt.plot(t_ref[idxBa], h_a_exec[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_a")
    plt.title("Acceleration barrier h_a(t)")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, cbf_p_res, linewidth=1.8, label="position CBF residual")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    plt.xlabel("Time t")
    plt.ylabel("CBF residual")
    plt.title("Position CBF residual (should be >= 0)")
    plt.legend(loc="best")

    # Cartesian axis tracking.
    axis_names = ["x", "y", "z"]
    for k, nm in enumerate(axis_names):
        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, x_ref[:, k], "k--", linewidth=1.2, label=f"{nm}_ref")
        plt.plot(t_ref, x_exec[:, k], "b", linewidth=1.8, label=f"{nm}_exec")
        plt.xlabel("Time t")
        plt.ylabel(f"{nm} [m]")
        plt.title(f"Cartesian axis tracking: {nm}(t)")
        plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, err, linewidth=1.8, label="position tracking error")
    plt.xlabel("Time t")
    plt.ylabel("error [m]")
    plt.title("Task-space tracking error norm")
    plt.legend(loc="best")

    comp = ["x", "y", "z"]
    for k, nm in enumerate(comp):
        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, v_exec[:, k], linewidth=1.8, label=f"v_exec_{nm}")
        plt.xlabel("Time t")
        plt.ylabel(f"v_{nm} [m/s]")
        plt.title(f"Executed task-space velocity component {nm}(t)")
        plt.legend(loc="best")

    for k, nm in enumerate(comp):
        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, a_exec[:, k], linewidth=1.8, label=f"a_exec_{nm}")
        plt.xlabel("Time t")
        plt.ylabel(f"a_{nm} [m/s^2]")
        plt.title(f"Executed task-space acceleration component {nm}(t)")
        plt.legend(loc="best")

    if orientation_track:
        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, ori_err, linewidth=1.8, label="orientation error")
        plt.xlabel("Time t")
        plt.ylabel("error [rad]")
        plt.title("Orientation error norm")
        plt.legend(loc="best")

    # Joint-space diagnostics.
    lo = np.where(np.isfinite(q_min), q_min + args.joint_margin, -np.inf)
    hi = np.where(np.isfinite(q_max), q_max - args.joint_margin, np.inf)

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, max_qdot_usage, linewidth=1.8, label="max_j |qdot_j|/qdot_max_j")
    plt.plot(t_ref, max_qddot_usage, linewidth=1.8, label="max_j |qddot_j|/qddot_max_j")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="limit")
    plt.xlabel("Time t")
    plt.ylabel("normalized usage")
    plt.title("Worst-case joint velocity/acceleration usage")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t_ref, min_margin_t, linewidth=1.8, label="min_j dist to hard limit")
    plt.axhline(args.joint_margin, color="k", linestyle="--", linewidth=1.2, label="margin target")
    plt.xlabel("Time t")
    plt.ylabel("margin [rad]")
    plt.title("Closest distance to joint hard limits over time")
    plt.legend(loc="best")

    plt.figure()
    idx = np.arange(ARM_DOFS)
    max_qdot_joint = np.max(qdot_usage, axis=0)
    max_qddot_joint = np.max(qddot_usage, axis=0)
    w = 0.38
    plt.bar(idx - w / 2.0, max_qdot_joint, width=w, label="max |qdot|/qdot_max")
    plt.bar(idx + w / 2.0, max_qddot_joint, width=w, label="max |qddot|/qddot_max")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="limit")
    plt.xticks(idx, [f"q{j}" for j in range(ARM_DOFS)])
    plt.ylabel("peak normalized usage")
    plt.title("Per-joint peak dynamic usage")
    plt.grid(True, axis="y")
    plt.legend(loc="best")

    for j in range(ARM_DOFS):
        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, q_hist[:, j], linewidth=1.8, label=f"q{j}")
        if np.isfinite(q_min[j]):
            plt.axhline(q_min[j], color="k", linestyle="--", linewidth=1.2, label="q_min")
            plt.axhline(lo[j], color="k", linestyle=":", linewidth=1.2, label="q_min+margin")
        if np.isfinite(q_max[j]):
            plt.axhline(q_max[j], color="k", linestyle="--", linewidth=1.2, label="q_max")
            plt.axhline(hi[j], color="k", linestyle=":", linewidth=1.2, label="q_max-margin")
        plt.xlabel("Time t")
        plt.ylabel(f"q{j} [rad]")
        plt.title(f"Joint position q{j}(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, qdot_hist[:, j], linewidth=1.8, label=f"qdot{j}")
        plt.axhline(qdot_max[j], color="k", linestyle="--", linewidth=1.2, label="qdot_max")
        plt.axhline(-qdot_max[j], color="k", linestyle="--", linewidth=1.2)
        plt.xlabel("Time t")
        plt.ylabel(f"qdot{j} [rad/s]")
        plt.title(f"Joint velocity qdot{j}(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t_ref, qddot_hist[:, j], linewidth=1.8, label=f"qddot{j}")
        plt.axhline(qddot_max[j], color="k", linestyle="--", linewidth=1.2, label="qddot_max")
        plt.axhline(-qddot_max[j], color="k", linestyle="--", linewidth=1.2)
        plt.xlabel("Time t")
        plt.ylabel(f"qddot{j} [rad/s^2]")
        plt.title(f"Joint acceleration qddot{j}(t)")
        plt.legend(loc="best")

    if args.save_plots_dir is not None:
        args.save_plots_dir.mkdir(parents=True, exist_ok=True)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            fig.savefig(args.save_plots_dir / f"jointspace_altering_{fignum:03d}.png", dpi=160, bbox_inches="tight")
        print(f"Saved plots to: {args.save_plots_dir}")
    if args.no_plot or args.save_plots_dir is not None:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
