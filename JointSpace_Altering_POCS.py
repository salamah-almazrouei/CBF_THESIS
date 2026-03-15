import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

# ============================================================
# How to run:
#   mjpython JointSpace_Altering_POCS.py \
#       --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#       --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv
#
#   With seed and bounds:
#   mjpython JointSpace_Altering_POCS.py \
#       --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#       --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#       --ik-seed 0.413164 -0.105627 -0.306975 -2.46688 -0.0954591 2.39035 -2.16711 \
#       --center 0.36 -0.27 0.46 --x_m 0.45 --gamma_p 10.0 \
#       --q-min -2.8 -1.7 -2.8 -3.0 -2.8 0.5 -2.8 \
#       --q-max  2.8  1.7  2.8 -0.1  2.8 3.8  2.8
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


def project_halfspace(x, a, b):
    viol = float(a @ x - b)
    if viol <= 0.0:
        return x
    a2 = float(a @ a)
    if a2 < 1e-12:
        return x
    return x - (viol / a2) * a


def project_qdot_to_task_velocity_ball(Jp, qdot, center_v, radius, reg=1e-8):
    """
    Project qdot so that ||Jp qdot - center_v|| <= radius.
    Uses minimum-norm correction in joint space.
    """
    v = Jp @ qdot
    dv = v - center_v
    n = float(np.linalg.norm(dv))
    if n <= radius + 1e-12:
        return qdot
    v_proj = center_v + (radius / max(n, 1e-12)) * dv
    delta_v = v_proj - v
    M = Jp @ Jp.T + reg * np.eye(Jp.shape[0])
    delta_q = Jp.T @ np.linalg.solve(M, delta_v)
    return qdot + delta_q


def solve_qdot_pocs(H, g, lb, ub, A_ineq, b_ineq, x0, Jp, v_prev_task, dt, v_m, a_m, iters=80):
    """
    POCS-based filtering:
      1) gradient step toward quadratic objective
      2) projection onto box set [lb, ub]
      3) cyclic projection onto each half-space a_i x <= b_i
      4) projection onto exact task-space speed/accel norm balls
    """
    x = np.clip(x0.copy(), lb, ub)
    L = float(np.linalg.norm(H, ord=2))
    alpha = 1.0 / max(L, 1e-6)
    for _ in range(iters):
        x = x - alpha * (H @ x - g)
        x = np.clip(x, lb, ub)
        for a, b in zip(A_ineq, b_ineq):
            x = project_halfspace(x, a, b)
            x = np.clip(x, lb, ub)
        # Exact speed bound: ||Jp x|| <= v_m
        x = project_qdot_to_task_velocity_ball(Jp, x, np.zeros(3), v_m)
        x = np.clip(x, lb, ub)
        # Exact accel bound: ||(Jp x - v_prev_task)/dt|| <= a_m
        x = project_qdot_to_task_velocity_ball(Jp, x, v_prev_task, a_m * dt)
        x = np.clip(x, lb, ub)
    return x


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
        description="Alter task-space trajectory using joint-space-feasible IK + POCS filtering."
    )
    parser.add_argument("--in_csv", type=Path, required=True, help="Input task-space CSV.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output altered CSV.")
    parser.add_argument("--no-gripper", action="store_true", help="Use scene_nohand.xml.")
    parser.add_argument("--dt_fallback", type=float, default=0.001, help="Fallback dt for CSV without time.")
    parser.add_argument("--k_track", type=float, default=1.0, help="Task tracking gain.")
    parser.add_argument("--k_ori", type=float, default=2.0, help="Orientation tracking gain (if orientation columns exist).")
    parser.add_argument("--w_max", type=float, default=1.0, help="Angular velocity clamp for orientation tracking.")
    parser.add_argument("--k_home", type=float, default=0.05, help="Nullspace home gain.")
    parser.add_argument("--k_limit", type=float, default=0.02, help="Nullspace limit-avoid gain.")
    parser.add_argument("--w_smooth", type=float, default=5e-2, help="Weight for qdot smoothness (closer to previous qdot).")
    parser.add_argument("--w_null", type=float, default=1e-2, help="Weight for secondary/nullspace objective.")
    parser.add_argument("--pocs-iters", type=int, default=500, help="Number of POCS projection iterations per step.")
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
    parser.add_argument("--v_m", type=float, default=1.15, help="Task-space speed bound (exact norm-ball projection in POCS).")
    parser.add_argument("--a_m", type=float, default=13.0, help="Task-space accel bound (exact norm-ball projection in POCS).")
    parser.add_argument("--gamma_p", type=float, default=10.0, help="Task-space position CBF gain.")
    parser.add_argument("--gamma_v", type=float, default=10.0, help="Task-space velocity CBF gain.")
    parser.add_argument("--ik-seed", nargs=7, type=float, default=None, help="Initial q seed q0..q6.")
    parser.add_argument(
        "--q-home",
        nargs=7,
        type=float,
        default=[0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8],
        help="Home posture for nullspace.",
    )
    args = parser.parse_args()

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

    t_ref, x_ref, R_ref, q_ref = load_trajectory_csv(args.in_csv, args.dt_fallback)
    has_orientation_ref = R_ref is not None and len(R_ref) == len(x_ref)
    n = len(t_ref)

    q = data.qpos[:ARM_DOFS].copy()
    if args.ik_seed is not None:
        q = np.asarray(args.ik_seed, dtype=float)
    q = np.clip(q, q_min, q_max)
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
    data.qpos[:ARM_DOFS] = q
    data.ctrl[:ARM_DOFS] = q
    mujoco.mj_forward(model, data)
    if has_orientation_ref:
        print(f"Start alignment | pos_err={start_pos_err:.6e} m, ori_err={start_ori_err:.6e} rad")
    else:
        print(f"Start alignment | pos_err={start_pos_err:.6e} m")
    print("Solver backend: POCS (Projection Onto Convex Sets)")

    q_hist = np.zeros((n, ARM_DOFS))
    qdot_hist = np.zeros((n, ARM_DOFS))
    x_exec = np.zeros((n, 3))
    R_exec = np.zeros((n, 3, 3))
    err = np.zeros(n)
    ori_err = np.full(n, np.nan)
    cbf_p_res = np.zeros(n)
    qdot_prev = np.zeros(ARM_DOFS)

    for i in range(n):
        dt = t_ref[i] - t_ref[i - 1] if i > 0 else (t_ref[1] - t_ref[0])
        dt = max(dt, 1e-6)

        mujoco.mj_forward(model, data)
        x = data.xpos[ee_body_id].copy()
        R = data.xmat[ee_body_id].reshape(3, 3).copy()
        q_hist[i] = q.copy()
        x_exec[i] = x
        R_exec[i] = R
        err[i] = np.linalg.norm(x_ref[i] - x)

        if i < n - 1:
            v_ff = (x_ref[i + 1] - x_ref[i]) / max(t_ref[i + 1] - t_ref[i], 1e-6)
        else:
            v_ff = np.zeros(3)
        v_des = v_ff + args.k_track * (x_ref[i] - x)
        h = args.x_m * args.x_m - float((x - c) @ (x - c))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        Jp = jacp[:, :ARM_DOFS]
        Jr = jacr[:, :ARM_DOFS]
        if has_orientation_ref:
            if i < n - 1:
                dt_ff = max(t_ref[i + 1] - t_ref[i], 1e-6)
                R_rel = R_ref[i].T @ R_ref[i + 1]
                w_ff = rot_log_vee(project_to_so3(R_rel)) / dt_ff
            else:
                w_ff = np.zeros(3)
            R_err = R_ref[i] @ R.T
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

        A_ineq = np.asarray(A_rows, dtype=float)
        b_ineq = np.asarray(b_rows, dtype=float)

        qdot = solve_qdot_pocs(
            H=H,
            g=g,
            lb=lb,
            ub=ub,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            x0=qdot_prev,
            Jp=Jp,
            v_prev_task=v_prev_task,
            dt=dt,
            v_m=args.v_m,
            a_m=args.a_m,
            iters=args.pocs_iters,
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
    arrays = [t_ref, x_ref, x_exec, v_exec, a_exec, err, cbf_p_res]
    if q_ref is not None:
        header += ["qx_ref", "qy_ref", "qz_ref", "qw_ref"]
        arrays.append(q_ref)
    if has_orientation_ref:
        header += [f"r{i}{j}_ref" for i in (1, 2, 3) for j in (1, 2, 3)]
        header += [f"r{i}{j}_exec" for i in (1, 2, 3) for j in (1, 2, 3)]
        header += ["ori_err"]
        arrays.append(R_ref.reshape(n, 9))
        arrays.append(R_exec.reshape(n, 9))
        arrays.append(ori_err.reshape(n, 1))
    arrays += [q_hist, qdot_hist]
    out = np.column_stack(arrays)
    np.savetxt(args.out_csv, out, delimiter=",", header=",".join(header), comments="")

    print(f"Saved altered trajectory to: {args.out_csv}")
    print(f"Samples: {n}")
    print(f"Tracking error mean/max: {np.mean(err):.6f} / {np.max(err):.6f} m")
    print(f"Task-space CBF residual min: {np.min(cbf_p_res):.6e}")
    if has_orientation_ref:
        valid = ~np.isnan(ori_err)
        print(f"Orientation err mean/max: {np.mean(ori_err[valid]):.6f} / {np.max(ori_err[valid]):.6f} rad")


if __name__ == "__main__":
    main()
