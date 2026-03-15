import time
import csv
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
# Run:
#   mjpython Franka_sim.py
#   mjpython Franka_sim.py --csv your_trajectory.csv
#   mjpython Franka_sim.py --csv your_trajectory.csv --ik-seed q0 q1 q2 q3 q4 q5 q6
#   mjpython Franka_sim.py --csv your_trajectory.csv --no-gripper
#   mjpython Franka_sim.py --csv track.csv --plot-csv plot_only.csv
# ============================================================

XML_PATH_WITH_GRIPPER = "scene.xml"
XML_PATH_NO_GRIPPER = "scene_nohand.xml"

# ---------- PERFORMANCE SWITCH ----------
real_time = False
viewer_skip = 2
SIM_RATE_HZ = 1000.0
CHECK_HZ = 100.0

# ============================================================
# VISUAL SPHERE (must exist in scene.xml)
# ============================================================
SPHERE_GEOM_NAME = "task_sphere_geom"

# ============================================================
# CHECKS
# ============================================================
Z_MIN = 0.0
CHECK_CENTER = np.array([0.36, -0.27, 0.46], dtype=float)
X_M = 0.43
SPHERE_COLOR_TOL = 2e-3
V_M = 1.15
A_M = 13.0

# ============================================================
# TRAJECTORY VISUALIZATION
# ============================================================
CSV_PLOT_MAX_POINTS = 800
CSV_PLOT_POINT_SIZE = 0.001
CSV_MAIN_PLOT_RGBA = np.array([0.2, 0.6, 1.0, 0.9], dtype=float)   # main --csv trajectory
CSV_AUX_PLOT_RGBA = np.array([1.0, 0.7, 0.1, 0.9], dtype=float)    # --plot-csv trajectory
TRACE_EVERY_N_STEPS = 5
TRACE_POINT_SIZE = 0.001
TRACE_RGBA = np.array([1.0, 0.2, 0.2, 0.9], dtype=float)      # executed trajectory

# ============================================================
# IK / TRACKING
# ============================================================
QDOT_MAX = 1.0
LAMBDA_DLS = 2e-2
K_TRACK = 1.0
JOINT_LIMIT_MARGIN = 0.02  # rad

# ============================================================
# CSV TRAJECTORY SETTINGS
# ============================================================
TRAJECTORY_CSV_PATH = "ee_trajectory.csv"  # CSV with columns x,y,z (optional t/time)
CSV_DT_FALLBACK = 1.0 / SIM_RATE_HZ        # used when CSV has no time column


# ============================================================
# Helpers
# ============================================================

def clamp_norm(v, vmax):
    n = np.linalg.norm(v)
    if n > vmax and n > 1e-12:
        return v * (vmax / n)
    return v

def damped_pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))

def get_arm_joint_limits(model, arm_dofs):
    q_min = -np.inf * np.ones(arm_dofs)
    q_max = np.inf * np.ones(arm_dofs)
    for j in range(arm_dofs):
        if model.jnt_limited[j]:
            q_min[j] = model.jnt_range[j, 0]
            q_max[j] = model.jnt_range[j, 1]
    return q_min, q_max

def filter_joint_space_qdot(q, qdot, dt, q_min, q_max, margin):
    lo = np.where(np.isfinite(q_min), q_min + margin, -np.inf)
    hi = np.where(np.isfinite(q_max), q_max - margin, np.inf)

    # If margin makes interval invalid, use hard limits.
    bad = lo > hi
    lo = np.where(bad, q_min, lo)
    hi = np.where(bad, q_max, hi)

    qdot_lo = (lo - q) / dt
    qdot_hi = (hi - q) / dt
    return np.clip(qdot, qdot_lo, qdot_hi)

def preposition_ee_to_point(model, data, ee_body_id, arm_dofs, p_target, iters=2000, tol=1e-5):
    """
    Solve a short position-only IK phase so simulation starts at first CSV point.
    """
    err_norm = np.inf
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        p = data.xpos[ee_body_id].copy()
        err = p_target - p
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        Jp = jacp[:, :arm_dofs]

        dq = damped_pinv(Jp, 1e-2) @ (5.0 * err)
        dq = np.clip(dq, -0.05, 0.05)
        data.qpos[:arm_dofs] += dq

        # Clamp to joint limits.
        for j in range(arm_dofs):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)

    mujoco.mj_forward(model, data)
    p_end = data.xpos[ee_body_id].copy()
    return float(np.linalg.norm(p_target - p_end))

def _get_float(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return float(row[k])
    raise ValueError(f"Missing one of required columns: {keys}")

def _is_float_token(token):
    try:
        float(token)
        return True
    except Exception:
        return False

def load_trajectory_csv(path, dt_fallback):
    times = []
    points = []
    q_points = []

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

        # Prefer executed columns when both ref and exec exist in the same CSV.
        ix = find_col(["x_exec", "x_ref", "x", "px", "pos_x", "position_x", "o_t_ee[12]"])
        iy = find_col(["y_exec", "y_ref", "y", "py", "pos_y", "position_y", "o_t_ee[13]"])
        iz = find_col(["z_exec", "z_ref", "z", "pz", "pos_z", "position_z", "o_t_ee[14]"])
        it = find_col(["t", "time", "timestamp"])
        iq = [find_col([f"q{k}"]) for k in range(7)]
        has_q = all(idx is not None for idx in iq)

        if ix is None or iy is None or iz is None:
            raise ValueError("Header must contain position columns (x/y/z or x_ref/y_ref/z_ref or x_exec/y_exec/z_exec).")

        for i, row in enumerate(data_rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+2} does not have enough columns.")
            x = float(row[ix])
            y = float(row[iy])
            z = float(row[iz])
            points.append([x, y, z])
            if has_q:
                if max(iq) >= len(row):
                    raise ValueError(f"Row {i+2} does not have enough columns for q0..q6.")
                q_points.append([float(row[j]) for j in iq])
            if it is not None and it < len(row):
                times.append(float(row[it]))
            else:
                times.append(i * dt_fallback)
    else:
        data_rows = rows
        ncols = len(first)

        if ncols >= 4:
            it, ix, iy, iz = 0, 1, 2, 3
        elif ncols >= 3:
            it, ix, iy, iz = None, 0, 1, 2
        else:
            raise ValueError("CSV without header must have at least 3 columns.")

        for i, row in enumerate(data_rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+1} does not have enough columns.")
            x = float(row[ix])
            y = float(row[iy])
            z = float(row[iz])
            points.append([x, y, z])
            if it is not None:
                times.append(float(row[it]))
            else:
                times.append(i * dt_fallback)

    if len(points) < 2:
        raise ValueError("CSV trajectory must contain at least 2 points.")

    t = np.asarray(times, dtype=float)
    p = np.asarray(points, dtype=float)
    q = np.asarray(q_points, dtype=float) if len(q_points) == len(points) and len(q_points) > 0 else None

    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time values in CSV must be strictly increasing.")

    return t, p, q

def sample_trajectory(t_query, t_arr, p_arr):
    if t_query <= t_arr[0]:
        return p_arr[0].copy(), np.zeros(3)
    if t_query >= t_arr[-1]:
        return p_arr[-1].copy(), np.zeros(3)

    idx = int(np.searchsorted(t_arr, t_query, side="right") - 1)
    idx = max(0, min(idx, len(t_arr) - 2))

    t0, t1 = t_arr[idx], t_arr[idx + 1]
    p0, p1 = p_arr[idx], p_arr[idx + 1]
    alpha = (t_query - t0) / (t1 - t0)
    p_des = (1.0 - alpha) * p0 + alpha * p1
    v_ff = (p1 - p0) / (t1 - t0)
    return p_des, v_ff

def sample_trajectory_vec(t_query, t_arr, x_arr):
    if t_query <= t_arr[0]:
        return x_arr[0].copy()
    if t_query >= t_arr[-1]:
        return x_arr[-1].copy()
    idx = int(np.searchsorted(t_arr, t_query, side="right") - 1)
    idx = max(0, min(idx, len(t_arr) - 2))
    t0, t1 = t_arr[idx], t_arr[idx + 1]
    x0, x1 = x_arr[idx], x_arr[idx + 1]
    alpha = (t_query - t0) / (t1 - t0)
    return (1.0 - alpha) * x0 + alpha * x1

def sample_points_for_plot(traj_p, max_points=CSV_PLOT_MAX_POINTS):
    n = traj_p.shape[0]
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, max_points).astype(int)
    return traj_p[indices]

def draw_points_user_scene(viewer, csv_main_points, csv_aux_points, trace_points):
    scn = viewer.user_scn
    if scn.maxgeom <= 0:
        return

    ngeom = 0
    mat = np.eye(3).reshape(-1)

    for p in csv_main_points:
        if ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([CSV_PLOT_POINT_SIZE, 0.0, 0.0], dtype=float),
            p,
            mat,
            CSV_MAIN_PLOT_RGBA,
        )
        ngeom += 1

    for p in csv_aux_points:
        if ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([CSV_PLOT_POINT_SIZE, 0.0, 0.0], dtype=float),
            p,
            mat,
            CSV_AUX_PLOT_RGBA,
        )
        ngeom += 1

    free_slots = max(0, scn.maxgeom - ngeom)
    if len(trace_points) <= free_slots:
        trace_to_draw = trace_points
    elif free_slots > 0:
        idx = np.linspace(0, len(trace_points) - 1, free_slots).astype(int)
        trace_to_draw = [trace_points[i] for i in idx]
    else:
        trace_to_draw = []

    for p in trace_to_draw:
        if ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([TRACE_POINT_SIZE, 0.0, 0.0], dtype=float),
            p,
            mat,
            TRACE_RGBA,
        )
        ngeom += 1

    scn.ngeom = ngeom

def apply_ik_seed(model, data, q_seed, arm_dofs):
    if q_seed is None:
        return
    q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
    if q_seed.size != arm_dofs:
        raise RuntimeError(f"IK seed must have {arm_dofs} values.")
    for j in range(arm_dofs):
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            q_seed[j] = np.clip(q_seed[j], lo, hi)
    data.qpos[:arm_dofs] = q_seed
    mujoco.mj_forward(model, data)


# ============================================================
# MAIN
# ============================================================

def main(csv_path, ik_seed=None, no_gripper=False, plot_csv_path=None):
    xml_path = XML_PATH_NO_GRIPPER if no_gripper else XML_PATH_WITH_GRIPPER
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1.0 / SIM_RATE_HZ
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep

    # End-effector body (depends on whether gripper is present).
    ee_candidates = ["hand", "attachment", "link7"]
    ee_body_id = -1
    for name in ee_candidates:
        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if ee_body_id != -1:
            break
    if ee_body_id == -1:
        raise RuntimeError("End-effector body not found (tried: hand, attachment, link7).")

    # Sphere geom
    sphere_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SPHERE_GEOM_NAME)
    if sphere_geom_id == -1:
        raise RuntimeError(
            f"geom '{SPHERE_GEOM_NAME}' not found.\n"
            "Fix: add in scene.xml inside <worldbody>:\n"
            "<geom name=\"task_sphere_geom\" type=\"sphere\" pos=\"...\" size=\"...\" rgba=\"0 1 0 0.10\" contype=\"0\" conaffinity=\"0\"/>"
        )

    arm_dofs = 7
    q_min, q_max = get_arm_joint_limits(model, arm_dofs)

    apply_ik_seed(model, data, ik_seed, arm_dofs)

    model.geom_pos[sphere_geom_id] = CHECK_CENTER.copy()
    model.geom_size[sphere_geom_id][0] = X_M
    model.geom_rgba[sphere_geom_id] = np.array([0.0, 1.0, 0.0, 0.50])

    try:
        traj_t, traj_p, traj_q = load_trajectory_csv(csv_path, CSV_DT_FALLBACK)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CSV trajectory '{csv_path}'. "
            "Supported formats: header with x/y/z, x_ref/y_ref/z_ref, or x_exec/y_exec/z_exec (optional t/time), "
            "or no-header numeric rows as [x,y,z,...] or [t,x,y,z,...]."
        ) from exc
    traj_duration = float(traj_t[-1] - traj_t[0])
    joint_playback_mode = traj_q is not None and traj_q.shape == (len(traj_t), arm_dofs)

    print(
        f"Loaded trajectory from '{csv_path}': "
        f"{len(traj_p)} points, duration={traj_duration:.3f}s"
    )
    if joint_playback_mode:
        print("Control mode: joint playback from CSV q0..q6")
    else:
        print("Control mode: IK tracking from task-space positions")
    csv_main_plot_points = sample_points_for_plot(traj_p)
    csv_aux_plot_points = []

    if plot_csv_path is not None:
        try:
            _, plot_traj_p, _ = load_trajectory_csv(plot_csv_path, CSV_DT_FALLBACK)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load plot CSV '{plot_csv_path}'. "
                "Supported formats: header x,y,z (optional t/time) or no-header numeric rows as [x,y,z,...] or [t,x,y,z,...]."
            ) from exc
        print(f"Loaded plot-only trajectory from '{plot_csv_path}': {len(plot_traj_p)} points")
        csv_aux_plot_points = sample_points_for_plot(plot_traj_p)

    if joint_playback_mode:
        q_des = np.clip(traj_q[0].copy(), q_min, q_max)
        data.qpos[:arm_dofs] = q_des
        data.ctrl[:arm_dofs] = q_des
        mujoco.mj_forward(model, data)
    else:
        p_start = traj_p[0].copy()
        start_err = preposition_ee_to_point(model, data, ee_body_id, arm_dofs, p_start)
        if start_err > 2e-3:
            raise RuntimeError(
                f"Failed to start at first trajectory point. "
                f"Final startup EE error = {start_err:.6f} m. "
                "Try providing --ik-seed closer to the first point."
            )
        q_des = data.qpos[:arm_dofs].copy()

    step_counter = 0
    check_every = max(1, int(SIM_RATE_HZ / CHECK_HZ))
    p_prev = data.xpos[ee_body_id].copy()
    v_prev = np.zeros(3)
    trace_points = []
    finished = False
    finish_reported = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_traj = data.time + traj_t[0]
            if (not finished) and (t_traj >= traj_t[-1]):
                finished = True
                if not finish_reported:
                    print(f"Trajectory completed at sim time {data.time:.3f}s. Viewer stays open with trace.")
                    finish_reported = True

            # EE pose
            p = data.xpos[ee_body_id].copy()

            # =============================
            # Desired position: CSV trajectory playback
            # =============================
            p_des, v_ff = sample_trajectory(min(t_traj, traj_t[-1]), traj_t, traj_p)
            xm = p_des.copy()

            d_ref = float(np.linalg.norm(xm - CHECK_CENTER))
            d_exec = float(np.linalg.norm(p - CHECK_CENTER))
            # Avoid false red on tiny numeric overshoot and tracking lag.
            if (d_ref > X_M + SPHERE_COLOR_TOL) and (d_exec > X_M + SPHERE_COLOR_TOL):
                model.geom_rgba[sphere_geom_id] = np.array([1.0, 0.0, 0.0, 0.50])
            else:
                model.geom_rgba[sphere_geom_id] = np.array([0.0, 1.0, 0.0, 0.50])

            if joint_playback_mode:
                q_cmd = sample_trajectory_vec(min(t_traj, traj_t[-1]), traj_t, traj_q)
                q_cmd = np.clip(q_cmd, q_min, q_max)
                qdot = (q_cmd - q_des) / dt
                qdot = np.clip(qdot, -QDOT_MAX, QDOT_MAX)
                qdot = filter_joint_space_qdot(q_des, qdot, dt, q_min, q_max, JOINT_LIMIT_MARGIN)
                q_des = q_des + qdot * dt
                q_des = np.clip(q_des, q_min, q_max)
            else:
                if finished:
                    v_nom = np.zeros(3)
                else:
                    v_nom = v_ff + K_TRACK * (p_des - p)
                v_nom = clamp_norm(v_nom, V_M)

                # =============================
                # Jacobians
                # =============================
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
                Jp = jacp[:, :arm_dofs]

                # =============================
                # IK solve (position-only)
                # =============================
                J_pinv = damped_pinv(Jp, LAMBDA_DLS)
                qdot = J_pinv @ v_nom
                qdot = np.clip(qdot, -QDOT_MAX, QDOT_MAX)
                qdot = filter_joint_space_qdot(q_des, qdot, dt, q_min, q_max, JOINT_LIMIT_MARGIN)

                # Integrate joint positions
                q_des += qdot * dt
                q_des = np.clip(q_des, q_min, q_max)

            data.ctrl[:arm_dofs] = q_des

            v_ee = (p - p_prev) / dt
            a_ee = (v_ee - v_prev) / dt

            if step_counter % check_every == 0:
                err = float(np.linalg.norm(p_des - p))
                d = float(np.linalg.norm(p - CHECK_CENTER))
                in_sphere = d <= X_M
                above_floor = p[2] >= Z_MIN
                speed_ok = float(np.linalg.norm(v_ee)) <= V_M
                accel_ok = float(np.linalg.norm(a_ee)) <= A_M
                joint_ok = bool(np.all(q_des >= q_min - 1e-9) and np.all(q_des <= q_max + 1e-9))
                print(
                    f"[CHECK] t={data.time:.3f}s err={err:.4f} "
                    f"dist={d:.4f}/{X_M:.4f} in_sphere={in_sphere} "
                    f"v_ok={speed_ok} a_ok={accel_ok} z_ok={above_floor} joint_ok={joint_ok}"
                )
            if (not finished) and (step_counter % TRACE_EVERY_N_STEPS == 0):
                trace_points.append(p.copy())

            if not finished:
                mujoco.mj_step(model, data)
                p_prev = p.copy()
                v_prev = v_ee.copy()

            if step_counter % viewer_skip == 0:
                draw_points_user_scene(viewer, csv_main_plot_points, csv_aux_plot_points, trace_points)
                viewer.sync()

            if real_time:
                time.sleep(dt)

            step_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka EE trajectory tracking from CSV.")
    parser.add_argument(
        "--csv",
        default=TRAJECTORY_CSV_PATH,
        help="Path to trajectory CSV with columns x,y,z and optional t/time."
    )
    parser.add_argument(
        "--ik-seed",
        nargs=7,
        type=float,
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Initial IK seed q[0..6] used before solving IK to the first CSV point."
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Run with no-gripper model (uses scene_nohand.xml)."
    )
    parser.add_argument(
        "--plot-csv",
        default=None,
        help="Optional second CSV used only for plotting (does not affect control)."
    )
    args = parser.parse_args()
    main(
        args.csv,
        ik_seed=args.ik_seed,
        no_gripper=args.no_gripper,
        plot_csv_path=args.plot_csv,
    )
