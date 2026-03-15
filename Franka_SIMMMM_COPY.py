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
# ============================================================

XML_PATH = "scene.xml"

# ---------- PERFORMANCE SWITCH ----------
real_time = False
viewer_skip = 2
SIM_RATE_HZ = 1000.0

# ============================================================
# VISUAL SPHERE (must exist in scene.xml)
# ============================================================
SPHERE_GEOM_NAME = "task_sphere_geom"

# "On-boundary" band thickness (meters)
BOUNDARY_BAND = 0.02

# ============================================================
# FLOOR SAFETY (prevents hitting floor)
# ============================================================
Z_MIN = 0.10
GAMMA_FLOOR = 6.0

# ============================================================
# SPEED / ACCEL (slow + smooth)
# ============================================================
V_m = 0.11
A_m = 0.20
GAMMA_SPHERE = 6.0

# ============================================================
# ORIENTATION HOLD (reduces twisting)
# ============================================================
K_ORI = 1.5
W_ORI = 0.35
W_MAX = 0.5

# ============================================================
# JOINT SPACE (stability)
# ============================================================
QDOT_MAX = 1.0
LAMBDA_DLS = 2e-2

K_POSTURE = 1.5
Q_HOME = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=float)

# ============================================================
# CSV TRAJECTORY SETTINGS
# ============================================================
TRAJECTORY_CSV_PATH = "ee_trajectory.csv"  # CSV with columns x,y,z (optional t/time)
CSV_DT_FALLBACK = 1.0 / SIM_RATE_HZ        # used when CSV has no time column
START_DELAY = 0.0      # trajectory starts immediately
PREP_TO_START_TIME = 0.0  # not needed (we pre-position to first CSV point)
K_TRACK = 1.0            # position tracking gain

# Trace visualization
TRACE_MAX = 250
TRACE_POINT_SIZE = 0.004
TRACE_ALPHA = 0.9
TRACE_EVERY_N_STEPS = 4


# ============================================================
# Helpers
# ============================================================

def clamp_norm(v, vmax):
    n = np.linalg.norm(v)
    if n > vmax and n > 1e-12:
        return v * (vmax / n)
    return v

def enforce_accel_step(v, v_prev, dt, A_m):
    dv = v - v_prev
    max_step = A_m * dt
    n = np.linalg.norm(dv)
    if n > max_step and n > 1e-12:
        return v_prev + dv * (max_step / n)
    return v

def enforce_floor_cbf(v, p, z_min, gamma):
    h = p[2] - z_min
    v2 = v.copy()
    v2[2] = max(v2[2], -gamma * h)
    return v2

def enforce_sphere_cbf(v, p, c, R, gamma):
    """
    Sphere CBF:
      h(p) = R^2 - ||p-c||^2
      hdot = -2(p-c)^T v
      CBF:  hdot >= -gamma h
      => (p-c)^T v <= (gamma/2) h

    Enforce by projection onto halfspace a^T v <= b.
    """
    a = (p - c).reshape(3)
    a_norm2 = float(a @ a)
    if a_norm2 < 1e-12:
        return v

    h = R*R - a_norm2
    b = 0.5 * gamma * h

    av = float(a @ v)
    if av <= b:
        return v

    return v - ((av - b) / a_norm2) * a

def damped_pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))

def preposition_ee_to_point(model, data, ee_body_id, arm_dofs, p_target, iters=350):
    """
    Solve a short position-only IK phase so simulation starts at first CSV point.
    """
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        p = data.xpos[ee_body_id].copy()
        err = p_target - p
        if np.linalg.norm(err) < 1e-4:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        Jp = jacp[:, :arm_dofs]

        dq = damped_pinv(Jp, 1e-2) @ (2.0 * err)
        dq = np.clip(dq, -0.02, 0.02)
        data.qpos[:arm_dofs] += dq

        # Clamp to joint limits.
        for j in range(arm_dofs):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)

    mujoco.mj_forward(model, data)

def prompt_ik_seed(default_q):
    """
    Ask user for initial IK seed q[0..6]. Empty input keeps defaults.
    Accepts comma-separated or space-separated 7 values.
    """
    msg = (
        "Enter IK seed q[0..6] (7 values, comma/space separated),\n"
        "or press Enter to use default: "
    )
    raw = input(msg).strip()
    if raw == "":
        return default_q.copy()

    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    tokens = [t for t in cleaned.split() if t]
    if len(tokens) != 7:
        raise ValueError(f"IK seed must have 7 values, got {len(tokens)}.")
    return np.asarray([float(t) for t in tokens], dtype=float)

def min_jerk_profile(t, T):
    """
    Quintic time-scaling for smooth point-to-point motion.
    Returns:
      s(t)  in [0,1]  : position scale
      sd(t) in [1/s]  : velocity scale
    """
    if T <= 1e-9:
        return 1.0, 0.0
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    sd = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T
    return s, sd

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

        ix = find_col(["x"])
        iy = find_col(["y"])
        iz = find_col(["z"])
        it = find_col(["t", "time", "timestamp"])

        if ix is None or iy is None or iz is None:
            raise ValueError("Header must contain x,y,z columns.")

        for i, row in enumerate(data_rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+2} does not have enough columns.")
            x = float(row[ix])
            y = float(row[iy])
            z = float(row[iz])
            points.append([x, y, z])
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

    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time values in CSV must be strictly increasing.")

    return t, p

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

def add_trace_marker(model, pos, rgba, size=0.004):
    """
    Adds a visual-only site marker at a position.
    Note: sites are cheap and visible. We add many over time for a trail.
    """
    # MuJoCo requires unique names. We'll create a site with an incrementing id.
    name = f"trace_{model.nsite}"
    mujoco.mj_addSite(model, name.encode('utf-8'))
    sid = model.nsite - 1
    model.site_type[sid] = mujoco.mjtGeom.mjGEOM_SPHERE
    model.site_size[sid][0] = size
    model.site_pos[sid][:] = pos
    model.site_rgba[sid][:] = rgba
    # make it visual only
    model.site_group[sid] = 5
    return sid


# ============================================================
# MAIN
# ============================================================

def main(csv_path, ik_seed=None):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = 1.0 / SIM_RATE_HZ
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep

    # End-effector body
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if ee_body_id == -1:
        raise RuntimeError("End-effector body 'hand' not found.")

    # Sphere geom
    sphere_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SPHERE_GEOM_NAME)
    if sphere_geom_id == -1:
        raise RuntimeError(
            f"geom '{SPHERE_GEOM_NAME}' not found.\n"
            "Fix: add in scene.xml inside <worldbody>:\n"
            "<geom name=\"task_sphere_geom\" type=\"sphere\" pos=\"...\" size=\"...\" rgba=\"0 1 0 0.10\" contype=\"0\" conaffinity=\"0\"/>"
        )

    arm_dofs = 7

    if ik_seed is None:
        try:
            ik_seed = prompt_ik_seed(data.qpos[:arm_dofs].copy())
        except EOFError:
            ik_seed = data.qpos[:arm_dofs].copy()
            print("No stdin available. Using default IK seed.")

    ik_seed = np.asarray(ik_seed, dtype=float).reshape(-1)
    if ik_seed.size != arm_dofs:
        raise RuntimeError(f"IK seed must have {arm_dofs} values.")

    # Clamp seed to joint limits.
    for j in range(arm_dofs):
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            ik_seed[j] = np.clip(ik_seed[j], lo, hi)
    data.qpos[:arm_dofs] = ik_seed
    mujoco.mj_forward(model, data)

    # Hold initial orientation
    R0 = data.xmat[ee_body_id].reshape(3, 3).copy()

    # Read sphere center + radius
    c = model.geom_pos[sphere_geom_id].copy()
    R_sphere = float(model.geom_size[sphere_geom_id][0])

    try:
        traj_t, traj_p = load_trajectory_csv(csv_path, CSV_DT_FALLBACK)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CSV trajectory '{csv_path}'. "
            "Supported formats: header x,y,z (optional t/time) or no-header numeric rows as [x,y,z,...] or [t,x,y,z,...]."
        ) from exc
    traj_duration = float(traj_t[-1] - traj_t[0])

    print(
        f"Loaded trajectory from '{csv_path}': "
        f"{len(traj_p)} points, duration={traj_duration:.3f}s"
    )

    p_start = traj_p[0].copy()
    preposition_ee_to_point(model, data, ee_body_id, arm_dofs, p_start)
    q_des = data.qpos[:arm_dofs].copy()

    v_prev = np.zeros(3)
    step_counter = 0

    # Executability reporting (don’t spam every step)
    print_every = int(0.5 / dt)  # ~2 Hz
    last_report_step = -10**9

    # Trace bookkeeping (we keep ids to delete old ones if needed)
    trace_site_ids = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # EE pose
            p = data.xpos[ee_body_id].copy()
            Rm = data.xmat[ee_body_id].reshape(3, 3).copy()

            # Read sphere center + radius (in case XML changed)
            c = model.geom_pos[sphere_geom_id].copy()
            R_sphere = float(model.geom_size[sphere_geom_id][0])

            # =============================
            # Desired position: CSV trajectory playback
            # =============================
            t_now = data.time
            if t_now < START_DELAY:
                p_des = p_start
                v_ff = np.zeros(3)
            else:
                t_after_delay = t_now - START_DELAY
                if t_after_delay < PREP_TO_START_TIME:
                    p_des = p_start
                    v_ff = np.zeros(3)
                else:
                    t_traj = (t_after_delay - PREP_TO_START_TIME) + traj_t[0]
                    p_des, v_ff = sample_trajectory(t_traj, traj_t, traj_p)

            # xm = commanded Cartesian point. Sphere color follows xm (not measured EE p).
            xm = p_des
            d_color = float(np.linalg.norm(xm - c))
            if d_color > R_sphere:
                zone = -1
            elif d_color >= (R_sphere - BOUNDARY_BAND):
                zone = 0
            else:
                zone = 1

            if zone == -1:
                model.geom_rgba[sphere_geom_id] = np.array([1.0, 0.0, 0.0, 0.10])
            elif zone == 0:
                model.geom_rgba[sphere_geom_id] = np.array([1.0, 1.0, 0.0, 0.10])
            else:
                model.geom_rgba[sphere_geom_id] = np.array([0.0, 1.0, 0.0, 0.10])

            # Keep safety checks based on measured EE state.
            d = float(np.linalg.norm(p - c))

            v_nom = v_ff + K_TRACK * (p_des - p)
            v_nom = clamp_norm(v_nom, V_m)

            # =============================
            # Executability (nominal only)
            # =============================
            reasons = []

            # inside sphere?
            if d > R_sphere:
                reasons.append("EE outside sphere")

            # floor?
            if p[2] < Z_MIN:
                reasons.append("EE below floor z_min")

            # speed?
            if np.linalg.norm(v_nom) > V_m + 1e-9:
                reasons.append("speed limit violated")

            # accel-step?
            if np.linalg.norm(v_nom - v_prev) > A_m*dt + 1e-9:
                reasons.append("accel-step violated")

            executable = (len(reasons) == 0)

            if step_counter - last_report_step >= print_every:
                last_report_step = step_counter
                if executable:
                    print(f"[OK] Executable nominal | d={d:.3f} R={R_sphere:.3f} | ||v_nom||={np.linalg.norm(v_nom):.3f}")
                else:
                    print(f"[NO] NOT executable nominal | d={d:.3f} R={R_sphere:.3f} | reasons: {', '.join(reasons)}")

            # =============================
            # Jacobians
            # =============================
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
            Jp = jacp[:, :arm_dofs]
            Jr = jacr[:, :arm_dofs]

            # =============================
            # FILTERING (CBF + floor + accel + speed)
            # =============================
            v_f = v_nom.copy()
            v_f = enforce_sphere_cbf(v_f, p, c, R_sphere, GAMMA_SPHERE)
            v_f = enforce_floor_cbf(v_f, p, Z_MIN, GAMMA_FLOOR)
            v_f = enforce_accel_step(v_f, v_prev, dt, A_m)
            v_f = clamp_norm(v_f, V_m)

            # =============================
            # Orientation hold
            # =============================
            R_err = R0.T @ Rm
            w = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
            w = -K_ORI * w
            w = clamp_norm(w, W_MAX)
            w = W_ORI * w

            # =============================
            # IK solve (6D)
            # =============================
            v6 = np.hstack([v_f, w])
            J6 = np.vstack([Jp, Jr])
            J_pinv = damped_pinv(J6, LAMBDA_DLS)

            qdot_task = J_pinv @ v6

            # Nullspace posture
            q = data.qpos[:arm_dofs]
            N = np.eye(arm_dofs) - J_pinv @ J6
            qdot = qdot_task + N @ (K_POSTURE * (Q_HOME - q))
            qdot = np.clip(qdot, -QDOT_MAX, QDOT_MAX)

            # Integrate joint positions
            q_des += qdot * dt
            data.ctrl[:arm_dofs] = q_des

            # =============================
            # Trajectory trace (filtered EE)
            # =============================
            if step_counter % TRACE_EVERY_N_STEPS == 0:
                # Color trace based on zone
                if zone == -1:
                    rgba = np.array([1.0, 0.0, 0.0, TRACE_ALPHA])
                elif zone == 0:
                    rgba = np.array([1.0, 1.0, 0.0, TRACE_ALPHA])
                else:
                    rgba = np.array([0.0, 1.0, 0.0, TRACE_ALPHA])

                try:
                    sid = add_trace_marker(model, p, rgba, size=TRACE_POINT_SIZE)
                    trace_site_ids.append(sid)

                    # limit trace length (simple: if too many, just stop adding more)
                    if len(trace_site_ids) > TRACE_MAX:
                        # no safe deletion API in runtime; so we just stop adding
                        pass
                except Exception:
                    # If mj_addSite is not supported in your version, ignore trace.
                    pass

            v_prev = v_f.copy()

            mujoco.mj_step(model, data)

            if step_counter % viewer_skip == 0:
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
        help="Initial IK seed q[0..6]. If omitted, script prompts at startup."
    )
    args = parser.parse_args()
    main(args.csv, ik_seed=args.ik_seed)
