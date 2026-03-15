import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

# ============================================================
# How to run:
#   mjpython Precheck_TaskJoint.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_filtered.csv
#
#   With explicit bounds:
#   mjpython Precheck_TaskJoint.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#       --center 0.36 -0.27 0.46 --x_m 0.45 --v_m 1.15 --a_m 13.0 --gamma_p 10.0 \
#       --q-min -2.8 -1.7 -2.8 -3.0 -2.8 0.5 -2.8 \
#       --q-max  2.8  1.7  2.8 -0.1  2.8 3.8  2.8
#
# Recommended order:
#   1) Precheck_TaskJoint.py   (diagnose raw CSV)
#   2) JointSpace_Altering.py  (generate feasible altered trajectory)
#   3) JointSpace_Check.py     (verify altered output)
# ============================================================


SCENE_WITH_GRIPPER = "scene.xml"
SCENE_NO_GRIPPER = "scene_nohand.xml"
ARM_DOFS = 7

# Franka Panda nominal joint dynamic limits (rad/s, rad/s^2)
# Used as defaults for precheck unless user overrides.
FRANKA_QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=float)
FRANKA_QDDOT_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)


def damped_pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(J.shape[0]))


def _is_float_token(token):
    try:
        float(token)
        return True
    except Exception:
        return False


def load_task_csv(path, dt_fallback):
    times = []
    points = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) > 0]

    if len(rows) < 2:
        raise ValueError("CSV must contain at least 2 rows.")

    first = [c.strip() for c in rows[0]]
    has_header = not all(_is_float_token(c) for c in first)

    if has_header:
        header = [h.strip().lower() for h in first]
        data_rows = rows[1:]

        def find_col(cands):
            for c in cands:
                if c in header:
                    return header.index(c)
            return None

        ix = find_col(["x", "px", "pos_x", "position_x", "o_t_ee[12]"])
        iy = find_col(["y", "py", "pos_y", "position_y", "o_t_ee[13]"])
        iz = find_col(["z", "pz", "pos_z", "position_z", "o_t_ee[14]"])
        it = find_col(["t", "time", "timestamp"])
        if ix is None or iy is None or iz is None:
            raise ValueError("Header must contain x/y/z-like columns.")

        for i, row in enumerate(data_rows):
            if max(ix, iy, iz) >= len(row):
                raise ValueError(f"Row {i+2} does not have enough columns.")
            points.append([float(row[ix]), float(row[iy]), float(row[iz])])
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
    x = np.asarray(points, dtype=float)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time must be strictly increasing.")
    return t, x


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


def parse_scalar_or_7(vals, name):
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr.item(), ARM_DOFS)
    if arr.size == ARM_DOFS:
        return arr
    raise ValueError(f"{name} must have either 1 value or {ARM_DOFS} values.")


def main():
    parser = argparse.ArgumentParser(
        description="Precheck raw task-space CSV for task-space and IK-induced joint-space issues (before altering)."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Raw task-space CSV.")
    parser.add_argument("--no-gripper", action="store_true", help="Use scene_nohand.xml.")
    parser.add_argument("--dt_fallback", type=float, default=0.001)
    parser.add_argument("--center", nargs=3, type=float, default=[0.36, -0.27, 0.46], metavar=("CX", "CY", "CZ"))
    parser.add_argument("--x_m", type=float, default=0.45)
    parser.add_argument("--v_m", type=float, default=1.15)
    parser.add_argument("--a_m", type=float, default=13.0)
    parser.add_argument("--gamma_p", type=float, default=10.0)
    parser.add_argument("--lambda_dls", type=float, default=2e-2)
    parser.add_argument(
        "--qdot_max",
        nargs="+",
        type=float,
        default=None,
        help="Joint velocity limits. Provide 1 value (applied to all) or 7 values.",
    )
    parser.add_argument(
        "--qddot_max",
        nargs="+",
        type=float,
        default=None,
        help="Joint acceleration limits. Provide 1 value (applied to all) or 7 values.",
    )
    parser.add_argument("--q-min", nargs="+", type=float, default=None, help="Joint lower bounds override (1 or 7 values).")
    parser.add_argument("--q-max", nargs="+", type=float, default=None, help="Joint upper bounds override (1 or 7 values).")
    parser.add_argument("--joint_margin", type=float, default=0.02)
    parser.add_argument("--ik-seed", nargs=7, type=float, default=None)
    args = parser.parse_args()

    t, x_ref = load_task_csv(args.csv, args.dt_fallback)
    dt = np.gradient(t)
    v_ref = np.gradient(x_ref, t, axis=0)
    a_ref = np.gradient(v_ref, t, axis=0)

    c = np.asarray(args.center, dtype=float)
    r = np.linalg.norm(x_ref - c[None, :], axis=1)
    vn = np.linalg.norm(v_ref, axis=1)
    an = np.linalg.norm(a_ref, axis=1)
    h = args.x_m * args.x_m - np.sum((x_ref - c[None, :]) ** 2, axis=1)
    hdot = -2.0 * np.sum((x_ref - c[None, :]) * v_ref, axis=1)
    cbf = hdot + args.gamma_p * h

    xml_path = SCENE_NO_GRIPPER if args.no_gripper else SCENE_WITH_GRIPPER
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ee_id = get_ee_body_id(model)
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
    elif len(args.qdot_max) == 1:
        qdot_max = np.full(ARM_DOFS, float(args.qdot_max[0]), dtype=float)
    elif len(args.qdot_max) == ARM_DOFS:
        qdot_max = np.asarray(args.qdot_max, dtype=float)
    else:
        raise ValueError("--qdot_max must have either 1 value or 7 values.")

    if args.qddot_max is None:
        qddot_max = FRANKA_QDDOT_MAX.copy()
    elif len(args.qddot_max) == 1:
        qddot_max = np.full(ARM_DOFS, float(args.qddot_max[0]), dtype=float)
    elif len(args.qddot_max) == ARM_DOFS:
        qddot_max = np.asarray(args.qddot_max, dtype=float)
    else:
        raise ValueError("--qddot_max must have either 1 value or 7 values.")

    if args.ik_seed is not None:
        q = np.asarray(args.ik_seed, dtype=float).copy()
    else:
        q = data.qpos[:ARM_DOFS].copy()
    q = np.clip(q, q_min, q_max)
    data.qpos[:ARM_DOFS] = q
    mujoco.mj_forward(model, data)

    q_hist = np.zeros((len(t), ARM_DOFS))
    qdot_hist = np.zeros((len(t), ARM_DOFS))
    qdot_prev = np.zeros(ARM_DOFS)

    # IK rollout WITHOUT feasibility filtering to reveal baseline issues.
    for k in range(len(t)):
        mujoco.mj_forward(model, data)
        x = data.xpos[ee_id].copy()
        if k < len(t) - 1:
            dtk = max(t[k + 1] - t[k], 1e-6)
            v_ff = (x_ref[k + 1] - x_ref[k]) / dtk
        else:
            dtk = max(dt[k], 1e-6)
            v_ff = np.zeros(3)
        v_des = v_ff + (x_ref[k] - x)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
        J = jacp[:, :ARM_DOFS]
        qdot = damped_pinv(J, args.lambda_dls) @ v_des

        q = q + qdot * dtk
        data.qpos[:ARM_DOFS] = q
        q_hist[k] = q
        qdot_hist[k] = qdot
        qdot_prev = qdot

    qddot_hist = np.gradient(qdot_hist, t, axis=0)
    q_lo = np.where(np.isfinite(q_min), q_min + args.joint_margin, -np.inf)
    q_hi = np.where(np.isfinite(q_max), q_max - args.joint_margin, np.inf)

    q_ok = np.all(q_hist >= q_lo - 1e-9) and np.all(q_hist <= q_hi + 1e-9)
    qdot_abs_max = np.max(np.abs(qdot_hist), axis=0)
    qddot_abs_max = np.max(np.abs(qddot_hist), axis=0)
    qdot_ok = bool(np.all(qdot_abs_max <= qdot_max + 1e-9))
    qddot_ok = bool(np.all(qddot_abs_max <= qddot_max + 1e-9))

    print("\n=== PRECHECK: RAW TASK CSV (BEFORE ALTERING) ===")
    print(f"CSV: {args.csv}")
    print(f"samples: {len(t)} | duration: {t[-1]-t[0]:.6f}s")
    print("\nTask-space (raw reference):")
    print(f"  pos bound: {'PASS' if np.max(r) <= args.x_m else 'FAIL'} | max ||x-c||={np.max(r):.6f} (X_m={args.x_m:.6f})")
    print(f"  vel bound: {'PASS' if np.max(vn) <= args.v_m else 'FAIL'} | max ||v||={np.max(vn):.6f} (V_m={args.v_m:.6f})")
    print(f"  acc bound: {'PASS' if np.max(an) <= args.a_m else 'FAIL'} | max ||a||={np.max(an):.6f} (A_m={args.a_m:.6f})")
    print(f"  CBF cond: {'PASS' if np.min(cbf) >= -1e-8 else 'FAIL'} | min(hdot+gamma*h)={np.min(cbf):.6e}")
    print("\nJoint-space (baseline IK rollout, no filtering):")
    print(f"  q bounds: {'PASS' if q_ok else 'FAIL'}")
    print(
        "  qdot max: "
        f"{'PASS' if qdot_ok else 'FAIL'} | observed per joint {np.array2string(qdot_abs_max, precision=4)} "
        f"(limits {np.array2string(qdot_max, precision=4)})"
    )
    print(
        "  qddot max: "
        f"{'PASS' if qddot_ok else 'FAIL'} | observed per joint {np.array2string(qddot_abs_max, precision=4)} "
        f"(limits {np.array2string(qddot_max, precision=4)})"
    )


if __name__ == "__main__":
    main()
