import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

# ============================================================
# How to run:
#   Recommended pipeline (compare altered vs nominal):
#   1) mjpython JointSpace_Altering.py \
#          --in_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv \
#          --out_csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#          --out_nominal_csv /Users/salamahalmazrouei/Desktop/test_joint_nominal.csv
#   2) mjpython JointSpace_Check.py \
#          --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#          --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_check_feasible_plots
#   3) mjpython JointSpace_Check.py \
#          --csv /Users/salamahalmazrouei/Desktop/test_joint_nominal.csv \
#          --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_check_nominal_plots
#
#   Optional pre-filter for noisy task references:
#   python3 CBF_Altering.py \
#       --in_csv /Users/salamahalmazrouei/Desktop/test.csv \
#       --out_csv /Users/salamahalmazrouei/Desktop/test_filtered.csv
#
#   Optional extra safety plots:
#   python3 Check_Safety.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv
#
#   Quick check only (no windows):
#   mjpython JointSpace_Check.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#       --no_plot
#
#   With explicit thresholds and per-joint dynamic limits:
#   mjpython JointSpace_Check.py \
#       --csv /Users/salamahalmazrouei/Desktop/test_joint_feasible.csv \
#       --center 0.36 -0.27 0.46 --x_m 0.45 --v_m 1.15 --a_m 13.0 \
#       --gamma_p 10.0 --tol_cbf 1e-4 --ori_err_max 0.10 \
#       --qdot_max 2.175 2.175 2.175 2.175 2.610 2.610 2.610 \
#       --qddot_max 15 7.5 10 12.5 15 20 20 \
#       --q-min -2.8 -1.7 -2.8 -3.0 -2.8 0.5 -2.8 \
#       --q-max  2.8  1.7  2.8 -0.1  2.8 3.8  2.8 \
#       --save_plots_dir /Users/salamahalmazrouei/Desktop/joint_check_plots
# ============================================================


SCENE_WITH_GRIPPER = "scene.xml"
SCENE_NO_GRIPPER = "scene_nohand.xml"
ARM_DOFS = 7
FRANKA_QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=float)
FRANKA_QDDOT_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)


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
    raise ValueError(f"{name} must be 1 value or {ARM_DOFS} values.")


def find_col(header, candidates):
    h = [x.strip().lower() for x in header]
    for c in candidates:
        if c.lower() in h:
            return h.index(c.lower())
    return None


def main():
    parser = argparse.ArgumentParser(description="Check joint-space feasibility of altered trajectory CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Altered CSV from JointSpace_Altering.py")
    parser.add_argument("--no-gripper", action="store_true", help="Use scene_nohand.xml for joint limits.")
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
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=[0.36, -0.27, 0.45],
        metavar=("CX", "CY", "CZ"),
        help="Task-space sphere center.",
    )
    parser.add_argument("--x_m", type=float, default=0.45, help="Task-space position bound radius.")
    parser.add_argument("--v_m", type=float, default=1.15, help="Task-space speed bound.")
    parser.add_argument("--a_m", type=float, default=13.0, help="Task-space acceleration bound.")
    parser.add_argument(
        "--tol_plot_outside",
        type=float,
        default=1e-3,
        help="Plot-only tolerance for outside highlighting (abs margin added to bounds).",
    )
    parser.add_argument("--gamma_p", type=float, default=10.0, help="Task-space position CBF gain.")
    parser.add_argument("--tol_cbf", type=float, default=1e-8, help="CBF tolerance.")
    parser.add_argument("--ori_err_max", type=float, default=0.10, help="Orientation error threshold [rad].")
    parser.add_argument("--no_plot", action="store_true", help="Disable interactive plotting.")
    parser.add_argument("--save_plots_dir", type=Path, default=None, help="Save plots to this directory (PNG).")
    args = parser.parse_args()
    if args.no_plot or args.save_plots_dir is not None:
        plt.switch_backend("Agg")
        plt.rcParams["figure.max_open_warning"] = 0

    xml_path = SCENE_NO_GRIPPER if args.no_gripper else SCENE_WITH_GRIPPER
    model = mujoco.MjModel.from_xml_path(xml_path)
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

    # Load with header.
    with open(args.csv, "r", newline="") as f:
        first = f.readline().strip()
    if "," not in first:
        raise ValueError("CSV must contain a header row.")
    header = [x.strip() for x in first.split(",")]
    M = np.genfromtxt(args.csv, delimiter=",", skip_header=1, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    M = M[~np.all(np.isnan(M), axis=1)]
    if M.shape[0] < 2:
        raise ValueError("Need at least 2 samples.")

    t_idx = find_col(header, ["t", "time", "timestamp"])
    if t_idx is None:
        raise ValueError("Missing time column.")
    t = M[:, t_idx]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing.")

    q_idx = [find_col(header, [f"q{i}"]) for i in range(ARM_DOFS)]
    if any(idx is None for idx in q_idx):
        raise ValueError("Missing q0..q6 columns.")
    q = M[:, q_idx]

    qdot_idx = [find_col(header, [f"qdot{i}"]) for i in range(ARM_DOFS)]
    if any(idx is None for idx in qdot_idx):
        qdot = np.gradient(q, t, axis=0)
    else:
        qdot = M[:, qdot_idx]
    qddot = np.gradient(qdot, t, axis=0)

    q_lo = np.where(np.isfinite(q_min), q_min + args.joint_margin, -np.inf)
    q_hi = np.where(np.isfinite(q_max), q_max - args.joint_margin, np.inf)

    q_ok = np.all(q >= q_lo - 1e-9) and np.all(q <= q_hi + 1e-9)
    qdot_ok = bool(np.all(np.abs(qdot) <= qdot_max[None, :] + 1e-9))
    qddot_ok = bool(np.all(np.abs(qddot) <= qddot_max[None, :] + 1e-9))

    min_margin = np.min(np.minimum(q - q_min, q_max - q))

    err_idx = find_col(header, ["err", "tracking_error"])
    if err_idx is not None:
        err = M[:, err_idx]
        err_msg = f"tracking err mean/max: {np.mean(err):.6f}/{np.max(err):.6f} m"
    else:
        err_msg = "tracking err: column not found"

    ori_idx = find_col(header, ["ori_err"])
    if ori_idx is not None:
        ori_err = M[:, ori_idx]
        ori_ok = np.nanmax(ori_err) <= args.ori_err_max + 1e-9
        ori_msg = (
            f"orientation: {'PASS' if ori_ok else 'FAIL'} "
            f"(mean/max={np.nanmean(ori_err):.6f}/{np.nanmax(ori_err):.6f} rad, "
            f"limit={args.ori_err_max:.6f})"
        )
    else:
        ori_msg = "orientation: skipped (no ori_err column)"

    # Task-space checks from executed trajectory.
    x = None
    r = None
    vn = None
    an = None
    h = None
    hdot = None
    cbf = None
    outside_p = None
    outside_v = None
    outside_a = None
    idxBp = np.array([], dtype=int)
    idxBv = np.array([], dtype=int)
    idxBa = np.array([], dtype=int)
    x_idx = find_col(header, ["x_exec", "x", "px", "pos_x", "position_x", "o_t_ee[12]"])
    y_idx = find_col(header, ["y_exec", "y", "py", "pos_y", "position_y", "o_t_ee[13]"])
    z_idx = find_col(header, ["z_exec", "z", "pz", "pos_z", "position_z", "o_t_ee[14]"])
    if x_idx is not None and y_idx is not None and z_idx is not None:
        x = M[:, [x_idx, y_idx, z_idx]]
        c = np.asarray(args.center, dtype=float)
        r = np.linalg.norm(x - c[None, :], axis=1)
        v = np.gradient(x, t, axis=0)
        a = np.gradient(v, t, axis=0)
        vn = np.linalg.norm(v, axis=1)
        an = np.linalg.norm(a, axis=1)
        pos_ok = np.max(r) <= args.x_m + 1e-9
        vel_ok = np.max(vn) <= args.v_m + 1e-9
        acc_ok = np.max(an) <= args.a_m + 1e-9
        h = args.x_m * args.x_m - np.sum((x - c[None, :]) ** 2, axis=1)
        hdot = -2.0 * np.sum((x - c[None, :]) * v, axis=1)
        cbf = hdot + args.gamma_p * h
        cbf_ok = np.min(cbf) >= -args.tol_cbf
        outside_p = r > (args.x_m + args.tol_plot_outside)
        outside_v = vn > (args.v_m + args.tol_plot_outside)
        outside_a = an > (args.a_m + args.tol_plot_outside)
        idxBp = np.where(np.abs(h) <= -0.01)[0]
        h_v = args.v_m * args.v_m - vn * vn
        h_a = args.a_m * args.a_m - an * an
        idxBv = np.where(np.abs(h_v) <= -0.01)[0]
        idxBa = np.where(np.abs(h_a) <= -0.01)[0]
        task_msg = (
            f"task-space: pos={'PASS' if pos_ok else 'FAIL'} (max ||x-c||={np.max(r):.6f}, X_m={args.x_m:.6f}), "
            f"vel={'PASS' if vel_ok else 'FAIL'} (max ||v||={np.max(vn):.6f}, V_m={args.v_m:.6f}), "
            f"acc={'PASS' if acc_ok else 'FAIL'} (max ||a||={np.max(an):.6f}, A_m={args.a_m:.6f}), "
            f"cbf_p={'PASS' if cbf_ok else 'FAIL'} (min={np.min(cbf):.6e})"
        )
    else:
        task_msg = "task-space: skipped (missing executed x/y/z columns)"

    print("\n=== Joint Space Feasibility Check ===")
    print(f"samples: {len(t)}")
    print(f"q bounds check (with margin {args.joint_margin:.4f}): {'PASS' if q_ok else 'FAIL'}")
    print(
        "qdot limit check: "
        f"{'PASS' if qdot_ok else 'FAIL'} | "
        f"worst normalized usage={np.max(np.abs(qdot) / np.maximum(qdot_max[None, :], 1e-12)):.6f}"
    )
    print(
        "qddot limit check: "
        f"{'PASS' if qddot_ok else 'FAIL'} | "
        f"worst normalized usage={np.max(np.abs(qddot) / np.maximum(qddot_max[None, :], 1e-12)):.6f}"
    )
    print(f"minimum distance to hard joint limits: {min_margin:.6f} rad")
    print(task_msg)
    print(ori_msg)
    print(err_msg)

    # ==================== PLOTS ====================
    qdot_usage = np.abs(qdot) / np.maximum(qdot_max[None, :], 1e-12)
    qddot_usage = np.abs(qddot) / np.maximum(qddot_max[None, :], 1e-12)
    max_qdot_usage = np.max(qdot_usage, axis=1)
    max_qddot_usage = np.max(qddot_usage, axis=1)
    min_margin_t = np.min(np.minimum(q - q_min[None, :], q_max[None, :] - q), axis=1)

    plt.figure()
    plt.grid(True)
    plt.plot(t, max_qdot_usage, linewidth=1.8, label="max_j |qdot_j|/qdot_max")
    plt.plot(t, max_qddot_usage, linewidth=1.8, label="max_j |qddot_j|/qddot_max")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="limit")
    plt.xlabel("Time t")
    plt.ylabel("normalized usage")
    plt.title("Worst-case joint velocity/acceleration usage")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t, min_margin_t, linewidth=1.8, label="min_j dist to hard limit")
    plt.axhline(args.joint_margin, color="k", linestyle="--", linewidth=1.2, label="margin target")
    plt.xlabel("Time t")
    plt.ylabel("margin [rad]")
    plt.title("Closest distance to joint hard limits over time")
    plt.legend(loc="best")

    # Joint positions with bounds and margin bounds.
    for j in range(ARM_DOFS):
        plt.figure()
        plt.grid(True)
        plt.plot(t, q[:, j], linewidth=1.8, label=f"q{j}")
        if np.isfinite(q_min[j]):
            plt.axhline(q_min[j], color="k", linestyle="--", linewidth=1.2, label="q_min")
            plt.axhline(q_min[j] + args.joint_margin, color="k", linestyle=":", linewidth=1.2, label="q_min+margin")
        if np.isfinite(q_max[j]):
            plt.axhline(q_max[j], color="k", linestyle="--", linewidth=1.2, label="q_max")
            plt.axhline(q_max[j] - args.joint_margin, color="k", linestyle=":", linewidth=1.2, label="q_max-margin")
        plt.xlabel("Time t")
        plt.ylabel(f"q{j} [rad]")
        plt.title(f"Joint position q{j}(t)")
        plt.legend(loc="best")

    # Joint velocities/accelerations with limits.
    for j in range(ARM_DOFS):
        plt.figure()
        plt.grid(True)
        plt.plot(t, qdot[:, j], linewidth=1.8, label=f"qdot{j}")
        plt.axhline(qdot_max[j], color="k", linestyle="--", linewidth=1.2, label="qdot_max")
        plt.axhline(-qdot_max[j], color="k", linestyle="--", linewidth=1.2)
        plt.xlabel("Time t")
        plt.ylabel(f"qdot{j} [rad/s]")
        plt.title(f"Joint velocity qdot{j}(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t, qddot[:, j], linewidth=1.8, label=f"qddot{j}")
        plt.axhline(qddot_max[j], color="k", linestyle="--", linewidth=1.2, label="qddot_max")
        plt.axhline(-qddot_max[j], color="k", linestyle="--", linewidth=1.2)
        plt.xlabel("Time t")
        plt.ylabel(f"qddot{j} [rad/s^2]")
        plt.title(f"Joint acceleration qddot{j}(t)")
        plt.legend(loc="best")

    # Task-space plots if executed x/y/z columns exist.
    if x is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Executed trajectory with safety sphere")
        u = np.linspace(0.0, 2.0 * np.pi, 61)
        vv = np.linspace(0.0, np.pi, 61)
        xs = args.center[0] + args.x_m * np.outer(np.cos(u), np.sin(vv))
        ys = args.center[1] + args.x_m * np.outer(np.sin(u), np.sin(vv))
        zs = args.center[2] + args.x_m * np.outer(np.ones_like(u), np.cos(vv))
        ax.plot_surface(xs, ys, zs, alpha=0.20, linewidth=0.0, color=(0.85, 0.90, 1.00))
        ax.plot(x[:, 0], x[:, 1], x[:, 2], "b", linewidth=1.6, label="Executed")
        x_out = x.copy()
        x_out[~outside_p, :] = np.nan
        ax.plot(x_out[:, 0], x_out[:, 1], x_out[:, 2], "r", linewidth=2.2, label="Outside")
        if idxBp.size > 0:
            ax.plot(x[idxBp, 0], x[idxBp, 1], x[idxBp, 2], "ro", markersize=5, label="Boundary-near")
        ax.legend(loc="best")
        ax.set_box_aspect((1, 1, 1))

        plt.figure()
        plt.grid(True)
        plt.plot(t, r, linewidth=1.8, label="||x-c||")
        plt.axhline(args.x_m, color="k", linestyle="--", linewidth=1.2, label="X_m")
        r_out = r.copy()
        r_out[~outside_p] = np.nan
        plt.plot(t, r_out, "r", linewidth=2.2, label="outside")
        if idxBp.size > 0:
            plt.plot(t[idxBp], r[idxBp], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("||x-c||")
        plt.title("Task-space distance from center")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t, h, linewidth=1.8, label="h_p")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
        if idxBp.size > 0:
            plt.plot(t[idxBp], h[idxBp], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("h_p")
        plt.title("Position barrier h_p(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t, vn, linewidth=1.8, label="||v||")
        plt.axhline(args.v_m, color="k", linestyle="--", linewidth=1.2, label="V_m")
        vn_out = vn.copy()
        vn_out[~outside_v] = np.nan
        plt.plot(t, vn_out, "r", linewidth=2.2, label="outside")
        if idxBv.size > 0:
            plt.plot(t[idxBv], vn[idxBv], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("||v||")
        plt.title("Task-space speed magnitude")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        h_v = args.v_m * args.v_m - vn * vn
        plt.plot(t, h_v, linewidth=1.8, label="h_v")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
        if idxBv.size > 0:
            plt.plot(t[idxBv], h_v[idxBv], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("h_v")
        plt.title("Velocity barrier h_v(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t, an, linewidth=1.8, label="||a||")
        plt.axhline(args.a_m, color="k", linestyle="--", linewidth=1.2, label="A_m")
        an_out = an.copy()
        an_out[~outside_a] = np.nan
        plt.plot(t, an_out, "r", linewidth=2.2, label="outside")
        if idxBa.size > 0:
            plt.plot(t[idxBa], an[idxBa], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("||a||")
        plt.title("Task-space acceleration magnitude")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        h_a = args.a_m * args.a_m - an * an
        plt.plot(t, h_a, linewidth=1.8, label="h_a")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
        if idxBa.size > 0:
            plt.plot(t[idxBa], h_a[idxBa], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("h_a")
        plt.title("Acceleration barrier h_a(t)")
        plt.legend(loc="best")

        plt.figure()
        plt.grid(True)
        plt.plot(t, cbf, linewidth=1.8, label="h_p_dot + gamma_p*h_p")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
        plt.axhline(-args.tol_cbf, color="k", linestyle=":", linewidth=1.2, label="-tol")
        if idxBp.size > 0:
            plt.plot(t[idxBp], cbf[idxBp], "ro", markersize=5, label="boundary-near")
        plt.xlabel("Time t")
        plt.ylabel("CBF residual")
        plt.title("Position CBF residual")
        plt.legend(loc="best")

        comp = ["x", "y", "z"]
        for k, nm in enumerate(comp):
            plt.figure()
            plt.grid(True)
            plt.plot(t, v[:, k], linewidth=1.8, label=f"v_{nm}")
            plt.xlabel("Time t")
            plt.ylabel(f"v_{nm} [m/s]")
            plt.title(f"Task-space velocity component {nm}(t)")
            plt.legend(loc="best")

        for k, nm in enumerate(comp):
            plt.figure()
            plt.grid(True)
            plt.plot(t, a[:, k], linewidth=1.8, label=f"a_{nm}")
            plt.xlabel("Time t")
            plt.ylabel(f"a_{nm} [m/s^2]")
            plt.title(f"Task-space acceleration component {nm}(t)")
            plt.legend(loc="best")

    if err_idx is not None:
        plt.figure()
        plt.grid(True)
        plt.plot(t, err, linewidth=1.8, label="tracking error")
        plt.xlabel("Time t")
        plt.ylabel("error [m]")
        plt.title("Task-space tracking error")
        plt.legend(loc="best")

    if ori_idx is not None:
        plt.figure()
        plt.grid(True)
        plt.plot(t, ori_err, linewidth=1.8, label="orientation error")
        plt.axhline(args.ori_err_max, color="k", linestyle="--", linewidth=1.2, label="ori_err_max")
        plt.xlabel("Time t")
        plt.ylabel("error [rad]")
        plt.title("Orientation error")
        plt.legend(loc="best")

    if args.save_plots_dir is not None:
        args.save_plots_dir.mkdir(parents=True, exist_ok=True)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            fig.savefig(args.save_plots_dir / f"jointspace_check_{fignum:03d}.png", dpi=160, bbox_inches="tight")
        print(f"Saved plots to: {args.save_plots_dir}")
    if args.no_plot or args.save_plots_dir is not None:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
