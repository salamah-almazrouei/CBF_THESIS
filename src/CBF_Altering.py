import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -------------------- User Inputs --------------------
csvFileIn = Path("/Users/salamahalmazrouei/Desktop/test.csv")
csvFileOut = Path("/Users/salamahalmazrouei/Desktop/test_filtered.csv")

# Sphere center
c = np.array([0.36, -0.27, 0.46], dtype=float)

# Bounds
X_m = 0.45  # meters (sphere radius)
V_m = 1.15  # m/s (OVERALL speed limit)
A_m = 13.0  # m/s^2 (OVERALL translational acceleration)

# CBF gains
gamma_p = 10.0  # position
gamma_v = 10.0  # velocity CBF-like (discrete)

# Tracking gains
Kp_base = 2.0
Kp_boost = 6.0
Kp_power = 3.0

# Projection iterations
nProj = 10000

# Scale-up behavior
scaleUpWhenNominalViolates = True
tol_spd = 1e-6

# Plot/check tolerances
tol_pos = 1e-6


def load_csv(csv_path: Path) -> np.ndarray:
    M = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    M = M[~np.all(np.isnan(M), axis=1)]
    if M.size == 0:
        raise ValueError("CSV is empty or invalid.")
    if np.isnan(M).any():
        raise ValueError("CSV contains non-numeric values. Remove headers/text rows.")
    if M.shape[1] < 4:
        raise ValueError("CSV must have at least columns: [t, px, py, pz].")
    return M


def clamp_ball(v: np.ndarray, Vmax: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > Vmax and n > 0.0:
        return (Vmax / n) * v
    return v


def project_ball_around(v: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    # Project v onto ball {x: ||x-center|| <= radius}
    d = v - center
    nd = np.linalg.norm(d)
    if nd > radius and nd > 0.0:
        return center + (radius / nd) * d
    return v


def project_halfspace(v: np.ndarray, q: np.ndarray, d: float) -> np.ndarray:
    # Project v onto half-space {x: q^T x <= d}
    qq = float(q @ q)
    if qq < 1e-14:
        return v  # degenerate, do nothing
    viol = float(q @ v) - d
    if viol > 0.0:
        return v - (viol / qq) * q
    return v


def scale_up_feasible(
    v: np.ndarray,
    v_prev: np.ndarray,
    Vmax: float,
    dv_max: float,
    a_p: np.ndarray,
    b_p: float,
    q: np.ndarray,
    d: float,
) -> np.ndarray:
    # Scales v -> alpha*v to make speed as large as possible while keeping all constraints.
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return v

    # (1) speed
    alpha_spd = Vmax / nv

    # (2) position CBF
    apv = float(a_p @ v)
    alpha_pos = (b_p / apv) if apv > 1e-12 else np.inf

    # (3) velocity CBF-like
    qv = float(q @ v)
    alpha_vel = (d / qv) if qv > 1e-12 else np.inf

    # (4) accel-step: ||alpha*v - v_prev|| <= dv_max  -> quadratic in alpha
    vv = float(v @ v)
    vp = float(v @ v_prev)
    pp = float(v_prev @ v_prev)

    A = vv
    B = -2.0 * vp
    C = pp - dv_max * dv_max

    disc = B * B - 4.0 * A * C
    if disc >= 0.0:
        sqrt_disc = np.sqrt(disc)
        r1 = (-B - sqrt_disc) / (2.0 * A)
        r2 = (-B + sqrt_disc) / (2.0 * A)
        alpha_acc = max(0.0, max(r1, r2))  # max feasible alpha
    else:
        alpha_acc = 1.0  # fallback

    alpha_max = min(alpha_spd, alpha_pos, alpha_vel, alpha_acc)

    # only scale up (never scale down here)
    alpha = max(1.0, alpha_max)
    v2 = alpha * v

    # final safety projection (numerical)
    v2 = clamp_ball(v2, Vmax)
    v2 = project_ball_around(v2, v_prev, dv_max)
    v2 = project_halfspace(v2, a_p, b_p)
    v2 = project_halfspace(v2, q, d)
    return v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Position CBF + Velocity CBF-like + speed/accel norm filtering from CSV."
    )
    parser.add_argument("--in_csv", type=Path, default=csvFileIn, help="Input CSV path.")
    parser.add_argument("--out_csv", type=Path, default=csvFileOut, help="Output CSV path.")
    args = parser.parse_args()

    # -------------------- Load CSV --------------------
    M = load_csv(args.in_csv)
    t = M[:, 0]
    Pnom = M[:, 1:4]
    N = Pnom.shape[0]

    if N < 5:
        raise ValueError("Need at least 5 samples.")
    dtv = np.diff(t)
    if np.any(dtv <= 0):
        raise ValueError("Time column t must be strictly increasing.")

    # -------------------- Nominal velocity (v_nom) --------------------
    hasVelCols = M.shape[1] >= 7
    Vnom = np.zeros((N, 3), dtype=float)
    if hasVelCols:
        Vnom = M[:, 4:7]
    else:
        for k in range(3):
            Vnom[:, k] = np.gradient(Pnom[:, k], t)

    # -------------------- Filtered arrays --------------------
    P = np.zeros((N, 3), dtype=float)
    V = np.zeros((N, 3), dtype=float)

    # Diagnostics: residuals of half-spaces (should be <= 0)
    pos_cbf_residual = np.zeros(N, dtype=float)  # a_p^T v - b_p
    vel_cbf_residual = np.zeros(N, dtype=float)  # q^T v - d

    # init
    P[0, :] = Pnom[0, :]
    v_prev = clamp_ball(Vnom[0, :].copy(), V_m)  # start feasible in speed
    V[0, :] = v_prev

    # ==================== MAIN LOOP ====================
    for i in range(N - 1):
        dt = t[i + 1] - t[i]
        dv_max = A_m * dt

        p = P[i, :].copy()
        p_nom = Pnom[i, :].copy()
        v_nom = Vnom[i, :].copy()

        # --- tracking reference velocity ---
        s = i / (N - 2)
        Kp_eff = Kp_base * (1.0 + Kp_boost * (s**Kp_power))
        v_ref = v_nom + Kp_eff * (p_nom - p)

        # -------------------- (1) Position CBF half-space --------------------
        pc = p - c
        h_p = X_m * X_m - float(pc @ pc)
        a_p = 2.0 * pc
        b_p = gamma_p * h_p

        # -------------------- (2) Velocity CBF-like half-space --------------------
        q = v_prev
        qn2 = float(q @ q)  # ||v_prev||^2
        d = qn2 + (dt / 2.0) * gamma_v * (V_m * V_m - qn2)

        # -------------------- iterative projections onto all sets --------------------
        v = v_ref.copy()
        for _ in range(nProj):
            v = project_ball_around(v, v_prev, dv_max)  # accel-step ball
            v = clamp_ball(v, V_m)  # speed ball
            v = project_halfspace(v, a_p, b_p)  # position CBF half-space
            v = project_halfspace(v, q, d)  # velocity CBF-like half-space

        # -------------------- scale up (only if nominal violates speed) --------------------
        if scaleUpWhenNominalViolates:
            if np.linalg.norm(v_nom) > V_m + tol_spd:
                v = scale_up_feasible(v, v_prev, V_m, dv_max, a_p, b_p, q, d)

        # store + integrate
        V[i, :] = v
        P[i + 1, :] = p + v * dt

        # diagnostics residuals (<= 0 desired)
        pos_cbf_residual[i] = float(a_p @ v - b_p)
        vel_cbf_residual[i] = float(q @ v - d)

        # update previous
        v_prev = v

    V[N - 1, :] = V[N - 2, :]
    pos_cbf_residual[N - 1] = pos_cbf_residual[N - 2]
    vel_cbf_residual[N - 1] = vel_cbf_residual[N - 2]

    # -------------------- Save filtered to CSV --------------------
    Out = np.column_stack((t, P, V))
    np.savetxt(args.out_csv, Out, delimiter=",")
    print(f"\n✅ Saved filtered trajectory to:\n{args.out_csv}")

    # ==================== Curves for plotting (NO ACC) ====================
    Pc_nom = Pnom - c[None, :]
    r_nom = np.sqrt(np.sum(Pc_nom * Pc_nom, axis=1))
    hp_nom = X_m * X_m - r_nom * r_nom
    vn_nom = np.sqrt(np.sum(Vnom * Vnom, axis=1))

    Pc_f = P - c[None, :]
    r_f = np.sqrt(np.sum(Pc_f * Pc_f, axis=1))
    hp_f = X_m * X_m - r_f * r_f
    vn_f = np.sqrt(np.sum(V * V, axis=1))

    outside_f = r_f > (X_m + tol_pos)
    en = np.sqrt(np.sum((Pnom - P) * (Pnom - P), axis=1))

    # -------------------- Verification prints --------------------
    print("\n==================== FILTER VERIFICATION ====================")
    print(f"Filtered: max ||p-c|| = {np.max(r_f):.9f} (X_m={X_m:.9f})")
    print(f"Filtered: min h_p     = {np.min(hp_f):.3e}")
    print(f"Filtered: max ||v||   = {np.max(vn_f):.9f} (V_m={V_m:.9f})")
    print(f"Endpoint error: ||p_nom(T)-p_f(T)|| = {np.linalg.norm(Pnom[-1, :] - P[-1, :]):.9e}")

    print(f"Position CBF residual max (should be <=0): {np.max(pos_cbf_residual):.3e}")
    print(f"Velocity CBF residual max (should be <=0): {np.max(vel_cbf_residual):.3e}")

    if np.any(outside_f):
        idx = int(np.where(outside_f)[0][0])
        print(
            "WARNING: OUTSIDE sphere (filtered): "
            f"first at i={idx + 1}, t={t[idx]:.6f}, r={r_f[idx]:.9f}"
        )
    else:
        print("✅ Filtered position inside sphere (within tolerance).")

    if np.max(vn_f) > V_m + tol_spd:
        idx = int(np.where(vn_f > V_m + tol_spd)[0][0])
        print(
            "WARNING: SPEED NORM violation: "
            f"first at i={idx + 1}, t={t[idx]:.6f}, ||v||={vn_f[idx]:.9f}"
        )
    else:
        print("✅ Filtered speed norm <= V_m.")

    # ==================== PLOTS (NO ACC) ====================
    # 3D trajectory + sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True)
    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    ax.set_zlabel("p_z")
    ax.set_title("3D: Nominal vs Filtered with safety sphere")

    uu = np.linspace(0.0, 2.0 * np.pi, 61)
    vv = np.linspace(0.0, np.pi, 61)
    Xs = np.outer(np.cos(uu), np.sin(vv))
    Ys = np.outer(np.sin(uu), np.sin(vv))
    Zs = np.outer(np.ones_like(uu), np.cos(vv))

    ax.plot_surface(
        c[0] + X_m * Xs,
        c[1] + X_m * Ys,
        c[2] + X_m * Zs,
        linewidth=0.0,
        alpha=0.15,
        color=(0.85, 0.90, 1.00),
    )
    ax.plot(Pnom[:, 0], Pnom[:, 1], Pnom[:, 2], "k--", linewidth=1.2, label="Nominal")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "b", linewidth=1.8, label="Filtered")

    P_out = P.copy()
    P_out[~outside_f, :] = np.nan
    ax.plot(P_out[:, 0], P_out[:, 1], P_out[:, 2], "r", linewidth=2.5, label="Filtered outside")
    ax.legend(loc="best")
    ax.set_box_aspect((1, 1, 1))

    # h_p(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, hp_nom, "k--", linewidth=1.3, label="h_p nominal")
    plt.plot(t, hp_f, "b", linewidth=1.8, label="h_p filtered")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.1, label="0")
    plt.xlabel("Time t")
    plt.ylabel("h_p(t)")
    plt.title("Barrier h_p(t)")
    plt.legend(loc="best")

    # ||v|| vs V_m
    plt.figure()
    plt.grid(True)
    plt.plot(t, vn_nom, "k--", linewidth=1.3, label="||v|| nominal")
    plt.plot(t, vn_f, "b", linewidth=1.8, label="||v|| filtered")
    plt.axhline(V_m, color="k", linestyle="--", linewidth=1.1, label="V_m")
    plt.xlabel("Time t")
    plt.ylabel("||v||")
    plt.title("Overall speed norm")
    plt.legend(loc="best")

    # v components
    plt.figure()
    plt.grid(True)
    plt.plot(t, Vnom[:, 0], "k--", linewidth=1.2, label="v_x nominal")
    plt.plot(t, V[:, 0], "b", linewidth=1.8, label="v_x filtered")
    plt.xlabel("Time t")
    plt.ylabel("v_x")
    plt.title("v_x(t)")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t, Vnom[:, 1], "k--", linewidth=1.2, label="v_y nominal")
    plt.plot(t, V[:, 1], "b", linewidth=1.8, label="v_y filtered")
    plt.xlabel("Time t")
    plt.ylabel("v_y")
    plt.title("v_y(t)")
    plt.legend(loc="best")

    plt.figure()
    plt.grid(True)
    plt.plot(t, Vnom[:, 2], "k--", linewidth=1.2, label="v_z nominal")
    plt.plot(t, V[:, 2], "b", linewidth=1.8, label="v_z filtered")
    plt.xlabel("Time t")
    plt.ylabel("v_z")
    plt.title("v_z(t)")
    plt.legend(loc="best")

    # tracking error norm
    plt.figure()
    plt.grid(True)
    plt.plot(t, en, linewidth=1.8, label="||p_{nom}-p_f||")
    plt.xlabel("Time t")
    plt.ylabel("Position error norm")
    plt.title("Tracking error norm")
    plt.legend(loc="best")

    # Position CBF residual diagnostic
    plt.figure()
    plt.grid(True)
    plt.plot(t, pos_cbf_residual, linewidth=1.6, label="a_p^T v - b_p")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.1, label="0")
    plt.xlabel("Time t")
    plt.ylabel("Residual")
    plt.title("Position CBF half-space residual (should be <= 0)")
    plt.legend(loc="best")

    # Velocity CBF-like residual diagnostic
    plt.figure()
    plt.grid(True)
    plt.plot(t, vel_cbf_residual, linewidth=1.6, label="(v_prev)^T v - d")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.1, label="0")
    plt.xlabel("Time t")
    plt.ylabel("Residual")
    plt.title("Velocity CBF-like half-space residual (should be <= 0)")
    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
