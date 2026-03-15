import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -------------------- Inputs --------------------
CSV_FILE = Path("/Users/salamahalmazrouei/Desktop/test_filtered.csv")

# Center of position sphere
c = np.array([0.36, -0.27, 0.46], dtype=float)

# Bounds
X_m = 0.45
V_m = 1.15
A_m = 13.1

# CBF gains (gamma >= 0)
gamma_p = 1.0
gamma_v = 100.0
gamma_a = 10000.0

# -------------------- Tolerances --------------------
tol_hp = 1e-4
tol_hv = 1e-4
tol_ha = 1e-4

tol_r = 1e-3
tol_vn = 1e-3
tol_an = 1e-3

tol_cbf = 1e-8
tol_grad = 1e-10


def tf(flag: bool) -> str:
    return "✅ PASS" if flag else "❌ FAIL"


def load_csv(csv_file: Path) -> np.ndarray:
    M = np.genfromtxt(csv_file, delimiter=",", dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    M = M[~np.all(np.isnan(M), axis=1)]
    if np.isnan(M).any():
        raise ValueError("CSV contains non-numeric values. Remove headers/text rows.")
    if M.shape[1] < 4:
        raise ValueError("CSV must have at least 4 columns: [t, px, py, pz].")
    return M


def deriv(sig: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Match MATLAB gradient(sig, t) behavior closely.
    return np.gradient(sig, t)


def local_report_cbf(
    name: str,
    t: np.ndarray,
    h: np.ndarray,
    hdot: np.ndarray,
    cbf: np.ndarray,
    gradnorm: np.ndarray,
    idxB: np.ndarray,
    tol_h: float,
    tolCBF: float,
    tolG: float,
) -> None:
    print(f"\n[{name}]")

    kmin = int(np.argmin(np.abs(h)))
    print(
        "  Closest-to-boundary: "
        f"t={t[kmin]:.6f}, h={h[kmin]:.6e}, h_dot={hdot[kmin]:.6e}, cbf={cbf[kmin]:.6e}"
    )

    minCBF = np.min(cbf)
    nBadAll = int(np.sum(cbf < -tolCBF))
    print(
        "  Global CBF: "
        f"min(cbf)={minCBF:.6e}, violations(all)={nBadAll} (tol={tolCBF:.1e}) -> {tf(nBadAll == 0)}"
    )

    if idxB.size == 0:
        print("  Boundary-near samples: NONE (with current tolerances)")
        print("  -> Boundary CBF check not applicable (trajectory never gets near boundary).")
        return

    maxAbsH = np.max(np.abs(h[idxB]))
    pass_h0 = maxAbsH <= tol_h

    minCBFb = np.min(cbf[idxB])
    badMask = cbf[idxB] < -tolCBF
    numBad = int(np.sum(badMask))
    pass_cbf = numBad == 0

    minGrad = np.min(gradnorm[idxB])
    numZeroGrad = int(np.sum(gradnorm[idxB] <= tolG))
    pass_grad = numZeroGrad == 0

    print(f"  Boundary-near samples found: {idxB.size}")
    print(f"  (1) h≈0 near boundary:   max |h| = {maxAbsH:.6e} (tol={tol_h:.1e}) -> {tf(pass_h0)}")
    print(
        "  (2) CBF on boundary:     "
        f"min cbf = {minCBFb:.6e}, violations={numBad} (tol={tolCBF:.1e}) -> {tf(pass_cbf)}"
    )
    print(
        "  (3) ||grad||!=0:         "
        f"min ||grad|| = {minGrad:.6e}, zero-count={numZeroGrad} (tol={tolG:.1e}) -> {tf(pass_grad)}"
    )

    if not pass_cbf:
        idxFailLocal = int(np.where(badMask)[0][0])
        idxFail = int(idxB[idxFailLocal])
        print(
            "  FIRST CBF FAIL (boundary-near) at "
            f"t={t[idxFail]:.6f}: h={h[idxFail]:.6e}, cbf={cbf[idxFail]:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full safety check + h_dot + Nagumo/CBF conditions (position, velocity, acceleration) from CSV."
    )
    parser.add_argument("--csv", type=Path, default=CSV_FILE, help="CSV path.")
    args = parser.parse_args()

    # -------------------- Load CSV --------------------
    M = load_csv(args.csv)
    t = M[:, 0]
    P = M[:, 1:4]
    N = P.shape[0]

    if N < 5:
        raise ValueError("Need at least 5 samples for reliable derivatives using gradient.")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Time column t must be strictly increasing.")

    # ==================== Derivatives aligned on same grid (length N) ====================
    hasVelCols = M.shape[1] >= 7

    V = np.zeros((N, 3))
    A = np.zeros((N, 3))
    J = np.zeros((N, 3))

    if hasVelCols:
        V = M[:, 4:7]
        for k in range(3):
            A[:, k] = deriv(V[:, k], t)
            J[:, k] = deriv(A[:, k], t)
    else:
        for k in range(3):
            V[:, k] = deriv(P[:, k], t)
            A[:, k] = deriv(V[:, k], t)
            J[:, k] = deriv(A[:, k], t)

    # ==================== Barrier functions h and time derivatives h_dot ====================
    # Position barrier h_p(p)
    Pc = P - c[None, :]
    r2 = np.sum(Pc * Pc, axis=1)
    r = np.sqrt(r2)
    h_p = X_m * X_m - r2
    h_p_dot = -2.0 * np.sum(Pc * V, axis=1)
    grad_p_norm = 2.0 * r
    cbf_p = h_p_dot + gamma_p * h_p

    # Velocity barrier h_v(v)
    v2 = np.sum(V * V, axis=1)
    vn = np.sqrt(v2)
    h_v = V_m * V_m - v2
    h_v_dot = -2.0 * np.sum(V * A, axis=1)
    grad_v_norm = 2.0 * vn
    cbf_v = h_v_dot + gamma_v * h_v

    # Acceleration barrier h_a(a)
    a2 = np.sum(A * A, axis=1)
    an = np.sqrt(a2)
    h_a = A_m * A_m - a2
    h_a_dot = -2.0 * np.sum(A * J, axis=1)
    grad_a_norm = 2.0 * an
    cbf_a = h_a_dot + gamma_a * h_a

    # ==================== Basic safety flags (h>=0) ====================
    outside_p = h_p < 0.0
    outside_v = h_v < 0.0
    outside_a = h_a < 0.0

    safe_p = bool(np.all(h_p >= 0.0))
    safe_v = bool(np.all(h_v >= 0.0))
    safe_a = bool(np.all(h_a >= 0.0))

    # ==================== Boundary-near indices ====================
    idxBp = np.unique(np.concatenate((np.where(np.abs(h_p) <= tol_hp)[0], np.where(np.abs(r - X_m) <= tol_r)[0])))
    idxBv = np.unique(np.concatenate((np.where(np.abs(h_v) <= tol_hv)[0], np.where(np.abs(vn - V_m) <= tol_vn)[0])))
    idxBa = np.unique(np.concatenate((np.where(np.abs(h_a) <= tol_ha)[0], np.where(np.abs(an - A_m) <= tol_an)[0])))

    # ==================== CBF global pass/fail ====================
    pass_cbf_p = bool(np.all(cbf_p >= -tol_cbf))
    pass_cbf_v = bool(np.all(cbf_v >= -tol_cbf))
    pass_cbf_a = bool(np.all(cbf_a >= -tol_cbf))

    # ==================== Print summary ====================
    print("\n==================== BASIC SAFETY SUMMARY ====================")

    print("\n[POSITION]")
    print("h_p = X_m^2 - ||p-c||^2, safe if h_p >= 0")
    print(f"Min h_p = {np.min(h_p):.6e}")
    print(f"Max ||p-c|| = {np.max(r):.9f} (X_m={X_m:.9f})")
    print(f"CBF condition: h_p_dot >= -gamma_p*h_p  (gamma_p={gamma_p:.3g})")
    print(f"Min residual (h_p_dot + gamma*h) = {np.min(cbf_p):.6e}")
    print(f"CBF global check -> {tf(pass_cbf_p)}")
    if safe_p:
        print("Bound check -> ✅ SAFE")
    else:
        idx = int(np.where(h_p < 0.0)[0][0])
        print(f"Bound check -> ❌ UNSAFE (outside at index {idx + 1}, time {t[idx]:.6f})")

    print("\n[VELOCITY]")
    print("h_v = V_m^2 - ||v||^2, safe if h_v >= 0")
    print(f"Min h_v = {np.min(h_v):.6e}")
    print(f"Max ||v|| = {np.max(vn):.9f} (V_m={V_m:.9f})")
    print(f"CBF condition: h_v_dot >= -gamma_v*h_v  (gamma_v={gamma_v:.3g})")
    print(f"Min residual (h_v_dot + gamma*h) = {np.min(cbf_v):.6e}")
    print(f"CBF global check -> {tf(pass_cbf_v)}")
    if safe_v:
        print("Bound check -> ✅ SAFE")
    else:
        idx = int(np.where(h_v < 0.0)[0][0])
        print(f"Bound check -> ❌ UNSAFE (violation at index {idx + 1}, time {t[idx]:.6f})")

    print("\n[ACCELERATION]")
    print("h_a = A_m^2 - ||a||^2, safe if h_a >= 0")
    print(f"Min h_a = {np.min(h_a):.6e}")
    print(f"Max ||a|| = {np.max(an):.9f} (A_m={A_m:.9f})")
    print(f"CBF condition: h_a_dot >= -gamma_a*h_a  (gamma_a={gamma_a:.3g})")
    print(f"Min residual (h_a_dot + gamma*h) = {np.min(cbf_a):.6e}")
    print(f"CBF global check -> {tf(pass_cbf_a)}")
    if safe_a:
        print("Bound check -> ✅ SAFE")
    else:
        idx = int(np.where(h_a < 0.0)[0][0])
        print(f"Bound check -> ❌ UNSAFE (violation at index {idx + 1}, time {t[idx]:.6f})")

    print("\n[OVERALL]")
    if safe_p and safe_v and safe_a:
        print("✅ SAFE overall: bounds satisfied (h>=0).")
    else:
        print("❌ UNSAFE overall: at least one bound violated (h<0).")
    if pass_cbf_p and pass_cbf_v and pass_cbf_a:
        print("✅ CBF overall: all residuals >= -tol_cbf.")
    else:
        print("❌ CBF overall: at least one residual violated (< -tol_cbf).")

    # ==================== CBF (boundary-near detailed) checks ====================
    print("\n==================== BOUNDARY-NEAR CBF CHECKS ====================")
    print("We check on boundary-near samples because boundary is not exact in sampled data.")
    print("Condition: (h_dot + gamma*h) >= 0  (accepted if >= -tol_cbf)")

    local_report_cbf("POSITION h_p", t, h_p, h_p_dot, cbf_p, grad_p_norm, idxBp, tol_hp, tol_cbf, tol_grad)
    local_report_cbf("VELOCITY h_v", t, h_v, h_v_dot, cbf_v, grad_v_norm, idxBv, tol_hv, tol_cbf, tol_grad)
    local_report_cbf("ACCEL    h_a", t, h_a, h_a_dot, cbf_a, grad_a_norm, idxBa, tol_ha, tol_cbf, tol_grad)

    # ==================== PLOTS ====================

    # Plot A: 3D trajectory + sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True)
    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    ax.set_zlabel("p_z")
    ax.set_title("3D trajectory with safety sphere (outside = red, boundary-near = circles)")

    u = np.linspace(0.0, 2.0 * np.pi, 61)
    v = np.linspace(0.0, np.pi, 61)
    xs = c[0] + X_m * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + X_m * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + X_m * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.20, linewidth=0.0, color=(0.85, 0.90, 1.00))

    ax.plot(P[:, 0], P[:, 1], P[:, 2], "b", linewidth=1.6, label="Trajectory")

    P_out3 = P.copy()
    P_out3[~outside_p, :] = np.nan
    ax.plot(P_out3[:, 0], P_out3[:, 1], P_out3[:, 2], "r", linewidth=2.2, label="Outside (h_p<0)")

    if idxBp.size > 0:
        ax.plot(P[idxBp, 0], P[idxBp, 1], P[idxBp, 2], "ro", markersize=5, label="Boundary-near")

    ax.set_box_aspect((1, 1, 1))
    minP = np.minimum(np.min(P, axis=0), c)
    maxP = np.maximum(np.max(P, axis=0), c)
    span = np.max(maxP - minP)
    lim = max(span / 2.0, X_m) * 1.4
    ax.set_xlim(c[0] - lim, c[0] + lim)
    ax.set_ylim(c[1] - lim, c[1] + lim)
    ax.set_zlim(c[2] - lim, c[2] + lim)
    ax.legend(loc="best")

    # Plot 1: h_p(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_p, linewidth=1.8, label="h_p(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBp.size > 0:
        plt.plot(t[idxBp], h_p[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_p(t)")
    plt.title("Position barrier: h_p(t) (safe if h_p >= 0)")
    plt.legend(loc="best")

    # Plot 2: ||p-c|| vs X_m
    plt.figure()
    plt.grid(True)
    plt.plot(t, r, linewidth=1.8, label="||p-c||")
    plt.axhline(X_m, color="k", linestyle="--", linewidth=1.2, label="X_m")
    r_out = r.copy()
    r_out[~outside_p] = np.nan
    plt.plot(t, r_out, "r", linewidth=2.2, label="outside")
    if idxBp.size > 0:
        plt.plot(t[idxBp], r[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("||p(t)-c||")
    plt.title("Position distance from center (outside = red)")
    plt.legend(loc="best")

    # Plot 3: h_v(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_v, linewidth=1.8, label="h_v(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBv.size > 0:
        plt.plot(t[idxBv], h_v[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_v(t)")
    plt.title("Velocity barrier: h_v(t) (safe if h_v >= 0)")
    plt.legend(loc="best")

    # Plot 4: ||v|| vs V_m
    plt.figure()
    plt.grid(True)
    plt.plot(t, vn, linewidth=1.8, label="||v||")
    plt.axhline(V_m, color="k", linestyle="--", linewidth=1.2, label="V_m")
    vn_out = vn.copy()
    vn_out[~outside_v] = np.nan
    plt.plot(t, vn_out, "r", linewidth=2.2, label="outside")
    if idxBv.size > 0:
        plt.plot(t[idxBv], vn[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("||v(t)||")
    plt.title("Speed magnitude (outside = red)")
    plt.legend(loc="best")

    # Plot 5: h_a(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_a, linewidth=1.8, label="h_a(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBa.size > 0:
        plt.plot(t[idxBa], h_a[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_a(t)")
    plt.title("Acceleration barrier: h_a(t) (safe if h_a >= 0)")
    plt.legend(loc="best")

    # Plot 6: ||a|| vs A_m
    plt.figure()
    plt.grid(True)
    plt.plot(t, an, linewidth=1.8, label="||a||")
    plt.axhline(A_m, color="k", linestyle="--", linewidth=1.2, label="A_m")
    an_out = an.copy()
    an_out[~outside_a] = np.nan
    plt.plot(t, an_out, "r", linewidth=2.2, label="outside")
    if idxBa.size > 0:
        plt.plot(t[idxBa], an[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("||a(t)||")
    plt.title("Acceleration magnitude (outside = red)")
    plt.legend(loc="best")

    # Plot 7: h_p_dot(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_p_dot, linewidth=1.8, label="h_p_dot(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBp.size > 0:
        plt.plot(t[idxBp], h_p_dot[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_p_dot(t)")
    plt.title("Time derivative: h_p_dot(t)")
    plt.legend(loc="best")

    # Plot 8: h_v_dot(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_v_dot, linewidth=1.8, label="h_v_dot(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBv.size > 0:
        plt.plot(t[idxBv], h_v_dot[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_v_dot(t)")
    plt.title("Time derivative: h_v_dot(t)")
    plt.legend(loc="best")

    # Plot 9: h_a_dot(t)
    plt.figure()
    plt.grid(True)
    plt.plot(t, h_a_dot, linewidth=1.8, label="h_a_dot(t)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    if idxBa.size > 0:
        plt.plot(t[idxBa], h_a_dot[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("h_a_dot(t)")
    plt.title("Time derivative: h_a_dot(t)")
    plt.legend(loc="best")

    # Plot 10: CBF residual for position
    plt.figure()
    plt.grid(True)
    plt.plot(t, cbf_p, linewidth=1.8, label="h_p_dot + gamma_p h_p")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    plt.axhline(-tol_cbf, color="k", linestyle=":", linewidth=1.2, label="-tol")
    if idxBp.size > 0:
        plt.plot(t[idxBp], cbf_p[idxBp], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("CBF residual")
    plt.title("CBF residual (position): h_p_dot + gamma_p h_p")
    plt.legend(loc="best")

    # Plot 11: CBF residual for velocity
    plt.figure()
    plt.grid(True)
    plt.plot(t, cbf_v, linewidth=1.8, label="h_v_dot + gamma_v h_v")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    plt.axhline(-tol_cbf, color="k", linestyle=":", linewidth=1.2, label="-tol")
    if idxBv.size > 0:
        plt.plot(t[idxBv], cbf_v[idxBv], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("CBF residual")
    plt.title("CBF residual (velocity): h_v_dot + gamma_v h_v")
    plt.legend(loc="best")

    # Plot 12: CBF residual for acceleration
    plt.figure()
    plt.grid(True)
    plt.plot(t, cbf_a, linewidth=1.8, label="h_a_dot + gamma_a h_a")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="0")
    plt.axhline(-tol_cbf, color="k", linestyle=":", linewidth=1.2, label="-tol")
    if idxBa.size > 0:
        plt.plot(t[idxBa], cbf_a[idxBa], "ro", markersize=5, label="boundary-near")
    plt.xlabel("Time t")
    plt.ylabel("CBF residual")
    plt.title("CBF residual (accel): h_a_dot + gamma_a h_a")
    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
