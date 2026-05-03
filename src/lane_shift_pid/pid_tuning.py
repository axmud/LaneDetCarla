"""
PID tuning from CARLA system-ID data.

Pipeline
--------
1.  Load step-response CSVs produced by carla_sysid.py (or generate synthetic
    data with --demo so you can preview the workflow without CARLA).
2.  Identify the longitudinal plant      G_lon(s) = K / (tau s + 1)
    via nonlinear least-squares on the throttle-step response.
3.  Identify the effective lateral plant G_lat(s) = (V^2 K_steer) / (L s^2)
    by fitting K_steer to the steady-state yaw rate of the steer step.
4.  Compute PID gains analytically by pole placement
        - Lateral  : (s + wn)^3              -> K_p, K_i, K_d
        - Longitudinal : (s + wn)^2          -> K_p, K_i  (PI is enough)
5.  Build closed-loop transfer functions with python-control, plot
    Bode, step response, root locus, and sensitivity.
6.  Print copy-pasteable gain dicts for pygame_controller.py.

Usage
-----
    python pid_tuning.py --demo                      # synthetic data
    python pid_tuning.py --lon sysid_longitudinal.csv \
                         --lat sysid_lateral.csv     \
                         --meta sysid_meta.csv

The tuning bandwidth wn is the main knob you'll want to play with.
Lower wn  = slower, smoother, more robust.
Higher wn = faster, snappier, less margin -> can oscillate in the real sim.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import control as ct


# =============================================================================
# 1. Loading / synthetic data
# =============================================================================
def load_real(lon_csv, lat_csv, meta_csv):
    lon = pd.read_csv(lon_csv)
    lat = pd.read_csv(lat_csv)
    meta = dict(zip(*pd.read_csv(meta_csv).values.T))
    meta = {k: float(v) if k != 'vehicle' else v for k, v in meta.items()}
    return lon, lat, meta


def make_synthetic():
    """
    Generate plausible CARLA-like step responses so the rest of the pipeline
    can be exercised end-to-end without a running simulator.

    True parameters (what the fit should recover):
        K_true   = 80 km/h per unit throttle
        tau_true = 2.2 s
        K_steer  = 1.05  (steer command [-1,1] -> roughly 1.05 rad max)
        L        = 2.875 m   (Tesla Model 3 wheelbase)
    """
    dt = 0.05
    L  = 2.875
    K_true, tau_true, K_steer = 80.0, 2.2, 1.05

    # Longitudinal step
    t_lon = np.arange(0.0, 10.0, dt)
    u_lon = 0.5
    v_clean = K_true * u_lon * (1.0 - np.exp(-t_lon / tau_true))
    v_noisy = v_clean + np.random.default_rng(0).normal(0, 0.3, len(t_lon))
    lon = pd.DataFrame({'t': t_lon, 'throttle': u_lon, 'speed_kmh': v_noisy})

    # Lateral step (yaw rate reaches steady state in well under a second)
    t_lat = np.arange(0.0, 2.5, dt)
    V = 30.0 / 3.6
    delta_cmd = 0.05
    psi_dot_ss = math.degrees(V / L * K_steer * delta_cmd)   # deg/s
    # crude first-order rise to steady state
    psi_dot = psi_dot_ss * (1.0 - np.exp(-t_lat / 0.15))
    psi_dot += np.random.default_rng(1).normal(0, 0.2, len(t_lat))
    lat = pd.DataFrame({
        't': t_lat,
        'steer_cmd': delta_cmd,
        'speed_kmh': 30.0 + np.random.default_rng(2).normal(0, 0.1, len(t_lat)),
        'yaw_rate_dps': psi_dot,
    })

    meta = {
        'vehicle': 'synthetic.tesla.model3',
        'wheelbase_m': L,
        'dt_s': dt,
        'lon_throttle_step': u_lon,
        'lat_target_speed_kmh': 30.0,
        'lat_steer_step': delta_cmd,
    }
    return lon, lat, meta


# =============================================================================
# 2. System identification
# =============================================================================
def fit_longitudinal(lon_df, u_step):
    """
    Fit  v(t) = K * u_step * (1 - exp(-t/tau))   to the step response.
    Returns (K, tau).
    """
    t = lon_df['t'].to_numpy()
    v = lon_df['speed_kmh'].to_numpy()

    def model(t, K, tau):
        return K * u_step * (1.0 - np.exp(-t / tau))

    # initial guess from steady-state and 63% rise time
    K0 = v[-1] / u_step if u_step > 0 else 60.0
    try:
        idx63 = np.argmax(v >= 0.63 * v[-1])
        tau0 = max(t[idx63], 0.5)
    except Exception:
        tau0 = 2.0

    popt, _ = curve_fit(model, t, v, p0=[K0, tau0], maxfev=5000)
    K, tau = popt
    return float(K), float(tau)


def fit_lateral_gain(lat_df, L, delta_cmd):
    """
    Fit K_steer from steady-state yaw rate of the steer step.
        psi_dot_ss = (V / L) * K_steer * delta_cmd
    -> K_steer = psi_dot_ss * L / (V * delta_cmd)

    We average the last ~25% of the trace as the steady-state estimate.
    """
    n = len(lat_df)
    tail = lat_df.iloc[int(0.75 * n):]
    psi_dot_ss = math.radians(tail['yaw_rate_dps'].mean())   # rad/s
    V = tail['speed_kmh'].mean() / 3.6                       # m/s
    K_steer = psi_dot_ss * L / (V * delta_cmd)
    return float(K_steer), float(V), float(psi_dot_ss)


# =============================================================================
# 3. PID tuning by pole placement
# =============================================================================
def tune_lateral(K_lat, wn):
    """
    Plant: G_lat(s) = K_lat / s^2
    Controller: C(s) = K_p + K_i/s + K_d s

    Closed-loop denominator:
        s^3 + (K_lat K_d) s^2 + (K_lat K_p) s + K_lat K_i

    Match to (s + wn)^3 = s^3 + 3 wn s^2 + 3 wn^2 s + wn^3:
        K_d = 3 wn       / K_lat
        K_p = 3 wn^2     / K_lat
        K_i =     wn^3   / K_lat
    """
    K_d = 3.0 * wn        / K_lat
    K_p = 3.0 * wn * wn   / K_lat
    K_i = wn * wn * wn    / K_lat
    return {'K_P': K_p, 'K_I': K_i, 'K_D': K_d}


def tune_longitudinal(K, tau, wn):
    """
    Plant: G_lon(s) = K / (tau s + 1)
    Use a PI controller C(s) = K_p + K_i/s.

    Closed-loop denominator (after multiplying through by tau s):
        tau s^2 + (1 + K K_p) s + K K_i

    Normalise by tau and match to (s + wn)^2 = s^2 + 2 wn s + wn^2:
        K_p = (2 wn tau - 1) / K
        K_i = (wn^2  * tau)  / K
    """
    K_p = (2.0 * wn * tau - 1.0) / K
    K_i = (wn * wn * tau)        / K
    return {'K_P': K_p, 'K_I': K_i, 'K_D': 0.0}


# =============================================================================
# 4. Plotting
# =============================================================================
def plot_identification(lon_df, lat_df, K, tau, K_steer, V_lat, L, u_step,
                        delta_cmd, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- longitudinal fit ---
    t = lon_df['t'].to_numpy()
    v_meas = lon_df['speed_kmh'].to_numpy()
    v_fit  = K * u_step * (1.0 - np.exp(-t / tau))

    ax = axes[0]
    ax.plot(t, v_meas, '.', ms=3, alpha=0.5, label='measured')
    ax.plot(t, v_fit,  '-', lw=2, label=f'fit: K={K:.2f}, τ={tau:.2f} s')
    ax.axhline(K * u_step, color='k', ls='--', lw=0.8, alpha=0.5,
               label=f'steady state = {K*u_step:.1f} km/h')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('speed [km/h]')
    ax.set_title(f'Longitudinal step (throttle = {u_step})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- lateral steady-state ---
    t = lat_df['t'].to_numpy()
    psi = lat_df['yaw_rate_dps'].to_numpy()
    psi_ss_pred = math.degrees(V_lat / L * K_steer * delta_cmd)

    ax = axes[1]
    ax.plot(t, psi, '.', ms=3, alpha=0.5, label='measured ψ̇')
    ax.axhline(psi_ss_pred, color='C1', lw=2,
               label=f'fit: K_steer={K_steer:.3f}\n(ψ̇_ss={psi_ss_pred:.2f}°/s)')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('yaw rate [°/s]')
    ax.set_title(f'Lateral step (δ_cmd = {delta_cmd}, V = {V_lat*3.6:.1f} km/h)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('System identification', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_lateral_analysis(K_lat, gains, wn, out_path):
    K_p, K_i, K_d = gains['K_P'], gains['K_I'], gains['K_D']
    s = ct.tf('s')
    G = K_lat / s**2
    C = K_p + K_i / s + K_d * s
    L_ol = C * G                       # open-loop
    T = ct.feedback(L_ol, 1)           # closed-loop reference -> output
    S = 1 / (1 + L_ol)                 # sensitivity

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Bode of open loop
    ax = axes[0, 0]
    w = np.logspace(-2, 2, 500)
    mag, phase, _ = ct.bode(L_ol, w, plot=False)
    ax.semilogx(w, 20 * np.log10(mag), lw=2)
    ax.axhline(0, color='k', lw=0.5)
    gm, pm, wg, wp = ct.margin(L_ol)
    pm_str = f'{pm:.1f}°' if np.isfinite(pm) else '∞'
    wp_str = f'{wp:.2f} rad/s' if np.isfinite(wp) else 'n/a'
    ax.set_title(f'Open-loop Bode (lateral)\n'
                 f'PM={pm_str}, ω_c={wp_str}')
    ax.set_xlabel('ω [rad/s]')
    ax.set_ylabel('|L(jω)| [dB]')
    ax.grid(True, which='both', alpha=0.3)

    # Phase
    ax = axes[0, 1]
    ax.semilogx(w, np.degrees(phase), lw=2)
    ax.axhline(-180, color='r', lw=0.5, ls='--')
    ax.set_title('Open-loop phase')
    ax.set_xlabel('ω [rad/s]')
    ax.set_ylabel('∠L(jω) [°]')
    ax.grid(True, which='both', alpha=0.3)

    # Closed-loop step response
    ax = axes[1, 0]
    t = np.linspace(0, 6, 600)
    t_out, y = ct.step_response(T, t)
    ax.plot(t_out, y, lw=2, label=f'ω_n = {wn} rad/s')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Closed-loop step response (cross-track tracking)')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('e_y / reference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Root locus
    ax = axes[1, 1]
    poles = ct.poles(T)
    zeros = ct.zeros(T)
    ax.scatter(poles.real, poles.imag, marker='x', s=100, color='C3',
               label='closed-loop poles', zorder=3)
    if len(zeros):
        ax.scatter(zeros.real, zeros.imag, marker='o', s=80,
                   facecolors='none', edgecolors='C0', label='zeros')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title('Closed-loop pole-zero map')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    fig.suptitle(
        f'Lateral controller analysis  '
        f'(K_p={K_p:.3f}, K_i={K_i:.3f}, K_d={K_d:.3f})',
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_longitudinal_analysis(K, tau, gains, wn, out_path):
    K_p, K_i = gains['K_P'], gains['K_I']
    s = ct.tf('s')
    G = K / (tau * s + 1)
    C = K_p + K_i / s
    L_ol = C * G
    T = ct.feedback(L_ol, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bode
    ax = axes[0]
    w = np.logspace(-2, 2, 500)
    mag, phase, _ = ct.bode(L_ol, w, plot=False)
    ax.semilogx(w, 20 * np.log10(mag), lw=2)
    ax.axhline(0, color='k', lw=0.5)
    gm, pm, wg, wp = ct.margin(L_ol)
    pm_str = f'{pm:.1f}°' if np.isfinite(pm) else '∞'
    wp_str = f'{wp:.2f} rad/s' if np.isfinite(wp) else 'n/a'
    ax.set_title(f'Open-loop Bode (longitudinal)\nPM={pm_str}, ω_c={wp_str}')
    ax.set_xlabel('ω [rad/s]')
    ax.set_ylabel('|L(jω)| [dB]')
    ax.grid(True, which='both', alpha=0.3)

    # Step
    ax = axes[1]
    t = np.linspace(0, 8, 800)
    t_out, y = ct.step_response(T, t)
    ax.plot(t_out, y, lw=2, label=f'ω_n = {wn} rad/s')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Closed-loop speed step response')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('v / v_ref')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Longitudinal controller analysis  '
        f'(K_p={K_p:.3f}, K_i={K_i:.3f})',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_speed_schedule(K_steer, L, wn, V_range_kmh, out_path):
    """
    Show how the lateral PID gains have to be scheduled with speed,
    because the plant gain V^2 K_steer / L scales with V^2.
    """
    V = np.array(V_range_kmh) / 3.6
    K_lat = V**2 * K_steer / L

    K_d = 3.0 * wn        / K_lat
    K_p = 3.0 * wn * wn   / K_lat
    K_i = wn * wn * wn    / K_lat

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(V_range_kmh, K_p, 'o-', label='K_P')
    ax.plot(V_range_kmh, K_i, 's-', label='K_I')
    ax.plot(V_range_kmh, K_d, '^-', label='K_D')
    ax.set_xlabel('vehicle speed V [km/h]')
    ax.set_ylabel('gain')
    ax.set_yscale('log')
    ax.set_title(f'Lateral gain schedule (ω_n = {wn} rad/s)\n'
                 'gains scale ∝ 1/V² because plant gain scales ∝ V²')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 5. Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lon', help='longitudinal CSV from carla_sysid.py')
    p.add_argument('--lat', help='lateral CSV from carla_sysid.py')
    p.add_argument('--meta', help='metadata CSV from carla_sysid.py')
    p.add_argument('--demo', action='store_true',
                   help='use synthetic data instead of real CSVs')
    p.add_argument('--wn-lat', type=float, default=1.5,
                   help='lateral closed-loop bandwidth [rad/s] (default 1.5)')
    p.add_argument('--wn-lon', type=float, default=1.0,
                   help='longitudinal closed-loop bandwidth [rad/s] (default 1.0)')
    p.add_argument('--target-speed', type=float, default=30.0,
                   help='speed [km/h] at which to evaluate the lateral gains')
    p.add_argument('--outdir', default='.')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.demo or not (args.lon and args.lat and args.meta):
        print('[load] Using synthetic data (--demo or missing CSVs)')
        lon_df, lat_df, meta = make_synthetic()
    else:
        lon_df, lat_df, meta = load_real(args.lon, args.lat, args.meta)

    L          = meta['wheelbase_m']
    u_step     = meta['lon_throttle_step']
    delta_cmd  = meta['lat_steer_step']

    # --- ID ---
    K, tau = fit_longitudinal(lon_df, u_step)
    K_steer, V_lat, psi_dot_ss = fit_lateral_gain(lat_df, L, delta_cmd)
    print(f'[ID]  Longitudinal: K = {K:.3f} km/h per throttle, τ = {tau:.3f} s')
    print(f'[ID]  Lateral:      K_steer = {K_steer:.4f}  '
          f'(ψ̇_ss = {math.degrees(psi_dot_ss):.2f}°/s @ V={V_lat*3.6:.1f} km/h)')

    # --- Tune ---
    V_target = args.target_speed / 3.6
    K_lat_at_target = V_target**2 * K_steer / L
    lat_gains = tune_lateral(K_lat_at_target, args.wn_lat)
    lon_gains = tune_longitudinal(K, tau, args.wn_lon)

    print()
    print(f'[tune] Lateral plant gain @ {args.target_speed} km/h: '
          f'V²·K_steer/L = {K_lat_at_target:.3f}')
    print(f'[tune] Lateral PID  (ω_n = {args.wn_lat} rad/s):')
    print(f"         args_lateral = {{'K_P': {lat_gains['K_P']:.4f}, "
          f"'K_I': {lat_gains['K_I']:.4f}, "
          f"'K_D': {lat_gains['K_D']:.4f}, 'dt': 0.05}}")
    print(f'[tune] Longitudinal PID (ω_n = {args.wn_lon} rad/s):')
    print(f"         args_longitudinal = {{'K_P': {lon_gains['K_P']:.4f}, "
          f"'K_I': {lon_gains['K_I']:.4f}, "
          f"'K_D': {lon_gains['K_D']:.4f}, 'dt': 0.05}}")

    # --- Plots ---
    plot_identification(
        lon_df, lat_df, K, tau, K_steer, V_lat, L, u_step, delta_cmd,
        os.path.join(args.outdir, 'src/lane_shift_pid/fig1_identification.png'),
    )
    plot_lateral_analysis(
        K_lat_at_target, lat_gains, args.wn_lat,
        os.path.join(args.outdir, 'src/lane_shift_pid/fig2_lateral_analysis.png'),
    )
    plot_longitudinal_analysis(
        K, tau, lon_gains, args.wn_lon,
        os.path.join(args.outdir, 'src/lane_shift_pid/fig3_longitudinal_analysis.png'),
    )
    plot_speed_schedule(
        K_steer, L, args.wn_lat, [10, 20, 30, 50, 70, 90, 110],
        os.path.join(args.outdir, 'src/lane_shift_pid/fig4_gain_schedule.png'),
    )
    print(f'\n[done] Plots written to {args.outdir}/')


if __name__ == '__main__':
    main()
