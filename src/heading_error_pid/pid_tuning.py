"""
PID tuning for the HEADING-ERROR controller (look-ahead / pure-pursuit-style).

How this differs from the cross-track tuner
-------------------------------------------
The error signal in your new controller is a small angle (radians) between the
car's forward vector and the line to a look-ahead target on the path.

That changes the plant model:

    Cross-track formulation:    G(s) = (V^2 * K_steer / L) / s^2     (double int.)
    Heading-error formulation:  G(s) = (V   * K_steer / L) / s       (single int.)

Why? Steering creates a yaw rate (psi_dot = V*delta/L). The heading error to a
look-ahead point integrates yaw rate ONCE; to integrate further to a sideways
position you'd need a second integration. Look-ahead controllers stop at the
first integration, which is why they're easier to tune.

Consequences:
  1. A PI controller is enough -- no derivative term needed for stability.
  2. Gains scale as 1/V (not 1/V^2) -- gain scheduling is much gentler.
  3. Pole-placement formulas are simpler.

System-ID is the SAME experiment as before (carla_sysid.py): identify K_steer
from the steady-state yaw rate of a small steer step.

Usage
-----
    python pid_tuning_heading.py --demo
    python pid_tuning_heading.py --lon sysid_longitudinal.csv \
                                 --lat sysid_lateral.csv \
                                 --meta sysid_meta.csv \
                                 --wn-lat 2.0 --wn-lon 1.0
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
# 1. Loading / synthetic data  (identical to the cross-track tuner)
# =============================================================================
def load_real(lon_csv, lat_csv, meta_csv):
    lon = pd.read_csv(lon_csv)
    lat = pd.read_csv(lat_csv)
    meta = dict(zip(*pd.read_csv(meta_csv).values.T))
    meta = {k: float(v) if k != 'vehicle' else v for k, v in meta.items()}
    return lon, lat, meta


def make_synthetic():
    """Same synthetic data as the other tuner -- the experiment is identical."""
    dt = 0.05
    L  = 2.875
    K_true, tau_true, K_steer = 80.0, 2.2, 1.05

    t_lon = np.arange(0.0, 10.0, dt)
    u_lon = 0.5
    v_clean = K_true * u_lon * (1.0 - np.exp(-t_lon / tau_true))
    v_noisy = v_clean + np.random.default_rng(0).normal(0, 0.3, len(t_lon))
    lon = pd.DataFrame({'t': t_lon, 'throttle': u_lon, 'speed_kmh': v_noisy})

    t_lat = np.arange(0.0, 2.5, dt)
    V = 30.0 / 3.6
    delta_cmd = 0.05
    psi_dot_ss = math.degrees(V / L * K_steer * delta_cmd)
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
# 2. System identification  (identical to before)
# =============================================================================
def fit_longitudinal(lon_df, u_step):
    t = lon_df['t'].to_numpy()
    v = lon_df['speed_kmh'].to_numpy()

    def model(t, K, tau):
        return K * u_step * (1.0 - np.exp(-t / tau))

    K0 = v[-1] / u_step if u_step > 0 else 60.0
    try:
        idx63 = np.argmax(v >= 0.63 * v[-1])
        tau0 = max(t[idx63], 0.5)
    except Exception:
        tau0 = 2.0

    popt, _ = curve_fit(model, t, v, p0=[K0, tau0], maxfev=5000)
    return float(popt[0]), float(popt[1])


def fit_lateral_gain(lat_df, L, delta_cmd):
    n = len(lat_df)
    tail = lat_df.iloc[int(0.75 * n):]
    psi_dot_ss = math.radians(tail['yaw_rate_dps'].mean())
    V = tail['speed_kmh'].mean() / 3.6
    K_steer = psi_dot_ss * L / (V * delta_cmd)
    return float(K_steer), float(V), float(psi_dot_ss)


# =============================================================================
# 3. Pole-placement tuning -- HEADING-ERROR formulation
# =============================================================================
def tune_lateral_heading(K_lat, wn, zeta=1.0, add_K_D=0.0):
    """
    Plant:      G(s) = K_lat / s         (single integrator)
    Controller: C(s) = K_p + K_i/s       (PI -- D not needed for stability)

    Closed-loop denominator: s^2 + K_lat*K_p*s + K_lat*K_i

    Match to s^2 + 2*zeta*wn*s + wn^2:
        K_p = 2*zeta*wn / K_lat
        K_i = wn^2      / K_lat

    `add_K_D` lets you optionally include a small D term for noise filtering,
    matching CARLA's PIDLateralController structure. It does NOT change the
    pole placement above (the zeta and wn assume PI), so keep it small.
    """
    K_p = 2.0 * zeta * wn / K_lat
    K_i = wn * wn         / K_lat
    K_d = add_K_D
    return {'K_P': K_p, 'K_I': K_i, 'K_D': K_d}


def tune_longitudinal(K, tau, wn):
    """Same PI tuning as before -- longitudinal plant unchanged."""
    K_p = (2.0 * wn * tau - 1.0) / K
    K_i = (wn * wn * tau)        / K
    return {'K_P': K_p, 'K_I': K_i, 'K_D': 0.0}


# =============================================================================
# 4. Plots
# =============================================================================
def plot_identification(lon_df, lat_df, K, tau, K_steer, V_lat, L, u_step,
                        delta_cmd, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    t = lon_df['t'].to_numpy()
    v_meas = lon_df['speed_kmh'].to_numpy()
    v_fit = K * u_step * (1.0 - np.exp(-t / tau))
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

    fig.suptitle('System identification (same experiments as cross-track tuner)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_lateral_heading_analysis(K_lat, gains, wn, out_path):
    K_p, K_i, K_d = gains['K_P'], gains['K_I'], gains['K_D']
    s = ct.tf('s')
    G = K_lat / s                         # single integrator!
    if K_d > 0:
        C = K_p + K_i / s + K_d * s
    else:
        C = K_p + K_i / s
    L_ol = C * G
    T = ct.feedback(L_ol, 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Bode magnitude
    ax = axes[0, 0]
    w = np.logspace(-2, 2, 500)
    mag, phase, _ = ct.bode(L_ol, w, plot=False)
    ax.semilogx(w, 20 * np.log10(mag), lw=2)
    ax.axhline(0, color='k', lw=0.5)
    gm, pm, wg, wp = ct.margin(L_ol)
    pm_str = f'{pm:.1f}°' if np.isfinite(pm) else '∞'
    wp_str = f'{wp:.2f} rad/s' if np.isfinite(wp) else 'n/a'
    ax.set_title(f'Open-loop Bode (lateral, heading-error)\n'
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

    # Step response
    ax = axes[1, 0]
    t = np.linspace(0, 5, 600)
    t_out, y = ct.step_response(T, t)
    ax.plot(t_out, y, lw=2, label=f'ω_n = {wn} rad/s')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Closed-loop step response\n(heading-error tracking)')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('e_h / reference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pole-zero
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

    ctrl_type = 'PID' if K_d > 0 else 'PI'
    fig.suptitle(
        f'Lateral controller analysis ({ctrl_type}, single integrator plant)\n'
        f'K_P={K_p:.3f}, K_I={K_i:.3f}, K_D={K_d:.3f}',
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
        f'(K_P={K_p:.3f}, K_I={K_i:.3f})',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_speed_schedule_heading(K_steer, L, wn, V_range_kmh, out_path):
    """
    Heading-error plant gain scales with V (not V^2), so gains scale with 1/V.
    Plot this so the user sees that the schedule is gentler than for the
    cross-track formulation.
    """
    V = np.array(V_range_kmh) / 3.6
    K_lat = V * K_steer / L                  # <-- single V, not V^2

    K_p = 2.0 * wn       / K_lat
    K_i = wn * wn        / K_lat

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(V_range_kmh, K_p, 'o-', label='K_P  ∝ 1/V')
    ax.plot(V_range_kmh, K_i, 's-', label='K_I  ∝ 1/V')
    ax.set_xlabel('vehicle speed V [km/h]')
    ax.set_ylabel('gain')
    ax.set_yscale('log')
    ax.set_title(f'Heading-error gain schedule (ω_n = {wn} rad/s)\n'
                 'gains scale ∝ 1/V (gentler than cross-track\'s 1/V²)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_schedule(K_steer, L, wn_heading, wn_xtrack,
                             V_range_kmh, out_path):
    """
    Side-by-side comparison: how aggressively must each formulation re-tune
    its gains as speed changes?
    """
    V = np.array(V_range_kmh) / 3.6

    # Heading-error plant: K_lat = V * K_steer / L
    K_lat_h = V * K_steer / L
    K_p_h = 2.0 * wn_heading / K_lat_h
    # Cross-track plant: K_lat = V^2 * K_steer / L
    K_lat_x = V**2 * K_steer / L
    K_p_x = 3.0 * wn_xtrack**2 / K_lat_x

    # Normalise both so they equal 1.0 at 30 km/h
    i30 = np.argmin(np.abs(np.array(V_range_kmh) - 30.0))
    K_p_h_n = K_p_h / K_p_h[i30]
    K_p_x_n = K_p_x / K_p_x[i30]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(V_range_kmh, K_p_h_n, 'o-', label='heading-error (1/V)')
    ax.plot(V_range_kmh, K_p_x_n, 's-', label='cross-track (1/V²)')
    ax.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(30.0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel('vehicle speed V [km/h]')
    ax.set_ylabel('K_P / K_P(at 30 km/h)')
    ax.set_yscale('log')
    ax.set_title('Why heading-error is friendlier to gain-schedule\n'
                 '(K_P normalised to 1.0 at 30 km/h)')
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
    p.add_argument('--wn-lat', type=float, default=2.0,
                   help='lateral closed-loop bandwidth [rad/s] (default 2.0). '
                        'Heading-error tolerates higher ω_n than cross-track.')
    p.add_argument('--wn-lon', type=float, default=1.0,
                   help='longitudinal closed-loop bandwidth [rad/s] (default 1.0)')
    p.add_argument('--zeta', type=float, default=1.0,
                   help='lateral damping ratio (default 1.0 = critical)')
    p.add_argument('--add-Kd', type=float, default=0.0,
                   help='optional small K_D for noise filtering '
                        '(matches CARLA structure; default 0.0)')
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

    L         = meta['wheelbase_m']
    u_step    = meta['lon_throttle_step']
    delta_cmd = meta['lat_steer_step']

    # --- ID ---
    K, tau = fit_longitudinal(lon_df, u_step)
    K_steer, V_lat, psi_dot_ss = fit_lateral_gain(lat_df, L, delta_cmd)
    print(f'[ID]  Longitudinal: K = {K:.3f} km/h per throttle, τ = {tau:.3f} s')
    print(f'[ID]  Lateral:      K_steer = {K_steer:.4f}  '
          f'(ψ̇_ss = {math.degrees(psi_dot_ss):.2f}°/s @ V={V_lat*3.6:.1f} km/h)')

    # --- Tune ---
    V_target = args.target_speed / 3.6
    K_lat_at_target = V_target * K_steer / L           # <-- single V
    lat_gains = tune_lateral_heading(
        K_lat_at_target, args.wn_lat, args.zeta, args.add_Kd
    )
    lon_gains = tune_longitudinal(K, tau, args.wn_lon)

    print()
    print(f'[tune] Lateral plant gain @ {args.target_speed} km/h: '
          f'V·K_steer/L = {K_lat_at_target:.3f}  (single integrator)')
    print(f'[tune] Lateral PI/PID  (ω_n = {args.wn_lat} rad/s, ζ = {args.zeta}):')
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
        os.path.join(args.outdir, 'src/heading_error_pid/fig1_identification.png'),
    )
    plot_lateral_heading_analysis(
        K_lat_at_target, lat_gains, args.wn_lat,
        os.path.join(args.outdir, 'src/heading_error_pid/fig2_lateral_heading_analysis.png'),
    )
    plot_longitudinal_analysis(
        K, tau, lon_gains, args.wn_lon,
        os.path.join(args.outdir, 'src/heading_error_pid/fig3_longitudinal_analysis.png'),
    )
    plot_speed_schedule_heading(
        K_steer, L, args.wn_lat, [10, 20, 30, 50, 70, 90, 110],
        os.path.join(args.outdir, 'src/heading_error_pid/fig4_gain_schedule_heading.png'),
    )
    plot_comparison_schedule(
        K_steer, L, args.wn_lat, args.wn_lat,
        [10, 20, 30, 50, 70, 90, 110],
        os.path.join(args.outdir, 'src/heading_error_pid/fig5_comparison.png'),
    )
    print(f'\n[done] Plots written to {args.outdir}/')


if __name__ == '__main__':
    main()
