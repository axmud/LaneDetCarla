"""
CARLA system identification recorder.

Runs two open-loop step experiments on the ego vehicle and saves the response
data to CSV, so that pid_tuning.py can fit transfer functions to it.

Experiment 1 -- LONGITUDINAL step:
    Hold steer = 0, brake the car to rest, then apply a constant throttle step.
    Log time, throttle command, and forward speed for several seconds.
    -> identifies K, tau in   V(s)/U(s) = K / (tau*s + 1)

Experiment 2 -- LATERAL gain identification:
    Hold the car at a constant target speed using a simple proportional
    throttle. Apply a small constant steer step. Log time, steer command,
    speed, and YAW RATE (not position -- yaw rate reaches steady state
    quickly and the car doesn't have to leave the road for us to measure it).
    -> identifies K_steer in   psi_dot = (V / L) * K_steer * delta_cmd
       which gives the effective lateral plant gain V^2 * K_steer / L.

Run on a long straight road. Town06 has long highway segments -- pick a spawn
point on one of them. Adjust SPAWN_INDEX below if the chosen spawn is bad.
"""

import argparse
import math
import time
import csv
import random

import carla


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOWN              = 'Town06'
SPAWN_INDEX       = None     # None -> random straight-ish spawn; or set an int
VEHICLE_FILTER    = 'vehicle.nissan.patrol'
DT                = 0.05     # 20 Hz, must match your control loop

# Longitudinal experiment
LON_THROTTLE_STEP = 0.5      # step amplitude applied at t = 0
LON_DURATION_S    = 10.0     # how long to log

# Lateral experiment
LAT_TARGET_SPEED  = 30.0     # km/h, held constant during the steer step
LAT_STEER_STEP    = 0.05     # small steer command (signed, in [-1, 1])
LAT_SETTLE_S      = 5.0      # seconds to reach target speed before steering
LAT_HOLD_S        = 2.5      # seconds to hold the steer step (kept short!)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def yaw_rate_dps(vehicle):
    """Yaw rate around z, in degrees/s (CARLA already returns deg/s)."""
    return vehicle.get_angular_velocity().z


def setup_world(client):
    world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    return world


def spawn_ego(world):
    bp = random.choice(world.get_blueprint_library().filter(VEHICLE_FILTER))
    spawn_points = world.get_map().get_spawn_points()
    if SPAWN_INDEX is not None:
        sp = spawn_points[SPAWN_INDEX]
    else:
        sp = random.choice(spawn_points)
    ego = world.try_spawn_actor(bp, sp)
    while ego is None:
        sp = random.choice(spawn_points)
        ego = world.try_spawn_actor(bp, sp)
    # let physics settle
    for _ in range(20):
        world.tick()
    return ego


def get_wheelbase(vehicle):
    """
    CARLA exposes wheel positions in physics control. Wheelbase = distance
    between front and rear axle midpoints.
    """
    pc = vehicle.get_physics_control()
    wheels = pc.wheels
    # wheels are usually [FL, FR, RL, RR] in cm (carla.Vector3D in cm units!)
    front = (wheels[0].position + wheels[1].position) * 0.5
    rear  = (wheels[2].position + wheels[3].position) * 0.5
    # convert from cm to m
    dx = (front.x - rear.x) / 100.0
    dy = (front.y - rear.y) / 100.0
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_longitudinal_step(world, ego, out_csv):
    print(f"[lon] Bringing vehicle to rest...")
    ctrl = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(int(2.0 / DT)):
        ego.apply_control(ctrl)
        world.tick()

    print(f"[lon] Applying throttle step = {LON_THROTTLE_STEP}")
    rows = []
    n_steps = int(LON_DURATION_S / DT)
    ctrl = carla.VehicleControl(throttle=LON_THROTTLE_STEP, brake=0.0, steer=0.0)
    for k in range(n_steps):
        ego.apply_control(ctrl)
        world.tick()
        rows.append({
            't': k * DT,
            'throttle': LON_THROTTLE_STEP,
            'speed_kmh': speed_kmh(ego),
        })

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t', 'throttle', 'speed_kmh'])
        w.writeheader()
        w.writerows(rows)
    print(f"[lon] Saved {len(rows)} samples -> {out_csv}")


def run_lateral_step(world, ego, out_csv):
    print(f"[lat] Settling vehicle at {LAT_TARGET_SPEED} km/h...")
    # crude P controller on speed for the settling phase
    n_settle = int(LAT_SETTLE_S / DT)
    for _ in range(n_settle):
        v = speed_kmh(ego)
        err = LAT_TARGET_SPEED - v
        thr = max(0.0, min(0.6, 0.05 * err))
        brk = 0.0 if err > -2 else 0.2
        ego.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=0.0))
        world.tick()

    print(f"[lat] Applying steer step = {LAT_STEER_STEP}")
    rows = []
    n_hold = int(LAT_HOLD_S / DT)
    for k in range(n_hold):
        v = speed_kmh(ego)
        err = LAT_TARGET_SPEED - v
        thr = max(0.0, min(0.6, 0.05 * err))
        ego.apply_control(carla.VehicleControl(
            throttle=thr, brake=0.0, steer=LAT_STEER_STEP
        ))
        world.tick()
        rows.append({
            't': k * DT,
            'steer_cmd': LAT_STEER_STEP,
            'speed_kmh': speed_kmh(ego),
            'yaw_rate_dps': yaw_rate_dps(ego),
        })

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t', 'steer_cmd', 'speed_kmh', 'yaw_rate_dps'])
        w.writeheader()
        w.writerows(rows)
    print(f"[lat] Saved {len(rows)} samples -> {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', default=2000, type=int)
    p.add_argument('--lon-out', default='sysid_longitudinal.csv')
    p.add_argument('--lat-out', default='sysid_lateral.csv')
    p.add_argument('--meta-out', default='sysid_meta.csv')
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = setup_world(client)
    ego = spawn_ego(world)
    L = get_wheelbase(ego)
    print(f"[info] Ego = {ego.type_id}, wheelbase L = {L:.3f} m")

    try:
        run_longitudinal_step(world, ego, args.lon_out)
        run_lateral_step(world, ego, args.lat_out)

        with open(args.meta_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['key', 'value'])
            w.writerow(['vehicle', ego.type_id])
            w.writerow(['wheelbase_m', L])
            w.writerow(['dt_s', DT])
            w.writerow(['lon_throttle_step', LON_THROTTLE_STEP])
            w.writerow(['lat_target_speed_kmh', LAT_TARGET_SPEED])
            w.writerow(['lat_steer_step', LAT_STEER_STEP])
        print(f"[info] Saved metadata -> {args.meta_out}")

    finally:
        if ego is not None and ego.is_alive:
            ego.destroy()
        # restore async mode so other scripts behave
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)


if __name__ == '__main__':
    main()
