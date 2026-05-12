"""
Draw recorded trajectories on top of the live CARLA world.

Usage:
    python src/draw_trajectory_in_carla.py
    python src/draw_trajectory_in_carla.py --log src/ego_trajectory.csv
    python src/draw_trajectory_in_carla.py --logs src/run_a.csv src/run_b.csv
    python src/draw_trajectory_in_carla.py --no-ref --life 30
"""
import argparse
import carla
import pandas as pd


# Distinct colors for overlaying multiple runs
PALETTE = [
    carla.Color(0, 200, 255),    # cyan
    carla.Color(255, 100, 0),    # orange
    carla.Color(0, 255, 100),    # green
    carla.Color(255, 0, 200),    # magenta
    carla.Color(255, 255, 0),    # yellow
]
REF_COLOR = carla.Color(180, 180, 180)  # gray for reference path


def get_ground_z(world, x, y, fallback_z=0.5):
    """Snap (x, y) to the road surface so the line isn't buried or floating."""
    wp = world.get_map().get_waypoint(
        carla.Location(x=x, y=y, z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    return wp.transform.location.z + 0.5 if wp else fallback_z


def draw_path(world, xs, ys, color, life_time, thickness, z_offset=0.5):
    """Draw a polyline in CARLA by stitching debug line segments."""
    debug = world.debug
    for i in range(len(xs) - 1):
        z1 = get_ground_z(world, xs[i],   ys[i])   + z_offset
        z2 = get_ground_z(world, xs[i+1], ys[i+1]) + z_offset
        debug.draw_line(
            carla.Location(x=float(xs[i]),   y=float(ys[i]),   z=z1),
            carla.Location(x=float(xs[i+1]), y=float(ys[i+1]), z=z2),
            thickness=thickness,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )


def draw_endpoints(world, xs, ys, life_time):
    """Mark start (green) and end (red) of a trajectory."""
    debug = world.debug
    z0 = get_ground_z(world, xs.iloc[0],  ys.iloc[0])  + 1.0
    z1 = get_ground_z(world, xs.iloc[-1], ys.iloc[-1]) + 1.0
    debug.draw_point(
        carla.Location(x=float(xs.iloc[0]), y=float(ys.iloc[0]), z=z0),
        size=0.2, color=carla.Color(0, 255, 0), life_time=life_time,
    )
    debug.draw_point(
        carla.Location(x=float(xs.iloc[-1]), y=float(ys.iloc[-1]), z=z1),
        size=0.2, color=carla.Color(255, 0, 0), life_time=life_time,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=["src/ego_trajectory_vision.csv"],
                        help="One or more trajectory CSVs to overlay")
    parser.add_argument("--ref", default="src/carla_data.csv",
                        help="Reference path CSV")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip drawing the reference path")
    parser.add_argument("--life", type=float, default=0.0,
                        help="Lifetime in seconds (0 = persist until world reload)")
    parser.add_argument("--thickness", type=float, default=0.025,
                        help="Line thickness in meters")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world("Town06")  # Load a specific map

    # -------- Reference path -------------------------------------------------
    if not args.no_ref:
        ref = pd.read_csv(args.ref, index_col=0)
        print(f"Drawing reference path ({len(ref)} points)...")
        draw_path(world,
                  ref["waypoint_x"], ref["waypoint_y"],
                  color=REF_COLOR,
                  life_time=args.life,
                  thickness=args.thickness * 0.7)  # thinner for the ref

    # -------- Logged trajectories -------------------------------------------
    for i, log_path in enumerate(args.logs):
        log = pd.read_csv(log_path)
        color = PALETTE[i % len(PALETTE)]
        print(f"Drawing {log_path} ({len(log)} points) "
              f"in RGB({color.r},{color.g},{color.b})...")
        draw_path(world,
                  log["x"], log["y"],
                  color=color,
                  life_time=args.life,
                  thickness=args.thickness)
        draw_endpoints(world, log["x"], log["y"], life_time=args.life)

    print("Done. Switch to the CARLA spectator window to view.")


if __name__ == "__main__":
    main()