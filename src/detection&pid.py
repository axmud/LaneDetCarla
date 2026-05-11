import carla
import random
import pygame
import math
import numpy as np
import pandas as pd
import cv2
from collections import deque

# =============================================================================
# CARLA / world setup
# =============================================================================
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town06')
delta_seconds = 0.05  # 20 Hz

settings = world.get_settings()
settings.synchronous_mode = True
settings.no_rendering_mode = True
settings.fixed_delta_seconds = delta_seconds
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

traffic_manager.set_random_device_seed(0)
random.seed(0)

spawn_points = world.get_map().get_spawn_points()

# =============================================================================
# Spawn traffic
# =============================================================================
# Renamed from `models` to avoid clash with the detection-models dict below.
vehicle_models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln',
                  'prius', 'nissan', 'crown', 'impala']
blueprints = []
for vehicle in world.get_blueprint_library().filter('*vehicle*'):
    if any(m in vehicle.id for m in vehicle_models):
        blueprints.append(vehicle)

max_vehicles = 50
max_vehicles = min([max_vehicles, len(spawn_points)])
vehicles = []
for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
    temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
    if temp is not None:
        vehicles.append(temp)

for vehicle in vehicles:
    vehicle.set_autopilot(True)
    traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))


# =============================================================================
# Rendering + lane detection callback
# =============================================================================
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


# Latest detected centerline expressed in the *vehicle frame*: shape (M, 2),
# where column 0 = X forward (m), column 1 = Y right (m). May be None until
# the first detection lands; may have shape (0, 2) if the IPM filter kills
# every sample (e.g. all above the horizon).
centerline_world = None


def pygame_callback(data, obj):
    """
    Camera callback: runs lane detection on the RGB image, projects the
    detected centerline onto the ground plane in the ego's body frame, and
    stores it in `centerline_world` for the main loop to consume.
    """
    global centerline_world

    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]            # drop alpha
    img = img[:, :, ::-1]          # BGRA -> RGB
    bgr_array = img[:, :, ::-1].copy()

    centerline, img_drawn = detect_lanes(
        bgr_array,
        model, cfg, transform,
        draw=True,
    )

    if centerline is not None:
        centerline_world = pixel_to_vehicle(
            centerline, 1.3 * bound_z,
            image_width=1280, image_height=720,
            fov_deg=90.0, cam_x_offset=0.8 * bound_x, max_range=40.0,
        )
    # else: keep the previous centerline_world; the main loop's `is None /
    # shape[0] > 0` checks plus the `fx_body > 0.1` guard handle staleness.

    obj.surface = pygame.surfarray.make_surface(img_drawn.swapaxes(0, 1))


# =============================================================================
# Data collectors (unchanged)
# =============================================================================
def carla_data_collector(vehicle: carla.Vehicle):
    frame = vehicle.get_world().get_snapshot().frame
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    waypoint = vehicle.get_world().get_map().get_waypoint(location)

    return {
        "frame": frame,
        "location": {"x": location.x, "y": location.y, "z": location.z},
        "rotation": {"pitch": rotation.pitch, "yaw": rotation.yaw, "roll": rotation.roll},
        "velocity": {"x": velocity.x, "y": velocity.y, "z": velocity.z},
        "acceleration": {"x": acceleration.x, "y": acceleration.y, "z": acceleration.z},
        "angular_velocity": {"x": angular_velocity.x, "y": angular_velocity.y, "z": angular_velocity.z},
        "waypoint": {"x": waypoint.transform.location.x,
                     "y": waypoint.transform.location.y,
                     "z": waypoint.transform.location.z},
    }


def carla_data_collector_2(vehicle: carla.Vehicle):
    waypoint = vehicle.get_world().get_map().get_waypoint(vehicle.get_transform().location)
    loc = waypoint.transform.location
    fwd = waypoint.transform.get_forward_vector()
    return {"x": loc.x, "y": loc.y, "vector_x": fwd.x, "vector_y": fwd.y}


# =============================================================================
# Path geometry
# =============================================================================
index = 0
FUTURE_HORIZON = 10

# Look-ahead policy (CARLA-style: distance grows with speed).
# Longer look-ahead = more robust to per-sample noise in the detected
# centerline, at the cost of slightly cutting tight corners.
LOOKAHEAD_MIN = 4.0   # m
LOOKAHEAD_K   = 0.6   # s   ->  Ld = LOOKAHEAD_MIN + LOOKAHEAD_K * v_mps

# First-order low-pass on the heading error fed to the PID. Helps when the
# detector updates slower than the control loop or produces jittery samples.
# 1.0 = no filtering, 0.0 = frozen. 0.4 is a reasonable starting point.
H_ERR_ALPHA = 0.4
_h_err_filt = 0.0


def vision_target(centerline_body, Ld, x_min=0.5, x_max=20.0, deg=1):
    """
    Fit a polynomial y = f(x) to the body-frame centerline and evaluate at
    x = Ld. Returns (Ld, y) or None if too few inliers.

    deg=1 (line) is robust to per-sample lateral noise and is the right
    default for a 30 km/h cruise on Town06's mostly-straight roads. Bump to
    deg=2 if you want to track curves at higher speeds.
    """
    if centerline_body is None or centerline_body.shape[0] < 3:
        return None
    pts = centerline_body[(centerline_body[:, 0] > x_min) &
                          (centerline_body[:, 0] < x_max)]
    if pts.shape[0] < 3:
        return None
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], deg=deg)
    y_at_Ld = float(np.polyval(coeffs, Ld))
    return Ld, y_at_Ld


def lane_shift_calculator(vehicle: carla.Vehicle):
    """
    Advances the global `index` to the nearest waypoint within the look-ahead
    window and returns:
        lane_shift   - signed perpendicular distance to the nearest waypoint (m)
        nearest_dist - Euclidean distance to that waypoint (m)
    """
    global index

    loc = vehicle.get_transform().location
    car_x, car_y = loc.x, loc.y

    end = min(index + FUTURE_HORIZON + 1, len(wp_x_arr))
    dx = wp_x_arr[index:end] - car_x
    dy = wp_y_arr[index:end] - car_y
    sq_dist = dx * dx + dy * dy
    offset = int(np.argmin(sq_dist))
    nearest_index = index + offset
    index = nearest_index

    vx = vec_x_arr[nearest_index]
    vy = vec_y_arr[nearest_index]
    lane_shift = dx[offset] * vy - dy[offset] * vx

    nearest_dist = float(np.sqrt(sq_dist[offset]))
    return lane_shift, nearest_dist


def get_lookahead_target(vehicle: carla.Vehicle):
    """
    Returns (target_x, target_y, target_idx): the first waypoint at distance
    >= Ld ahead of the current `index`. Clamps to the last waypoint when near
    the end of the path.
    """
    loc = vehicle.get_transform().location
    car_x, car_y = loc.x, loc.y

    v = vehicle.get_velocity()
    speed_mps = math.sqrt(v.x ** 2 + v.y ** 2)
    Ld = LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps
    Ld_sq = Ld * Ld

    for j in range(index, len(wp_x_arr)):
        ddx = wp_x_arr[j] - car_x
        ddy = wp_y_arr[j] - car_y
        if ddx * ddx + ddy * ddy >= Ld_sq:
            return float(wp_x_arr[j]), float(wp_y_arr[j]), j

    last = len(wp_x_arr) - 1
    return float(wp_x_arr[last]), float(wp_y_arr[last]), last


def heading_error(vehicle: carla.Vehicle, target_x: float, target_y: float):
    """
    Signed angle (radians) between the vehicle's forward vector and the vector
    from vehicle to target. CARLA convention: positive => target is to the
    right => steer right.
    """
    tf = vehicle.get_transform()
    fwd = tf.get_forward_vector()
    v_vec = np.array([fwd.x, fwd.y, 0.0])
    w_vec = np.array([target_x - tf.location.x,
                      target_y - tf.location.y,
                      0.0])

    norm = np.linalg.norm(v_vec) * np.linalg.norm(w_vec)
    if norm < 1e-6:
        return 0.0

    angle = math.acos(np.clip(np.dot(v_vec, w_vec) / norm, -1.0, 1.0))
    cross_z = v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]
    if cross_z < 0:
        angle = -angle
    return angle


def find_initial_index(vehicle, max_dist_m=20.0, min_align_cos=0.0):
    """
    Snap `index` to the nearest waypoint on the recorded path whose tangent
    points in roughly the same direction as the vehicle.

    Returns -1 if no waypoint within max_dist_m matches the vehicle's
    direction (i.e. the recorded path is for the other lane / a different
    route). The caller should treat -1 as "path mode disabled".
    """
    tf = vehicle.get_transform()
    fwd = tf.get_forward_vector()
    fx, fy = fwd.x, fwd.y
    cx, cy = tf.location.x, tf.location.y

    dx = wp_x_arr - cx
    dy = wp_y_arr - cy
    sq_dist = dx * dx + dy * dy

    # cos(angle between path tangent and ego forward); tangent is unit length.
    align = vec_x_arr * fx + vec_y_arr * fy

    sq_dist_masked = np.where(align > min_align_cos, sq_dist, np.inf)
    if not np.isfinite(sq_dist_masked).any():
        return -1

    best = int(np.argmin(sq_dist_masked))
    if sq_dist_masked[best] > max_dist_m ** 2:
        return -1
    return best


# =============================================================================
# PID controllers
# =============================================================================
class PIDLateralController:
    """Discrete PID on heading error (radians). Output clipped to [-1, 1]."""

    def __init__(self, K_P=1.95, K_I=0.05, K_D=0.2, dt=delta_seconds):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, heading_err):
        self._e_buffer.append(heading_err)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return float(np.clip(
            self._K_P * heading_err + self._K_D * _de + self._K_I * _ie,
            -1.0, 1.0
        ))


class PIDLongitudinalController:
    """Discrete PID on speed error (km/h). Output is signed accel in [-1, 1]."""

    def __init__(self, K_P=0.3, K_I=0.05, K_D=0.0, dt=delta_seconds):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_speed, current_speed):
        error = target_speed - current_speed
        self._e_buffer.append(error)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return float(np.clip(
            self._K_P * error + self._K_D * _de + self._K_I * _ie,
            -1.0, 1.0
        ))


class VehiclePIDController:
    """
    Two interfaces:
        run_step(target_speed, target_x, target_y) - world-frame waypoint;
            heading error is computed internally via heading_error().
        run_step_h(target_speed, heading_err)      - heading error is supplied
            directly (radians). Use this when the target is in the body frame
            (e.g. a camera-detected centerline point) — no world<->body
            transform needed.
    """

    def __init__(self, vehicle,
                 args_lateral=None, args_longitudinal=None,
                 max_throttle=0.75, max_brake=0.3, max_steering=0.8):

        if args_lateral is None:
            args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': delta_seconds}
        if args_longitudinal is None:
            args_longitudinal = {'K_P': 0.3, 'K_I': 0.05, 'K_D': 0.0, 'dt': delta_seconds}

        self._vehicle = vehicle
        self._max_throt = max_throttle
        self._max_brake = max_brake
        self._max_steer = max_steering

        self._lat_controller = PIDLateralController(**args_lateral)
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)

        self.past_steering = 0.0

    def set_vehicle(self, vehicle):
        """Reuse the same controller on a new vehicle (resets past steering)."""
        self._vehicle = vehicle
        self.past_steering = 0.0

    def reset(self):
        """
        Clear integral / derivative state. Call after a control hand-off (e.g.
        autopilot drove through a junction) so we don't re-apply integral
        wind-up against errors the PID never actually saw.
        """
        self._lat_controller._e_buffer.clear()
        self._lon_controller._e_buffer.clear()
        self.past_steering = 0.0

    # --- Internal: assemble carla.VehicleControl from accel + steer ----------
    def _build_control(self, acceleration, steering):
        control = carla.VehicleControl()

        # Longitudinal: signed acceleration -> throttle / brake
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self._max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self._max_brake)

        # Lateral: rate-limit steering to avoid jitter
        if steering > self.past_steering + 0.1:
            steering = self.past_steering + 0.1
        elif steering < self.past_steering - 0.1:
            steering = self.past_steering - 0.1

        steering = max(-self._max_steer, min(self._max_steer, steering))

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False

        self.past_steering = steering
        return control

    def run_step(self, target_speed, target_x, target_y):
        v = self._vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        h_err = heading_error(self._vehicle, target_x, target_y)

        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        steering     = self._lat_controller.run_step(h_err)
        return self._build_control(acceleration, steering)

    def run_step_h(self, target_speed, heading_err):
        """Variant that takes heading error directly (already in radians)."""
        v = self._vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        steering     = self._lat_controller.run_step(heading_err)
        return self._build_control(acceleration, steering)


# =============================================================================
# Manual control fallback (kept so you can compare PID vs. keyboard)
# =============================================================================
class ControlObject(object):
    def __init__(self, veh):
        self._vehicle = veh
        self._throttle = False
        self._brake = False
        self._steer = None
        self._steer_cache = 0
        self._control = carla.VehicleControl()

    def parse_control(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._throttle = True
            if event.key == pygame.K_DOWN:
                self._brake = True
            if event.key == pygame.K_RIGHT:
                self._steer = 1
            if event.key == pygame.K_LEFT:
                self._steer = -1
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                self._throttle = False
            if event.key == pygame.K_DOWN:
                self._brake = False
                self._control.reverse = False
            if event.key == pygame.K_RIGHT:
                self._steer = None
            if event.key == pygame.K_LEFT:
                self._steer = None

    def process_control(self):
        if self._throttle:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
            self._control.gear = 1
            self._control.brake = False
        elif not self._brake:
            self._control.throttle = 0.0

        if self._brake:
            if self._vehicle.get_velocity().length() < 0.01 and not self._control.reverse:
                self._control.brake = 0.0
                self._control.gear = 1
                self._control.reverse = True
                self._control.throttle = min(self._control.throttle + 0.1, 1)
            elif self._control.reverse:
                self._control.throttle = min(self._control.throttle + 0.1, 1)
            else:
                self._control.throttle = 0.0
                self._control.brake = min(self._control.brake + 0.3, 1)
        else:
            self._control.brake = 0.0

        if self._steer is not None:
            if self._steer == 1:
                self._steer_cache += 0.03
            if self._steer == -1:
                self._steer_cache -= 0.03
            self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache, 1)
        else:
            if self._steer_cache > 0.0:
                self._steer_cache *= 0.2
            if self._steer_cache < 0.0:
                self._steer_cache *= 0.2
            if 0.01 > self._steer_cache > -0.01:
                self._steer_cache = 0.0
            self._control.steer = round(self._steer_cache, 1)

        self._vehicle.apply_control(self._control)


class TrajectoryLogger:
    def __init__(self, enabled=True, output_path="ego_trajectory_heading.csv"):
        self.enabled = enabled
        self.output_path = output_path
        self._rows = []

    def log(self, **fields):
        if not self.enabled:
            return
        self._rows.append(fields)

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[LOG] {'ENABLED' if self.enabled else 'DISABLED'} "
              f"({len(self._rows)} samples buffered)")

    def set_output_path(self, path):
        self.output_path = path
        print(f"[LOG] Output path -> {path}")

    def save(self):
        if not self._rows:
            print("[LOG] Nothing to save.")
            return
        df = pd.DataFrame(self._rows)
        df.to_csv(self.output_path, index=False)
        print(f"[LOG] Saved {len(df)} samples to {self.output_path}")

    def clear(self):
        self._rows.clear()


# =============================================================================
# Lane-detection model registry + IPM helper
# =============================================================================
detection_models = {
    "SCNN_Tusimple": {
        "Config": "config/scnn/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/SCNN/ResNet18_AAAI.pth",
    },
    "RESA_Tusimple": {
        "Config": "config/resa/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/RESA/ResNet18_AAAI.pth",
    },
    "UFLD_Tusimple": {
        "Config": "config/ufld/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/UFLD/ResNet18_ECCV.pth",
    },
    "CLRNet_Tusimple": {
        "Config": "config/clrnet/resnet34_tusimple.py",
        "Weight": "releases/Tusimple/CLRNet/ResNet34_CVPR.pth",
    },
    "LaneATT_Tusimple": {
        "Config": "config/laneatt/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/LaneATT/ResNet34_CVPR.pth",
    },
    "ADNet_Tusimple": {
        "Config": "config/adnet/resnet34_tusimple.py",
        "Weight": "releases/Tusimple/ADNet/ResNet34_ICCV.pth",
    },
    "SRLane_Tusimple": {
        "Config": "config/srlane/resnet34_tusimple.py",
        "Weight": "releases/Tusimple/SRLane/ResNet34_AAAI.pth",
    },
    "BezierNet_Tusimple": {
        "Config": "config/beziernet/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/BezierNet/ResNet18_CVPR.pth",
    },
    "GANet_Tusimple": {
        "Config": "config/ganet/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/GANet/ResNet18_CVPR.pth",
    },
    "GSENet_Tusimple": {
        "Config": "config/gsenet/resnet18_tusimple.py",
        "Weight": "releases/Tusimple/GSENet/ResNet18_AAAI.pth",
    },
    "CLRNet_CULane_ResNet34": {
        "Config": "config/clrnet/resnet34_culane.py",
        "Weight": "releases/CULane/CLRNet/ResNet34_CVPR.pth",
    },
    "CLRNet_CULane_ResNet50": {
        "Config": "config/clrnet/resnet50_culane.py",
        "Weight": "releases/CULane/CLRNet/ResNet50_CVPR.pth",
    },
    "CLRNet_CULane_ConvNexT-Tiny": {
        "Config": "config/clrnet/convnext_culane.py",
        "Weight": "releases/CULane/CLRNet/ConvNext-Tiny_CVPR.pth",
    },
    "CLRerNet_CULane_ResNet34": {
        "Config": "config/clrernet/resnet34_culane.py",
        "Weight": "releases/CULane/CLRerNet/ResNet34_WACV.pth",
    },
    "CLRerNet_CULane_ConvNexT-Tiny": {
        "Config": "config/clrernet/convnext_culane.py",
        "Weight": "releases/CULane/CLRerNet/ConvNext-Tiny_WACV.pth",
    },
    "ADNet_CULane": {
        "Config": "config/adnet/resnet34_culane.py",
        "Weight": "releases/CULane/ADNet/ResNet34_ICCV.pth",
    },
    "ADNet_VIL100": {
        "Config": "config/adnet/resnet34_vil.py",
        "Weight": "releases/VIL100/ADNet/ResNet34_ICCV.pth",
    },
}

from helper import load_model, detect_lanes


def pixel_to_vehicle(pixel_pts, cam_height,
                     image_width=1280, image_height=720,
                     fov_deg=90.0, cam_x_offset=0.0,
                     max_range=40.0):
    """
    Project image-plane points onto the ground plane in the vehicle frame,
    assuming a pinhole camera with zero pitch/roll/yaw and flat ground.

    Vehicle frame (CARLA convention): +X forward, +Y right, +Z up.
    Returns (M, 2) ndarray of [X_forward, Y_right] in meters.
    """
    p = np.asarray(pixel_pts, dtype=np.float64)
    u, v = p[:, 0], p[:, 1]

    fx = fy = image_width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cx, cy = image_width / 2.0, image_height / 2.0

    below_horizon = v > cy + 1e-3
    u, v = u[below_horizon], v[below_horizon]
    if u.size == 0:
        return np.empty((0, 2))

    X_cam = cam_height * fy / (v - cy)
    Y_cam = X_cam * (u - cx) / fx

    pts = np.column_stack([X_cam + cam_x_offset, Y_cam])
    return pts[pts[:, 0] <= max_range]


# =============================================================================
# Pick the ego vehicle and disable its autopilot so the PID can drive it
# =============================================================================
ego_vehicle = random.choice(vehicles)
ego_vehicle.set_autopilot(False)

# Camera rig — bound_x / bound_z are read by pygame_callback as globals.
bound_x = 0.5 + ego_vehicle.bounding_box.extent.x
bound_y = 0.5 + ego_vehicle.bounding_box.extent.y
bound_z = 0.5 + ego_vehicle.bounding_box.extent.z
Attachment = carla.AttachmentType

camera_init_trans = carla.Transform(
    carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)
)
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "90")  # set BEFORE spawn so it actually applies
camera = world.spawn_actor(camera_bp, camera_init_trans,
                           attach_to=ego_vehicle, attachment_type=Attachment.Rigid)

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

renderObject = RenderObject(image_w, image_h)
controlObject = ControlObject(ego_vehicle)

# -----------------------------------------------------------------------------
# Build the PID controller for the ego vehicle
# -----------------------------------------------------------------------------
TARGET_SPEED = 30.0  # km/h

LOG_TRAJECTORY  = True
LOG_OUTPUT_PATH = "src/ego_trajectory_vision.csv"
LOG_TOGGLE_KEY  = pygame.K_l

vehicle_pid = VehiclePIDController(
    ego_vehicle,
    # Heading-error PID gains. K_I deliberately small (~K_P/20) — a large K_I
    # on a heading-error plant winds up faster than the steering can respond
    # and causes oscillation. K_D damps high-frequency wobble from the vision
    # feed.
    args_lateral={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.1, 'dt': delta_seconds},
    args_longitudinal={'K_P': 0.3143, 'K_I': 0.05, 'K_D': 0.0000, 'dt': delta_seconds},
    max_throttle=0.75,
    max_brake=0.3,
    max_steering=0.8,
)

pid_enabled = True

# -----------------------------------------------------------------------------
# PyGame init
# -----------------------------------------------------------------------------
pygame.init()
gameDisplay = pygame.display.set_mode((image_w, image_h),
                                      pygame.HWSURFACE | pygame.DOUBLEBUF)
gameDisplay.fill((0, 0, 0))
gameDisplay.blit(renderObject.surface, (0, 0))
pygame.display.flip()

# Load the pre-recorded path (waypoint_x, waypoint_y, vector_x, vector_y)
data = pd.read_csv("src/carla_data.csv", index_col=0)
wp_x_arr  = data["waypoint_x"].to_numpy()
wp_y_arr  = data["waypoint_y"].to_numpy()
vec_x_arr = data["vector_x"].to_numpy()
vec_y_arr = data["vector_y"].to_numpy()

# Direction-aware initial snap to the recorded path. If the chosen ego is on
# the wrong route (or the wrong side of the road), find_initial_index returns
# -1 and we disable path mode entirely — vision-only will run instead.
init_idx = find_initial_index(ego_vehicle)
if init_idx < 0:
    print("[PATH] Recorded path doesn't match this vehicle's direction. "
          "Path mode disabled — vision only.")
    path_valid = False
    index = 0
else:
    path_valid = True
    index = init_idx
    print(f"[PATH] Snapped to waypoint {index}")

# -----------------------------------------------------------------------------
# Pre-compute junction bounding boxes (extends "junction mode" by a buffer
# zone where lane markings often disappear before is_junction flips).
# -----------------------------------------------------------------------------
carla_map = world.get_map()
_junction_seed_locs = [
    carla.Location(x=648.76, y=138.43, z=0),  # junction_id=332
    carla.Location(x=658.88, y=64.32,  z=0),  # junction_id=268
    carla.Location(x=518.62, y=-10.41, z=0),  # junction_id=1203
]
_junction_waypoints = [carla_map.get_waypoint(loc) for loc in _junction_seed_locs]
junction_bboxes = [wp.get_junction().bounding_box for wp in _junction_waypoints]

logger = TrajectoryLogger(enabled=LOG_TRAJECTORY, output_path=LOG_OUTPUT_PATH)

# -----------------------------------------------------------------------------
# Load detection model (after camera so first callback has it ready)
# -----------------------------------------------------------------------------
model, cfg, transform = load_model(detection_models["CLRNet_Tusimple"])

# Now safe to start the camera (the callback needs `model`, `cfg`, `transform`,
# `bound_x`, `bound_z` to all exist before the first frame lands).
camera.listen(lambda image: pygame_callback(image, renderObject))


# =============================================================================
# Main loop
# =============================================================================
crashed = False
target_idx = 0

# Tracks whether the previous tick had us in a junction, for edge-detection of
# enter/exit transitions. Start False so the first tick computes correctly.
was_in_junction = False

# Make sure the ego is under our control at startup (the spawn loop set
# autopilot ON for every vehicle).
ego_vehicle.set_autopilot(False)

while not crashed:
    world.tick()

    ego_loc = ego_vehicle.get_transform().location
    ego_wp  = carla_map.get_waypoint(ego_loc)

    # --- Junction detection (waypoint flag OR pre-flagged buffer zone) -------
    in_junction = ego_wp.is_junction or any(
        jb.contains(ego_loc, carla.Transform()) for jb in junction_bboxes
    )

    # --- Edge transitions: hand off to / take back from traffic manager ------
    if in_junction and not was_in_junction:
        # Just entered a junction — let the TM drive through it.
        ego_vehicle.set_autopilot(True)
        print("[MODE] Junction entered -> autopilot ON")
    elif not in_junction and was_in_junction:
        # Just exited — take control back, but reset PID state first so
        # accumulated integral / past-steering from before the handoff don't
        # immediately saturate the actuator.
        ego_vehicle.set_autopilot(False)
        vehicle_pid.reset()
        _h_err_filt = 0.0

        # The TM may have changed lanes through the junction — re-snap the
        # path cursor to wherever we actually came out.
        init_idx = find_initial_index(ego_vehicle)
        if init_idx >= 0:
            path_valid = True
            index = init_idx
            print(f"[MODE] Junction exited -> PID ON, path snapped to idx {index}")
        else:
            path_valid = False
            print("[MODE] Junction exited -> PID ON, no matching path "
                  "(vision only)")

    was_in_junction = in_junction

    # =========================================================================
    # Inside a junction: autopilot is driving. Skip PID, just log and render.
    # =========================================================================
    if in_junction:
        # Snapshot for logging only — don't touch path cursor or PID state.
        rot = ego_vehicle.get_transform().rotation
        vel = ego_vehicle.get_velocity()
        applied = ego_vehicle.get_control()
        snap = world.get_snapshot()

        logger.log(
            frame        = snap.frame,
            time_s       = snap.timestamp.elapsed_seconds,
            x            = ego_loc.x,
            y            = ego_loc.y,
            z            = ego_loc.z,
            yaw_deg      = rot.yaw,
            speed_kmh    = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
            lane_shift   = float('nan'),
            nearest_dist = float('nan'),
            ref_index    = index,
            ref_x        = float('nan'),
            ref_y        = float('nan'),
            steer        = applied.steer,
            throttle     = applied.throttle,
            brake        = applied.brake,
            source       = "junction (autopilot)",
        )

        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            controlObject.parse_control(event)
            if event.type == pygame.KEYUP:
                if event.key == LOG_TOGGLE_KEY:
                    logger.toggle()
                if event.key == pygame.K_RETURN:
                    pid_enabled = not pid_enabled
                    print(f"[PID] {'ENABLED' if pid_enabled else 'DISABLED (manual)'}")
                if event.key == pygame.K_TAB:
                    # Same TAB-handling block as the non-junction branch below.
                    # Resetting was_in_junction = False makes the next tick
                    # re-evaluate the new ego from scratch.
                    ego_vehicle.set_autopilot(True)
                    ego_vehicle = random.choice(vehicles)
                    if ego_vehicle.is_alive:
                        ego_vehicle.set_autopilot(False)
                        camera.stop()
                        camera.destroy()

                        bound_x = 0.5 + ego_vehicle.bounding_box.extent.x
                        bound_y = 0.5 + ego_vehicle.bounding_box.extent.y
                        bound_z = 0.5 + ego_vehicle.bounding_box.extent.z
                        camera_init_trans = carla.Transform(
                            carla.Location(x=+0.8 * bound_x,
                                           y=+0.0 * bound_y,
                                           z=1.3 * bound_z)
                        )
                        camera = world.spawn_actor(camera_bp, camera_init_trans,
                                                   attach_to=ego_vehicle)
                        camera.listen(lambda image: pygame_callback(image, renderObject))

                        controlObject = ControlObject(ego_vehicle)
                        vehicle_pid.set_vehicle(ego_vehicle)

                        init_idx = find_initial_index(ego_vehicle)
                        if init_idx < 0:
                            path_valid = False
                            index = 0
                            print("[PATH] New vehicle direction doesn't match "
                                  "recorded path — vision only.")
                        else:
                            path_valid = True
                            index = init_idx
                            print(f"[PATH] Snapped to waypoint {index}")

                        centerline_world = None
                        _h_err_filt = 0.0
                        was_in_junction = False  # let next tick re-evaluate

                        gameDisplay.fill((0, 0, 0))
                        gameDisplay.blit(renderObject.surface, (0, 0))
                        pygame.display.flip()
        continue  # done with this tick — autopilot handles the driving

    # =========================================================================
    # Outside a junction: PID drives the car (vision -> path fallback).
    # =========================================================================

    # Advance the path cursor (only when the path is valid; cheap, keeps
    # `index` glued to the nearest CSV waypoint).
    if path_valid:
        lane_shift, nearest_dist = lane_shift_calculator(ego_vehicle)
    else:
        lane_shift, nearest_dist = float('nan'), float('nan')

    h_err  = None
    source = None
    target_x = target_y = None

    # ---- Vision mode (body-frame centerline) -------------------------------
    if centerline_world is not None and centerline_world.shape[0] >= 3:
        v = ego_vehicle.get_velocity()
        speed_mps = math.sqrt(v.x ** 2 + v.y ** 2)
        Ld = LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps

        target = vision_target(centerline_world, Ld)
        if target is not None:
            fx_body, fy_body = target
            if fx_body > 0.1:
                h_err  = math.atan2(fy_body, fx_body)
                source = "vision"

    # ---- Fallback: recorded path -------------------------------------------
    if h_err is None and path_valid:
        target_x, target_y, target_idx = get_lookahead_target(ego_vehicle)
        # Defensive: if look-ahead target is somehow behind us, drop to 0
        fwd = ego_vehicle.get_transform().get_forward_vector()
        if (target_x - ego_loc.x) * fwd.x + (target_y - ego_loc.y) * fwd.y > 0:
            h_err  = heading_error(ego_vehicle, target_x, target_y)
            source = "path (vision fallback)"

    # ---- Last resort: hold heading -----------------------------------------
    if h_err is None:
        h_err  = 0.0
        source = "no-data (h_err=0)"

    # --- Log ego trajectory --------------------------------------------------
    rot = ego_vehicle.get_transform().rotation
    vel = ego_vehicle.get_velocity()
    applied = ego_vehicle.get_control()
    snap = world.get_snapshot()

    logger.log(
        frame        = snap.frame,
        time_s       = snap.timestamp.elapsed_seconds,
        x            = ego_loc.x,
        y            = ego_loc.y,
        z            = ego_loc.z,
        yaw_deg      = rot.yaw,
        speed_kmh    = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
        lane_shift   = lane_shift,
        nearest_dist = nearest_dist,
        ref_index    = index,
        ref_x        = float(wp_x_arr[index]) if path_valid else float('nan'),
        ref_y        = float(wp_y_arr[index]) if path_valid else float('nan'),
        steer        = applied.steer,
        throttle     = applied.throttle,
        brake        = applied.brake,
        source       = source,
    )

    if path_valid and index == 1308:
        crashed = True

    # Low-pass filter on heading error.
    _h_err_filt = H_ERR_ALPHA * h_err + (1.0 - H_ERR_ALPHA) * _h_err_filt

    print(f"[{source:>22}] lane_shift: {lane_shift:+.2f} m, "
          f"h_err: {math.degrees(h_err):+5.1f} deg "
          f"(filt {math.degrees(_h_err_filt):+5.1f}), "
          f"idx: {index}, target_idx: {target_idx}")

    if pid_enabled:
        control = vehicle_pid.run_step_h(TARGET_SPEED, _h_err_filt)
        ego_vehicle.apply_control(control)
    else:
        controlObject.process_control()

    # --- Render --------------------------------------------------------------
    gameDisplay.blit(renderObject.surface, (0, 0))
    pygame.display.flip()

    # --- Events --------------------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

        controlObject.parse_control(event)

        if event.type == pygame.KEYUP:

            if event.key == LOG_TOGGLE_KEY:
                logger.toggle()

            # ENTER toggles PID on/off
            if event.key == pygame.K_RETURN:
                pid_enabled = not pid_enabled
                print(f"[PID] {'ENABLED' if pid_enabled else 'DISABLED (manual)'}")

            # TAB switches to a new ego vehicle
            if event.key == pygame.K_TAB:
                ego_vehicle.set_autopilot(True)
                ego_vehicle = random.choice(vehicles)

                if ego_vehicle.is_alive:
                    ego_vehicle.set_autopilot(False)

                    camera.stop()
                    camera.destroy()

                    bound_x = 0.5 + ego_vehicle.bounding_box.extent.x
                    bound_y = 0.5 + ego_vehicle.bounding_box.extent.y
                    bound_z = 0.5 + ego_vehicle.bounding_box.extent.z
                    camera_init_trans = carla.Transform(
                        carla.Location(x=+0.8 * bound_x,
                                       y=+0.0 * bound_y,
                                       z=1.3 * bound_z)
                    )
                    camera = world.spawn_actor(camera_bp, camera_init_trans,
                                               attach_to=ego_vehicle)
                    camera.listen(lambda image: pygame_callback(image, renderObject))

                    controlObject = ControlObject(ego_vehicle)
                    vehicle_pid.set_vehicle(ego_vehicle)

                    init_idx = find_initial_index(ego_vehicle)
                    if init_idx < 0:
                        print("[PATH] New vehicle's direction doesn't match "
                              "the recorded path — vision only.")
                        path_valid = False
                        index = 0
                    else:
                        path_valid = True
                        index = init_idx
                        print(f"[PATH] Snapped to waypoint {index}")

                    centerline_world = None
                    _h_err_filt = 0.0
                    was_in_junction = False  # let next tick re-evaluate

                    gameDisplay.fill((0, 0, 0))
                    gameDisplay.blit(renderObject.surface, (0, 0))
                    pygame.display.flip()

# =============================================================================
# Cleanup
# =============================================================================
camera.stop()
pygame.quit()
logger.save()