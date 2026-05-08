import carla
import random
import pygame
import math
import numpy as np
import pandas as pd
from collections import deque

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town06')
delta_seconds = 0.05  # 20 Hz
# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True        # Enables synchronous mode
settings.no_rendering_mode = True       # We render with PyGame, disable sim rendering
settings.fixed_delta_seconds = delta_seconds     # 20 Hz
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Repeatable behaviour
traffic_manager.set_random_device_seed(0)
random.seed(0)

# Retrieve the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Select some models from the blueprint library
models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln',
          'prius', 'nissan', 'crown', 'impala']
blueprints = []
for vehicle in world.get_blueprint_library().filter('*vehicle*'):
    if any(model in vehicle.id for model in models):
        blueprints.append(vehicle)

# Spawn traffic
max_vehicles = 50
max_vehicles = min([max_vehicles, len(spawn_points)])
vehicles = []
for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
    temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
    if temp is not None:
        vehicles.append(temp)

# Give control to the TM
for vehicle in vehicles:
    vehicle.set_autopilot(True)
    traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))


# =============================================================================
# Rendering
# =============================================================================
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


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

# Look-ahead policy (CARLA-style: distance grows with speed)
LOOKAHEAD_MIN = 1.0   # m
LOOKAHEAD_K   = 0.5   # s   ->  Ld = LOOKAHEAD_MIN + LOOKAHEAD_K * v_mps


def lane_shift_calculator(vehicle: carla.Vehicle):
    """
    Advances the global `index` to the nearest waypoint within the look-ahead
    window and returns:
        lane_shift   - signed perpendicular distance to the nearest waypoint (m)
                       Kept as a diagnostic of tracking quality.
        nearest_dist - Euclidean distance to that waypoint (m)
    CARLA is left-handed (+Y is right): lane_shift > 0 -> car is to the RIGHT.
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
    pointing from the vehicle to the target. Sign convention matches CARLA's
    PIDLateralController: positive => target is to the right => steer right.
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


# =============================================================================
# PID controllers
#   Lateral plant (heading-error formulation):  G_lat(s) = K_lat / s
#       K_lat = v * delta_max / L_wb   (single integrator, scales with speed)
#   Longitudinal plant:  G_lon(s) = K / (tau*s + 1) * exp(-L*s)
# =============================================================================
class PIDLateralController:
    """
    Discrete PID on heading error (radians). CARLA convention:
        heading_err > 0  ->  target to the right  ->  positive steer (right).
    Output clipped to CARLA's steering range [-1, 1].
    """

    def __init__(self, K_P=1.95, K_I=0.05, K_D=0.2, dt=delta_seconds):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, heading_err):
        # No sign flip: CARLA's heading_err sign already matches the steer sign.
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
    """
    Discrete PID on speed error (km/h).
    Output is signed acceleration in [-1, 1]; positive -> throttle, negative -> brake.
    """

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
    Mirrors CARLA's interface: run_step(target_speed, target_x, target_y).
    Heading error is computed internally from the target waypoint.
    """

    def __init__(self, vehicle,
                 args_lateral=None,
                 args_longitudinal=None,
                 max_throttle=0.75, max_brake=0.3, max_steering=0.8):

        # CARLA-style defaults for the heading-error plant
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

    def run_step(self, target_speed, target_x, target_y):
        # Current speed in km/h (CARLA velocity is m/s)
        v = self._vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        h_err = heading_error(self._vehicle, target_x, target_y)

        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        steering     = self._lat_controller.run_step(h_err)

        control = carla.VehicleControl()

        # Longitudinal: signed acceleration -> throttle / brake
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self._max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self._max_brake)

        # Lateral: rate-limit steering to avoid jitter (same trick as CARLA)
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
# Pick the ego vehicle and disable its autopilot so the PID can drive it
# =============================================================================
ego_vehicle = random.choice(vehicles)
ego_vehicle.set_autopilot(False)

# Camera rig
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
camera = world.spawn_actor(camera_bp, camera_init_trans,
                           attach_to=ego_vehicle, attachment_type=Attachment.Rigid)

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

renderObject = RenderObject(image_w, image_h)
controlObject = ControlObject(ego_vehicle)
camera.listen(lambda image: pygame_callback(image, renderObject))

# -----------------------------------------------------------------------------
# Build the PID controller for the ego vehicle (CARLA-style heading-error gains)
# -----------------------------------------------------------------------------
TARGET_SPEED = 30.0  # km/h
# =============================================================================
# Trajectory logging config
# =============================================================================
LOG_TRAJECTORY     = True                   # master on/off switch
LOG_OUTPUT_PATH    = "src/heading_error_pid/ego_trajectory_heading.csv"
LOG_TOGGLE_KEY     = pygame.K_l             # press 'L' in the PyGame window


vehicle_pid = VehiclePIDController(
    ego_vehicle,
    args_lateral={'K_P': 1.2314, 'K_I': 1.2314, 'K_D': 0.0000, 'dt': delta_seconds},
    args_longitudinal={'K_P': 0.3143, 'K_I': 0.1571, 'K_D': 0.0000, 'dt': delta_seconds},
    max_throttle=0.75,
    max_brake=0.3,
    max_steering=0.8,
)

pid_enabled = True  # toggle with ENTER

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

logger = TrajectoryLogger(enabled=LOG_TRAJECTORY, output_path=LOG_OUTPUT_PATH)

# =============================================================================
# Main loop
# =============================================================================
crashed = False
while not crashed:
    world.tick()

    # --- Path geometry: cross-track (diagnostic) + look-ahead target ---------
    lane_shift, nearest_dist = lane_shift_calculator(ego_vehicle)
    target_x, target_y, target_idx = get_lookahead_target(ego_vehicle)

    # --- Log ego trajectory --------------------------------------------------
    loc = ego_vehicle.get_transform().location
    rot = ego_vehicle.get_transform().rotation
    vel = ego_vehicle.get_velocity()
    applied = ego_vehicle.get_control()
    snap = world.get_snapshot()

    logger.log(
        frame        = snap.frame,
        time_s       = snap.timestamp.elapsed_seconds,
        x            = loc.x,
        y            = loc.y,
        z            = loc.z,
        yaw_deg      = rot.yaw,
        speed_kmh    = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2),
        lane_shift   = lane_shift,
        nearest_dist = nearest_dist,
        ref_index    = index,
        ref_x        = float(wp_x_arr[index]),
        ref_y        = float(wp_y_arr[index]),
        steer        = applied.steer,
        throttle     = applied.throttle,
        brake        = applied.brake,
    )
    h_err = heading_error(ego_vehicle, target_x, target_y)

    if index == 1308:
        crashed = True

    print(f"Lane shift: {lane_shift:+.2f} m, "
          f"Heading err: {math.degrees(h_err):+5.1f} deg, "
          f"Index: {index}, Target idx: {target_idx}")

    if pid_enabled:
        # CARLA-style call: pass the look-ahead target; PID computes heading error inside
        control = vehicle_pid.run_step(TARGET_SPEED, target_x, target_y)
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
                    camera = world.spawn_actor(camera_bp, camera_init_trans,
                                               attach_to=ego_vehicle)
                    camera.listen(lambda image: pygame_callback(image, renderObject))

                    controlObject = ControlObject(ego_vehicle)
                    vehicle_pid.set_vehicle(ego_vehicle)
                    index = 0  # reset path-follow cursor

                    gameDisplay.fill((0, 0, 0))
                    gameDisplay.blit(renderObject.surface, (0, 0))
                    pygame.display.flip()

# =============================================================================
# Cleanup
# =============================================================================
camera.stop()
pygame.quit()
logger.save()
