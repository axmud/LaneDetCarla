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

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True        # Enables synchronous mode
settings.no_rendering_mode = True       # We render with PyGame, disable sim rendering
settings.fixed_delta_seconds = 0.05     # 20 Hz
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
# Data collectors (unchanged except for the frame-number fix)
# =============================================================================
def carla_data_collector(vehicle: carla.Vehicle):
    # NOTE: carla.Timestamp.frame references the class, not a value.
    # Use the snapshot's frame instead.
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
# Lane shift (cross-track error) calculator
# =============================================================================
index = 0
FUTURE_HORIZON = 10

def lane_shift_calculator(vehicle: carla.Vehicle):
    """
    Signed perpendicular distance from the car to the recorded path.
    CARLA is left-handed (+Y is right):
        lane_shift > 0  -> car is to the RIGHT of the path
        lane_shift < 0  -> car is to the LEFT  of the path
    """
    global index

    loc = vehicle.get_transform().location
    car_x, car_y = loc.x, loc.y

    # Vectorised nearest-waypoint search over the look-ahead window
    end = min(index + FUTURE_HORIZON + 1, len(wp_x_arr))
    dx = wp_x_arr[index:end] - car_x
    dy = wp_y_arr[index:end] - car_y
    sq_dist = dx * dx + dy * dy            # compare squared distances, skip sqrt
    offset = int(np.argmin(sq_dist))
    nearest_index = index + offset
    index = nearest_index

    # CARLA forward vectors are already unit length -> no need to divide by ||v||
    vx = vec_x_arr[nearest_index]
    vy = vec_y_arr[nearest_index]
    lane_shift = dx[offset] * vy - dy[offset] * vx

    nearest_dist = float(np.sqrt(sq_dist[offset]))
    return lane_shift, nearest_dist


# =============================================================================
# PID controllers (inspired by CARLA's agents/navigation/controller.py and
# the structure of local_planner.py)
# =============================================================================
class PIDLateralController:
    """
    Discrete PID for steering, driven by cross-track error (lane_shift).
    Output is clipped to [-1, 1] (CARLA's steering range).
    """

    def __init__(self, K_P=0.5, K_I=0.01, K_D=0.1, dt=0.05):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)  # rolling window for I and D terms

    def run_step(self, lane_shift):
        # Sign flip: lane_shift > 0 means the car is to the RIGHT of the path,
        # so we need NEGATIVE steering (turn left). Define error accordingly.
        error = -lane_shift

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


class PIDLongitudinalController:
    """
    Discrete PID for throttle/brake, driven by speed error (km/h).
    Output is an 'acceleration command' in [-1, 1]; positive -> throttle,
    negative -> brake.
    """

    def __init__(self, K_P=0.3, K_I=0.05, K_D=0.0, dt=0.05):
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
    Combines a lateral and a longitudinal PID into a single carla.VehicleControl.
    Equivalent in spirit to agents.navigation.controller.VehiclePIDController.
    """

    def __init__(self, vehicle,
                 args_lateral=None,
                 args_longitudinal=None,
                 max_throttle=0.75, max_brake=0.3, max_steering=0.8):

        if args_lateral is None:
            args_lateral = {'K_P': 0.2680, 'K_I': 0.1340, 'K_D': 0.1786, 'dt': 0.05}
        if args_longitudinal is None:
            args_longitudinal = {'K_P': 0.0953, 'K_I': 0.0524, 'K_D': 0.0000, 'dt': 0.05}

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

    def run_step(self, target_speed, lane_shift):
        # Current speed in km/h (CARLA velocity is m/s)
        v = self._vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        steering = self._lat_controller.run_step(lane_shift)

        control = carla.VehicleControl()

        # Longitudinal: map signed acceleration to throttle / brake
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self._max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self._max_brake)

        # Lateral: rate-limit steering to avoid jitter (same trick as CARLA's controller)
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
            # FIX: actually clamp the value (your original line discarded the result)
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


# =============================================================================
# Pick the ego vehicle and disable its autopilot so the PID can drive it
# =============================================================================
ego_vehicle = random.choice(vehicles)
print(f"Selected ego vehicle: {ego_vehicle.type_id} (id={ego_vehicle.id})")
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
# Build the PID controller for the ego vehicle
# -----------------------------------------------------------------------------
TARGET_SPEED = 30.0  # km/h

vehicle_pid = VehiclePIDController(
    ego_vehicle,
    args_lateral={'K_P': 0.2494, 'K_I': 0.1247, 'K_D': 0.1662, 'dt': 0.05},
    args_longitudinal={'K_P': 0.3143, 'K_I': 0.1571, 'K_D': 0.0000, 'dt': 0.05},
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

# =============================================================================
# Main loop
# =============================================================================
crashed = False
while not crashed:
    world.tick()

    # --- Lateral error + PID control -----------------------------------------
    lane_shift, nearest_dist = lane_shift_calculator(ego_vehicle)
    if index == 1308:
        crashed = True
    print(f"Lane shift: {lane_shift:+.2f} m, Nearest dist to waypoint: {nearest_dist:.2f} m", "Index:", index)
    if pid_enabled:
        control = vehicle_pid.run_step(TARGET_SPEED, lane_shift)
        ego_vehicle.apply_control(control)
        # Light debug line; uncomment if you want to tune gains from the console
        # print(f"lane_shift={lane_shift:+.2f}  steer={control.steer:+.2f}  "
        #       f"thr={control.throttle:.2f}  brk={control.brake:.2f}")
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
            # ENTER toggles PID on/off
            if event.key == pygame.K_RETURN:
                pid_enabled = not pid_enabled
                print(f"[PID] {'ENABLED' if pid_enabled else 'DISABLED (manual)'}")

            # TAB switches to a new ego vehicle
            if event.key == pygame.K_TAB:
                # Hand old ego back to TM
                ego_vehicle.set_autopilot(True)
                ego_vehicle = random.choice(vehicles)

                if ego_vehicle.is_alive:
                    ego_vehicle.set_autopilot(False)

                    # Replace camera
                    camera.stop()
                    camera.destroy()
                    camera = world.spawn_actor(camera_bp, camera_init_trans,
                                               attach_to=ego_vehicle)
                    camera.listen(lambda image: pygame_callback(image, renderObject))

                    # Rebind controllers to the new vehicle
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
