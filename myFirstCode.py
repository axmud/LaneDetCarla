import carla
import cv2
import random
import time
import numpy as np
from helper import load_model, detect_lanes
import subprocess


models = {
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
latest_image = {"data": None}
weather_options = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    # carla.WeatherParameters.MidRainyNoon,
    # carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    # carla.WeatherParameters.MidRainSunset,
    # carla.WeatherParameters.HardRainSunset,
    # carla.WeatherParameters.SoftRainSunset,
]
tm_port = 8000
sprawn_vehicles = []
sprawn_vehicles_num = 50
IMG_HEIGHT = 720
IMG_WIDTH = 1280
relative_transform = carla.Transform(
    carla.Location(x=1.5, z=1.7), carla.Rotation(pitch=0)
)
maps_list = [
    "Town01",
    # "Town02",
    # "Town03",
    "Town04",
    "Town05",
    "Town06",
    # "Town10HD_Opt"
]
map_index = 0
weather_index = 0
actions = [0, ""]
needy_actions = ["Right", "ChangeLaneLeft", "ChangeLaneRight", "Left"]


def main():
    global map_index, weather_index

    client = connect_carla_and_create_client()

    result = subprocess.run(["python PythonAPI/util/config.py  --map Town06"], shell=True, capture_output=True, text=True)
    print(result.stdout)  # output of command
    print(result.stderr)  # errors (if any)

    world = create_world(client)

    map_index = 1

    set_weather(world)

    tm = create_traffic_manager(client)

    create_sprawn_vehicles(world)

    ego_vehicle = create_ego_vehicle(world)

    vehicle_behavior(tm, ego_vehicle, 100, 50, 50, -20)

    camera_rgb = create_rgb_camera(world, ego_vehicle)

    model, cfg, transform = load_model(models["CLRNet_Tusimple"])

    try:
        warmup_world(world)

        listen_rgb_camera(camera_rgb)

        print("Starting visualization loop.")
        print("q / ESC  : quit")
        print("e        : change weather")
        print("m        : change map")
        while True:
            world.tick()
            lane_offset_changer(tm, ego_vehicle)
            frame = latest_image["data"]
            if frame is not None:
                lanes, frame = detect_lanes(frame, model, cfg, transform)
                cv2.imshow(
                    "Lane detection (autopilot, q/ESC to quit, w weather, m map)", frame
                )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("e"):
                set_weather(world)
            elif key == ord("m"):
                world, camera_rgb, ego_vehicle = change_map(camera_rgb)
    except Exception as e:
        print("Error in Main try-except block: ", e)
    finally:
        destroy_camera(camera_rgb)
        destroy_vehicles()
        cv2.destroyAllWindows()
        print("Clean shutdown complete.")


def connect_carla_and_create_client():
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    return client


def create_world(client):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS sim
    world.apply_settings(settings)
    return world


def set_weather(world):
    """
    Randomly choose one of your preferred weather presets.
    """
    global weather_options, weather_index
    weather = random.randint(0, len(weather_options) - 1)

    world.set_weather(weather_options[weather])
    print("Weather set to {}".format(weather))
    weather_index += 1
    if weather_index == len(weather_options):
        weather_index = 0


def create_traffic_manager(client):
    global tm_port
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.5)
    tm.global_percentage_speed_difference(0)
    return tm


def create_sprawn_vehicles(world):
    global sprawn_vehicles, tm_port, sprawn_vehicles_num
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("*vehicle*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for _ in range(sprawn_vehicles_num):
        vehicle = world.try_spawn_actor(
            random.choice(vehicle_bp), random.choice(spawn_points)
        )
        if vehicle != None:
            sprawn_vehicles.append(vehicle)

    for vehicle in sprawn_vehicles:
        vehicle.set_autopilot(True, tm_port)
    print(f"{len(sprawn_vehicles)} vehicles spawned.")


def create_ego_vehicle(world, autopilot=True):
    global sprawn_vehicles, tm_port
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    while True:
        try:
            vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle:
                sprawn_vehicles.append(vehicle)
                print("Ego vehicle created")
                break
        except Exception as e:
            print(
                f"There is an error while creating ego vehicle :{e}\nTrying again...",
                flush=True,
            )
    if autopilot:
        vehicle.set_autopilot(True, tm_port)
    return vehicle


def create_rgb_camera(world, vehicle):
    global IMG_HEIGHT, IMG_WIDTH
    blueprint_library = world.get_blueprint_library()
    cmr_rgb = blueprint_library.find("sensor.camera.rgb")
    cmr_rgb.set_attribute("role_name", "camera")
    cmr_rgb.set_attribute("image_size_x", str(IMG_WIDTH))
    cmr_rgb.set_attribute("image_size_y", str(IMG_HEIGHT))
    cmr_rgb.set_attribute("fov", "90")
    camera_rgb = world.spawn_actor(cmr_rgb, relative_transform, attach_to=vehicle)
    return camera_rgb


def destroy_camera(camera):
    try:
        # stop stream & detach callback
        camera.stop()
        camera.listen(lambda image: None)  # clear callback
        time.sleep(0.1)  # small delay to let C++ side flush
    except Exception as e:
        print("Error stopping camera:", e)
    try:
        camera.destroy()
    except Exception as e:
        print("Error destroying camera:", e)
    print("Camera is destroyed")


def destroy_vehicles():
    global sprawn_vehicles
    try:
        for vehicle in list(sprawn_vehicles):  # iterate over a copy
            try:
                vehicle.destroy()
            except Exception as e:
                print("Error destroying vehicle:", e)
        sprawn_vehicles.clear()
    except Exception as e:
        print("error in destroy_vehicles: ", e)
    print("Vehicles are destroyed")


def warmup_world(world):
    for _ in range(5):
        world.tick()
        time.sleep(0.01)


def listen_rgb_camera(camera_rgb):
    camera_rgb.listen(lambda image: show_camera_image(image))


def change_map(camera_rgb):
    """
    Destroy current actors, load next map, re-create everything.
    Returns: new_world, new_camera_rgb, new_ego_vehicle
    """
    global maps_list, map_index
    destroy_camera(camera_rgb)
    destroy_vehicles()

    new_client = connect_carla_and_create_client()

    print(f"Loading new world: {maps_list[map_index]}")
    new_world = load_new_world(new_client, maps_list[map_index])

    set_weather(new_world)

    tm = create_traffic_manager(new_client)

    create_sprawn_vehicles(new_world)

    new_ego_vehicle = create_ego_vehicle(new_world)

    vehicle_behavior(tm, new_ego_vehicle, 100, 50, 50, -20)

    new_camera_rgb = create_rgb_camera(new_world, new_ego_vehicle)

    listen_rgb_camera(new_camera_rgb)

    print(f"Map: {maps_list[map_index]}")

    map_index = map_index + 1
    if map_index == len(maps_list):
        map_index = 0

    return new_world, new_camera_rgb, new_ego_vehicle


def load_new_world(client, town_name):
    new_world = client.load_world(town_name)
    settings = new_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    new_world.apply_settings(settings)
    return new_world


def show_camera_image(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3].copy()
    latest_image["data"] = bgr


def vehicle_behavior(tm, vh, light_ign, left_chan, right_chan, speed, leading=2):
    tm.ignore_lights_percentage(vh, light_ign)
    tm.distance_to_leading_vehicle(vh, leading)
    tm.random_left_lanechange_percentage(vh, left_chan)
    tm.random_right_lanechange_percentage(vh, right_chan)
    tm.vehicle_percentage_speed_difference(vh, speed)


def lane_offset_changer(tm, vh):
    a = tm.get_next_action(vh)
    if actions[0] == 0 and a[0] in needy_actions:
        actions[0] = 1
        actions[1] = a[0]
    elif a[0] == actions[1]:
        return
    if actions[0] == 1 and a[0] != actions[1]:
        a = (random.random() - 0.5) * 1.8
        tm.vehicle_lane_offset(vh, a)
        actions[0] = 0
        actions[1] = ""


if __name__ == "__main__":
    main()
