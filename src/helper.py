import torch
import cv2
import numpy as np
from copy import copy

from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.data.transform import Preprocess
from unlanedet.model.module.core.lane import Lane


# ---------------------------
# Load model once
# ---------------------------
def load_model(config, device="cuda"):
    config_path = config["Config"]
    ckpt_path = config["Weight"]
    cfg = LazyConfig.load(config_path)
    cfg = LazyConfig.apply_overrides(cfg, [])

    model = instantiate(cfg.model)
    model.to(device)
    model = create_ddp_model(model)
    model.eval()

    Checkpointer(model).load(ckpt_path)

    transform = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))

    return model, cfg, transform


# ---------------------------
# Preprocess
# ---------------------------
def preprocess(img_input, cfg, transform):
    if isinstance(img_input, str):
        ori_img = cv2.imread(img_input)
        img_path = img_input
    else:
        ori_img = img_input
        img_path = "frame.jpg"

    img = ori_img[cfg.param_config.cut_height :, :, :].astype(np.float32)
    vis_img = copy(img)

    data = {"img": img, "lanes": []}
    data = transform(data)

    data["img"] = data["img"].unsqueeze(0)
    data.update({"img_path": img_path, "ori_img": ori_img, "vis_img": vis_img})

    return data

def get_centerline(lanes, image_width=1280, image_height=720,
                   bottom_frac=0.7, y_step=10, order='near_to_far'):
    """
    Lane-center polyline in pixel coordinates.

    Parameters
    ----------
    lanes : list[np.ndarray]
        Detector output; each (N, 2) array is [x_px, y_px] for one lane line.
    bottom_frac : float
        Only lanes reaching y >= image_height*bottom_frac are treated as
        ego-lane boundary candidates (filters out far/adjacent lanes).
    y_step : int
        Sampling step along y in pixels.
    order : {'near_to_far', 'far_to_near'}
        'near_to_far' => centerline[0] is the point closest to the car
        (largest y, bottom of the image). This is the convention most
        path-following / pure-pursuit / heading controllers expect.

    Returns
    -------
    np.ndarray, shape (M, 2), columns [x_px, y_px].
    Empty array if a valid left/right pair cannot be found.
    """
    cx = image_width / 2.0
    y_cut = image_height * bottom_frac

    near = [lane for lane in lanes if lane[:, 1].max() >= y_cut]
    if len(near) < 2:
        return np.empty((0, 2))

    bottom_x = np.array([lane[np.argmax(lane[:, 1]), 0] for lane in near])
    left  = np.where(bottom_x <  cx)[0]
    right = np.where(bottom_x >= cx)[0]
    if left.size == 0 or right.size == 0:
        return np.empty((0, 2))

    L = near[left[np.argmax(bottom_x[left])]]      # closest left boundary
    R = near[right[np.argmin(bottom_x[right])]]    # closest right boundary

    L = L[np.argsort(L[:, 1])]
    R = R[np.argsort(R[:, 1])]
    y_min = max(L[0, 1],  R[0, 1])
    y_max = min(L[-1, 1], R[-1, 1])
    if y_max <= y_min:
        return np.empty((0, 2))

    ys = np.arange(y_min, y_max + 1, y_step)
    xs = (np.interp(ys, L[:, 1], L[:, 0]) +
          np.interp(ys, R[:, 1], R[:, 0])) / 2.0

    centerline = np.column_stack([xs, ys])
    if order == 'near_to_far':
        centerline = centerline[::-1]      # descending y => nearest first
    return centerline
# ---------------------------
# Inference (MAIN FUNCTION)
# ---------------------------
def detect_lanes(image, model, cfg, transform, draw=True, mask_top_frac=0.5):
    """
    image : np.ndarray (H, W, 3) BGR uint8, OR str path.
    Returns (centerline, img) where img is None if draw=False.
    """
    # Resolve input to a BGR ndarray
    if isinstance(image, str):
        original = cv2.imread(image)
        if original is None:
            raise FileNotFoundError(image)
    elif isinstance(image, np.ndarray):
        original = image
    else:
        raise TypeError(f"Expected ndarray or path, got {type(image)}")

    # Black out the upper portion before inference
    if mask_top_frac > 0:
        masked = original.copy()
        cutoff = int(original.shape[0] * mask_top_frac)
        masked[:cutoff] = 0
    else:
        masked = original

    # preprocess() must accept a BGR ndarray here — see note below if yours
    # currently only accepts paths.
    data = preprocess(masked, cfg, transform)
    data["ori_img"] = original   # draw on the unmasked frame

    with torch.no_grad():
        out = model(data)

    lanes = model.get_lanes(out)[0]

    if lanes is None or len(lanes) == 0:
        return None, (original.copy() if draw else None)

    if isinstance(lanes[0], Lane):
        lanes = [lane.to_array(cfg.param_config) for lane in lanes]
    else:
        lanes = [np.array(lane, dtype=np.float32) for lane in lanes]

    img_h, img_w = original.shape[:2]
    centerline = get_centerline(lanes, image_width=img_w, image_height=img_h)
    if centerline.size == 0:
        centerline = None

    if not draw:
        return centerline, None

    img = original.copy()

    # Visualize the mask boundary (optional, helpful for debugging)
    if mask_top_frac > 0:
        cutoff = int(img.shape[0] * mask_top_frac)
        cv2.line(img, (0, cutoff), (img.shape[1], cutoff),
                 (128, 128, 128), 1, cv2.LINE_AA)

    # Lane boundary points (blue)
    for lane in lanes:
        for x, y in lane:
            if x > 0 and y > 0:
                cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), 2)

    # Centerline (green polyline + samples) and nearest point (red)
    if centerline is not None:
        pts = centerline.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        for x, y in centerline:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        x0, y0 = centerline[0]
        cv2.circle(img, (int(x0), int(y0)), 6, (0, 0, 255), -1)

    return centerline, img
