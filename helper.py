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


# ---------------------------
# Inference (MAIN FUNCTION)
# ---------------------------
def detect_lanes(img_path, model, cfg, transform, draw=True):
    data = preprocess(img_path, cfg, transform)

    with torch.no_grad():
        out = model(data)

    lanes = model.get_lanes(out)[0]
    if lanes is None or len(lanes) == 0:
        img = data["ori_img"].copy()
        return None, img
    if isinstance(lanes[0], Lane):
        lanes = [lane.to_array(cfg.param_config) for lane in lanes]
    else:
        lanes = [np.array(lane, dtype=np.float32) for lane in lanes]

    if draw:
        img = data["ori_img"].copy()
        for lane in lanes:
            for x, y in lane:
                if x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), 2)
        return lanes, img

    return lanes
