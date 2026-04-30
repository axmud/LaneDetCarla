# Installation

## Required Preinstallers

To install this project correctly, we need to have **NVIDIA CUDA Toolkit 12.1** and **Microsoft Visual Studio Build Tools 2022 with CUDA toolkit (ver:12.1) [MSVC v143 - VS 2022 C++ x64/86 build tools (v14.39-17.9)]** should be pre-installed on windows PC.

## Environment Setup

```cmd
set PROJECT=LaneDetCarla

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.10.8/install.ps1 | iex"

git clone https://github.com/axmud/LaneDetCarla.git %PROJECT%

cd %PROJECT%

uv venv

.venv\Scripts\activate

uv pip install setuptools==66.1.1

uv pip install wheel==0.46.3

uv pip install albumentations==0.4.6 --no-build-isolation

uv sync --extra cu121

uv pip install --no-build-isolation -e .

uv pip install numpy==1.23.1

uv pip install hydra-core==1.3.2

```

for all release download:

```cmd
mkdir releases\Tusimple\SCNN
curl -L -o releases\Tusimple\SCNN\ResNet18_AAAI.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/scnn_model_best_tusimple.pth

mkdir releases\Tusimple\RESA
curl -L -o releases\Tusimple\RESA\ResNet18_AAAI.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/resa_model_best_tusimple.pth

mkdir releases\Tusimple\UFLD
curl -L -o releases\Tusimple\UFLD\ResNet18_ECCV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/ufld_model_best_tusimple.pth

mkdir releases\Tusimple\LaneATT
curl -L -o releases\Tusimple\LaneATT\ResNet34_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/laneatt_model_best_tusimple.pth

mkdir releases\Tusimple\ADNet
curl -L -o releases\Tusimple\ADNet\ResNet34_ICCV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_tusimple.pth

mkdir releases\Tusimple\SRLane
curl -L -o releases\Tusimple\SRLane\ResNet34_AAAI.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/srnet_r34_tusimple_model_best.pth

mkdir releases\Tusimple\BezierNet
curl -L -o releases\Tusimple\BezierNet\ResNet18_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/beizernet_model_best.pth

mkdir releases\Tusimple\GANet
curl -L -o releases\Tusimple\GANet\ResNet18_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/ganet_r18_tusimple_model_best.pth

mkdir releases\Tusimple\GSENet
curl -L -o releases\Tusimple\GSENet\ResNet18_AAAI.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/gsenet_r18_tusimple.pth

mkdir releases\CULane\CLRNet
curl -L -o releases\CULane\CLRNet\ResNet34_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_r50_culane_model_best.pth
curl -L -o releases\CULane\CLRNet\ResNet50_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_culane.pth
curl -L -o releases\CULane\CLRNet\ConvNexT-Tiny_CVPR.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_convnext_culane.pth

mkdir releases\CULane\CLRerNet
curl -L -o releases\CULane\CLRerNet\ResNet34_WACV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_model_best_culane.pth
curl -L -o releases\CULane\CLRerNet\ConvNexT-Tiny_WACV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_convnext_culane.pth

mkdir releases\CULane\ADNet
curl -L -o releases\CULane\ADNet\ResNet34_ICCV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_culane.pth

mkdir releases\VIL100\ADNet
curl -L -o releases\VIL100\ADNet\ResNet34_ICCV.pth https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_final_vil100.pth
```
