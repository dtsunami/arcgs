# arcgs — Gaussian Splatting for Intel Arc

Open-source 3D Gaussian Splatting pipeline for **Intel Arc GPUs** (and NVIDIA/CPU fallback).  
Uses [gsplat](https://github.com/isl-org/gsplat) (Intel fork) for GPU-accelerated training via PyTorch XPU.

```
extract → sfm → train → export → view
ffmpeg     COLMAP  gsplat   .splat   WebGL
```

Tested on: Intel Arc Pro B70 (32 GB VRAM), Windows 11, Python 3.12,
PyTorch `2.11.0+xpu`, oneAPI `2025.3.2`.

---

## Prerequisites

Install these before anything else.

### 1. Visual Studio Build Tools 2022

Required to compile the gsplat SYCL kernels.

Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

During install, select the **Desktop development with C++** workload.  
Required components:
- MSVC v143 C++ build tools
- Windows 11 SDK
- CMake tools for Windows

### 2. Intel oneAPI Base Toolkit

Required for the SYCL compiler (`icx`) that builds the Arc GPU kernels.

Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Default install path: `C:\Program Files (x86)\Intel\oneAPI`

The oneAPI Unified Runtime version must match the version bundled with your PyTorch XPU build.
Check what version you have after installing PyTorch:

```powershell
pip show intel-cmplr-lib-ur
# Version: 2025.3.2   ← must match your PyTorch XPU's expectation
```

Verify the compiler is functional after install:

```cmd
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
icx --version
```

### 3. Git

Required to clone gsplat and initialize its submodules.

```powershell
winget install Git.Git
```

### 4. Python 3.10+

```powershell
winget install Python.Python.3.12
```

### 5. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

### 6. COLMAP (no-CUDA build)

Download the portable `colmap-x64-windows-nocuda` zip from:  
https://github.com/colmap/colmap/releases

Extract to e.g. `C:\bin\colmap-x64-windows-nocuda`.

---

## Installation

### Step 1 — Create a virtual environment

```powershell
cd Z:\splatt          # or wherever your arcgs checkout lives
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Install PyTorch XPU

The plain PyPI torch is CPU-only. Install from Intel's XPU index:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Verify XPU is detected:

```python
import torch
print(torch.__version__)                  # e.g. 2.11.0+xpu
print(torch.xpu.is_available())           # True
print(torch.xpu.get_device_name(0))       # Intel(R) Arc(TM) Pro B70 Graphics
```

If `is_available()` returns False, update your Arc GPU driver from intel.com —
the Level Zero runtime ships with the driver.

### Step 3 — Build gsplat with SYCL/XPU support

The standard `pip install gsplat` ships CUDA-only wheels — **it does not work on Arc.**  
The pre-built wheels at `--find-links https://isl-org.github.io/gsplat/whl/gsplat`
detect the XPU backend but do not include compiled SYCL kernels.  
You must build Intel's fork from source.

#### 3a. Clone and initialize submodules

```powershell
git clone https://github.com/isl-org/gsplat C:\bin\gsplat-xpu
cd C:\bin\gsplat-xpu
git submodule update --init --recursive
```

The `--recursive` flag is **required** — GLM (the math library) is a submodule.
The build fails with `glm/glm.hpp not found` if you skip this.

#### 3b. Open an oneAPI build environment

Open a **cmd** window (not PowerShell) and source the oneAPI environment:

```cmd
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Then from that cmd window, switch to PowerShell:

```cmd
powershell
```

#### 3c. Build and install

```powershell
$env:DISTUTILS_USE_SDK = "1"
cd C:\bin\gsplat-xpu
pip install --extra-index-url=https://download.pytorch.org/whl/xpu .
```

The build takes 5–15 minutes. When it finishes, verify:

```python
import gsplat
# Output: gsplat: Using SYCL XPU backend.
```

### Step 4 — Install arcgs

```powershell
cd Z:\splatt
pip install -e .
```

### Step 5 — Configure tool paths

```powershell
copy .env.example .env
```

Edit `.env`:

```env
ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
COLMAP_PATH=C:\bin\colmap-x64-windows-nocuda\bin
FFMPEG_PATH=C:\bin\ffmpeg\bin
FFPROBE_PATH=C:\bin\ffmpeg\bin
```

`ONEAPI_ROOT` lets arcgs source `setvars.bat` automatically on every run —
you never need to set up the oneAPI environment manually in your shell.

Verify everything is found:

```powershell
arcgs info
```

Expected output:
- gsplat: **ready**
- XPU backend: **available**, device name, VRAM reported
- ffmpeg / ffprobe / colmap: all **green**

---

## Usage

### Full pipeline (video → splat)

```powershell
arcgs run footage.mp4 --device xpu
arcgs run footage.mp4 --output ./my_scene --device xpu
```

### Skip completed stages

```powershell
arcgs run footage.mp4 --device xpu --skip-extract   # frames already extracted
arcgs run footage.mp4 --device xpu --skip-sfm       # SfM already done
```

### Resume interrupted training

```powershell
arcgs run footage.mp4 --device xpu --skip-extract --skip-sfm --resume
```

### Run stages individually

```powershell
arcgs extract footage.mp4 --output output/frames
arcgs sfm output/frames --output output/sfm
arcgs train output/sfm/undistorted --device xpu
arcgs export output/train/splat.ply --format splat
arcgs view output/export/splat.splat
```

### Override training settings

Create a TOML file:

```toml
# fast.toml — quick test at 7k steps
[train]
iterations = 7000
sh_degree = 1
```

```powershell
arcgs run footage.mp4 --device xpu --config fast.toml
```

---

## Training parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `iterations` | 30,000 | Full quality; 7,000 for quick tests |
| `sh_degree` | 3 | View-dependent colour (0–3); higher = more VRAM |
| `lambda_dssim` | 0.2 | SSIM loss weight (L1 gets 1 − λ) |
| `densify_from` | 500 | Start densification at this step |
| `densify_until` | 15,000 | Stop densification at this step |
| `densify_every` | 100 | Densification interval (steps) |
| `opacity_reset_every` | 3,000 | Reset low-opacity Gaussians |
| `save_every` | 5,000 | Checkpoint save interval (0 = end only) |

---

## Performance (Arc Pro B70, 32 GB VRAM)

- 21 training frames, 5,200 SfM points, 30,000 iterations
- GPU utilization: ~78%, ~70 °C
- Memory: 2–6 GB VRAM depending on scene size (well within 32 GB)

---

## Compute backends

| Backend | Flag | Requires |
|---------|------|----------|
| `xpu` | `--device xpu` | PyTorch XPU + gsplat built from source (this guide) |
| `cuda` | `--device cuda` | NVIDIA GPU + `pip install gsplat` |
| `cpu` | `--device cpu` | Nothing extra; ~10–20× slower than GPU |

Auto-detection priority: **XPU → CUDA → CPU**

---

## Troubleshooting

**`glm/glm.hpp not found` during gsplat build**  
You skipped `git submodule update --init --recursive`.  
Run it inside `C:\bin\gsplat-xpu` before building.

**`Unable to find compiled sycl kernels package`**  
The gsplat wheel from PyPI or `--find-links` does not include compiled SYCL kernels.  
Follow Step 3 to build from source.

**`'NoneType' object has no attribute 'CameraModelType'`**  
Same root cause as above — wrong gsplat wheel. Build from source.

**`torch.xpu.is_available()` returns False**  
Either the CPU-only PyTorch was installed (reinstall from the XPU index),  
or the Level Zero runtime isn't installed (update your Arc GPU driver).

**`torch has no xpu module`**  
PyPI torch does not include XPU. Reinstall:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

**`setvars.bat not found`**  
Set `ONEAPI_ROOT` in `.env` to your oneAPI install directory.

**COLMAP mapper produces no reconstruction**  
Try sequential matching for ordered video frames:
```toml
# sequential.toml
[sfm]
matcher = "sequential"
```
```powershell
arcgs run footage.mp4 --config sequential.toml
```
