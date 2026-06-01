# Forza Designer 6 - PyTorch-version (Nvidia only)

Based on Forza Designer 6: https://github.com/tokyubevoxelverse/ForzaDesigner6   
Forza Designer 6 (FD6) is an image-to-vector utility designed to reconstruct raster images into layered, stylized vector shapes compatible with Forza Horizon's vinyl group system.

This branch replaces the CPU/PyOpenCL engine with a **strictly CUDA-accelerated PyTorch engine** paired with an **attention-guided search heuristic**. 
By executing shape rasterization, canvas blending, and mathematical scoring entirely on GPU tensors, the engine minimizes system RAM to GPU VRAM transfer overhead.

---

## Key Features

### 1. CUDA-Native Shape Generation & Scoring
* **GPU-Bound Math:** Rasterization, optimal color calculations, and parallel candidate scoring are processed natively as PyTorch tensors.
* **Minimized Latency:** Evaluates whole candidate batches directly on CUDA stream memory, significantly reducing CPU-GPU sync barriers.

### 2. Computer Vision Attention & Saliency Mapping
* **Feature Detection:** Utilizes OpenCV image analysis (LAB color space, CLAHE contrast enhancement, Sobel edge gradients, Laplacian, and Gaussian-blurred saliency) to construct a detailed density map of target visual features.
* **Hole-Aware Segmentation:** Analyzes contour hierarchies to distinguish foreground figures from background elements and negative space, ensuring precise shape placement.

### 3. Gradient-Aligned Heuristics
* **Smart Placement:** Positions candidate shapes probabilistically based on target attention density rather than uniform random placement.
* **Auto-Orientation:** Automatically rotates and scales candidate shapes to align directly with the local edge orientation and complexity of the target image coordinates.

### 4. Post-Generation Relaxation ("Wiggle & Prune")
* **Sequential Optimization:** Features an on-demand forward relaxation sweep that optimizes committed shapes in completed vinyl json files. 
* **Layer Caching:** Implements Canvas and Occlusion caches to track layer visibility, allowing the engine to safely wiggle shape parameters to minimize error and prune redundant shapes.

---

## Architectural Comparison

| Architectural Dimension | Original Developer's Version | This PyTorch Branch |
| :--- | :--- | :--- |
| **Backend Framework** | PyOpenCL (OpenCL C Kernel strings executed on host-allocated GPU buffers) | **PyTorch (CUDA-native)** using GPU tensors and optimized PyTorch mathematical operators |
| **Hardware Compatibility** | Cross-vendor (NVIDIA, AMD, Intel GPU support through graphics drivers) | **NVIDIA-specific** (requires CUDA runtime support) |
| **Concurrency Model** | Multi-process pool (`ProcessPoolExecutor`) using shared-memory buffers (`SharedMemory`) | Single-process batched execution offloading workloads directly to CUDA streams |
| **Placement Heuristic** | Uniform random coordinate generation (non-target-aware) | **Attention-guided placement** prioritizing high-contrast, high-gradient, and high-saliency zones |
| **Orientation & Scale** | Uniform random distribution, refined with local hill-climbing mutations | **Gradient alignment** (shapes rotate along local Sobel edge angles; sizes scale with priority density) |
| **Post-Processing** | Absent (shapes committed linearly, with no post-generation adjustments) | **"Wiggle and Prune" sweep** using caching layers to dynamically relax parameters and remove redundant elements |

---

## Technical Dependencies

To run this branch, ensure the following dependencies are installed:

# 1. Install torch (PyTorch) 
Use the pip command generator available on the following page, CUDA must be selected under compute platform: https://pytorch.org/get-started/locally/

# 2. Install the remaining dependencies from PyPI
**pip install opencv-python numpy Pillow PySide6**

* **`opencv-python` (`cv2`)** (For contrast enhancement, morphological operations, and contour hierarchy parsing)
* **`numpy`**
* **`Pillow` (`PIL`)**
* **`PySide6`** (For GUI, multi-threading, and event-loop signals)

**How to run:**
- Use the provided **run.bat**  
OR
- Open CMD in the parent directory where the fd6 folder is (or use cd to navigate to the parent folder of fd6) and run: **python -m fd6** 

---

## Hardware Compatibility & Requirements

* **GPU:** Strictly requires an NVIDIA GPU supporting CUDA. The application performs a startup validation check and will raise a runtime error if no compatible CUDA-enabled GPU is detected.
* **VRAM:** Memory usage scales dynamically with your configuration's target canvas resolution, shape count limit, and search batch size. Checkpointing and occlusion cache data structures are optimized to offload to system memory where appropriate.

**Disclaimer**
**Use entirely at your own risk. FD6 modifies the memory of a running Forza Horizon process to populate vinyl-group shapes. It does not patch the game executable, install drivers, modify save files, or attempt to bypass any anti-cheat or DRM system. However, memory modification of a live game process may be interpreted by Microsoft, Xbox Live, or the game's publisher (Turn 10 / Playground Games) as a violation of the Microsoft Services Agreement, the Xbox Community Standards, or the relevant Forza title's terms of use. Doing so may result in temporary suspension or permanent ban of your Xbox / Microsoft account, loss of access to purchased games, online services, achievements, and any content created with FD6.**

**The authors and contributors of Forza Designer 6 and of the PyTorch-version branch don't accept any responsibility or liability whatsoever for any consequences arising from the use of this software. By downloading, building, installing, or running FD6 you acknowledge these risks and accept them in full. This tool is provided as-is, with no warranties of any kind. Not affiliated with, endorsed by, or sponsored by Turn 10 Studios, Playground Games, Microsoft, Xbox, or any official Forza brand.**

**No support is provided for PyTorch-version whatsoever, please refrain from opening any issues.
Don't direct any issues arising from this branch to the original Forza Designer 6 repository**
