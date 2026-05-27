# Consolidated Knowledge Base - MOVi-E & YOLOE-26

This document consolidates and details the specifications of the Kubric MOVi-E dataset, the data loading pipelines, and the YOLOE-26 model architecture as currently implemented in the codebase.

---

## 1. Kubric MOVi-E Dataset & Streaming Pipeline

The project utilizes the **Kubric MOVi-E** dataset, which is designed for physical intuition learning, object discovery, and ego-motion vs. independent motion separation.

### 1.1 MOVi-E Dataset Characteristics
- **Dynamic Camera**: Features a linear camera translation along a semi-spherical shell at constant speed.
- **Dynamic Objects**: Combines 10–20 obstacle objects on the floor with 1–3 active objects that move independently.
- **Ego-motion Decoupling**: Solves the core challenge of separating camera perspective changes (ego-motion parallax) from independent object movement.

### 1.2 Dual-Buffer Asynchronous Data Pipeline
To maximize GPU throughput, the codebase employs an asynchronous pipeline:
1. **CPU Asynchronous Buffer (`AsyncDataBuffer`)**:
   - Runs in a background daemon thread to fetch video clips via TFDS (`movi_e/256x256`).
   - Converts raw data to PyTorch tensors and pushes them to a `deque` circular buffer (capacity 64).
   - Samples batches randomly to increase training diversity.
2. **GPU Concurrent Prefetcher (`CUDAPrefetcher`)**:
   - Uses a dedicated `torch.cuda.Stream` to concurrently move batches from CPU to GPU.
   - Performs resizing, padding, and normalization on GPU, allowing the main training loop to proceed without blocking.

### 1.3 Core Physical Quantities & Preprocessing
All batch variables are standardized via `process_batch_on_gpu` to target dimensions (256x256):

#### Video Frames (`video`)
- **Data Type**: `uint8` RGB sequence.
- **Preprocessing**: Scaled to `float32` in the range `[0.0, 1.0]`, and resized using bilinear interpolation.

#### Absolute Depth (`depth`)
- **Data Type**: `uint16` encoded format.
- **Decoding Formula**:
  $$\text{depth\_m} = \frac{\text{depth\_encoded}}{65535.0} \times (\text{depth\_range}[1] - \text{depth\_range}[0]) + \text{depth\_range}[0]$$
- **Processing**: The sky/infinite regions where raw depth equals 0 are extracted as a sky mask, and their absolute depth is set to `100.0` meters. Raw values are clipped to the range `[0.01, 100.0]`. The network predicts logarithmic depth `log_depth` mapping into `[-4.6, 4.6]`.

#### Dense Forward Flow (`forward_flow`)
- **Data Type**: `uint16` compressed pixel offsets.
- **Decoding Formula**:
  $$\text{flow\_px} = \frac{\text{flow\_encoded}}{65535.0} \times (\text{flow\_range}[1] - \text{flow\_range}[0]) + \text{flow\_range}[0]$$
- **GPU Normalization**: Rescaled relative to the image coordinates as `flow_raw * 2.0 / target_size` and clipped to `[-1.5, 1.5]` to filter transient rendering spikes.

#### Camera & Ego Pose (`cam_pos`, `cam_quat`)
- **Absolute Pose**: `cam_pos` represents absolute $(X, Y, Z)$ coordinates, and `cam_quat` represents absolute orientation quaternions.
- **Quaternion Orientation**: Parsed in $(w, x, y, z)$ order to represent correct viewing directions.
- **Relative Motion (Ego-Pose)**: The training loss calculates the relative translation ($\Delta T$) and relative rotation matrix ($\Delta R$) between adjacent frames. $\Delta R$ is flattened into a 6D continuous representation, forming a 9D relative pose vector.
- **Camera Intrinsics**: Extracted from metadata or computed as $f_x = f_y = \frac{35.0}{32.0} \times W$ (yielding $280.0$ at $W=256$) with the optical center at $(W/2, H/2)$ for inverse warping and photometric warp losses.

#### Segmentation & Instance Dynamics (`segmentation`, `is_dynamic`)
- **Instance Masks**: Provides unique integer IDs for individual objects. GPU preprocessing extracts dense bounding boxes (`bboxes_dense`) from these masks to supervise detection heads.
- **Instance Dynamics**: Maps background/static rigid instances (False) to Class 0, and moving/dynamic instances (True) to Class 1, decoupling ego-motion视差 from true dynamic physical displacement.

---

## 2. Local YOLOE-26 Architecture & Weight Loading

The YOLOE-26 architecture implemented in `models/tao_core.py` and `models/custom_heads.py` is a specialized, lightweight version adapted for physical feature prediction and temporal continuity.

### 2.1 Local `YOLOESegment26` Model Structure
The local implementation comprises:
- **YOLO-style Backbone & FPN/PAN**: Captures multi-scale features across P3, P4, and P5 stages.
- **Spatiotemporal Processing**: Features are passed through `SpatioTemporalMambaBlock` modules to incorporate sequential temporal context.
- **Dual Outputs**: Provides both dense (multi-anchor) and one-to-one (anchor-free) outputs for classification, localization, and mask coefficients.
- **LRPCLayer**: Handles multi-scale detection and prompt-free gating:
  - `vocab`: Outlines a 4585-class semantic classification space (preserving zero-shot ability).
  - `pf`: Generates objectness/prompt-free gates.
  - `loc`: Predicts bounding box locations.
- **Proto26**: Fuses multi-scale features to produce mask prototypes and semantic segmentation maps.

*Note: The local implementation does not include official modules like `RepRTA`, `SAVPE`, external text encoders, or custom two-class parameter matrices.*

### 2.2 Partial Weight Loading Logic
The model initializes using the official pre-trained checkpoint `yoloe-26s-seg-pf.pt`:
1. Custom `Conv` modules (consisting of `conv + bn`) are dynamically replaced with bare `nn.Conv2d` layers (with `bias=True` and `bn` set to identity) to match the pre-trained weight layout.
2. The checkpoint state dict maps official keys (e.g., `model.*` or `model.model.*`) to the local `segmenter.model.*` path.
3. Keys are loaded selectively: only keys with matching names and identical tensor shapes are loaded.
4. **Successful Load Metric**: Successfully loads **243 out of 298 keys** (81.5%). Divergent structures like `RepRTA`, `SAVPE`, or specific deep C3k/C3k2 layers are safely skipped.
