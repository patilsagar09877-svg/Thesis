
# LiDAR PCD Multi-Frame Viewer

A Python desktop GUI for loading a sequence of LiDAR `.pcd` files, choosing a
current frame, and accumulating `N` past and `N` future frames.

## Features

- Load one or many PCD files
- Natural filename ordering (`frame2.pcd` before `frame10.pcd`)
- Select the current frame
- Configure past and future frame counts independently
- Optional pose-based motion compensation
- Voxel downsampling after accumulation
- Height, original RGB, or current-vs-neighbor coloring
- Interactive rotate, pan, and zoom view
- Background loading to keep the interface responsive

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python lidar_pcd_viewer.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python lidar_pcd_viewer.py
```

## Pose CSV format

For moving LiDAR sensors, neighboring frames must be transformed into the
current frame coordinate system. The optional CSV uses one row per frame:

```text
filename,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33
frame_0012.pcd,1,0,0,1.2,0,1,0,0.0,0,0,1,0.1,0,0,0,1
```

Each matrix is interpreted as the transform from that LiDAR frame into a shared
world coordinate system. The program computes:

```text
T_neighbor_to_current = inverse(T_current_to_world) @ T_neighbor_to_world
```

The filename column is matched by basename, so it may contain either
`frame_0012.pcd` or a full path.

## Important

Without poses, multiple scans are simply overlaid. That is correct for a
stationary sensor, but a moving sensor will produce doubled or smeared geometry.
