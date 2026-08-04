from __future__ import annotations

import csv
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QVector3D, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


@dataclass
class PCDData:
    points: np.ndarray
    fields: Dict[str, np.ndarray]
    colors: Optional[np.ndarray]
    total_points: int
    valid_points: int
    removed_points: int


def _parse_pcd_header(file_path: Path) -> Tuple[dict, int]:
    header = {}
    header_size = 0

    with open(file_path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected end of file before DATA line.")

            header_size += len(line)
            decoded = line.decode("ascii", errors="ignore").strip()

            if not decoded or decoded.startswith("#"):
                continue

            parts = decoded.split(maxsplit=1)
            key = parts[0].upper()
            value = parts[1] if len(parts) > 1 else ""
            header[key] = value

            if key == "DATA":
                break

    return header, header_size


def _numpy_dtype_from_pcd(type_code: str, size: int):
    type_code = type_code.upper()

    mapping = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
        ("I", 8): np.int64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("U", 8): np.uint64,
    }

    key = (type_code, size)
    if key not in mapping:
        raise ValueError(f"Unsupported PCD field type: TYPE={type_code}, SIZE={size}")

    return mapping[key]


def read_pcd_fields(file_path: Path) -> PCDData:
    """
    Reads ASCII or binary PCD files and preserves scalar fields such as
    intensity, reflectivity, ring, ambient and range.
    """
    header, header_size = _parse_pcd_header(file_path)

    fields = header.get("FIELDS", header.get("FIELD", "")).split()
    sizes = [int(x) for x in header.get("SIZE", "").split()]
    types = header.get("TYPE", "").split()
    counts = [int(x) for x in header.get("COUNT", "").split()] if "COUNT" in header else [1] * len(fields)

    if not fields or not sizes or not types:
        raise ValueError("PCD header is missing FIELDS, SIZE or TYPE.")

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD header field metadata lengths do not match.")

    point_count = int(header.get("POINTS", "0") or 0)
    if point_count <= 0:
        width = int(header.get("WIDTH", "0") or 0)
        height = int(header.get("HEIGHT", "1") or 1)
        point_count = width * height

    data_mode = header.get("DATA", "").strip().lower()

    # NumPy structured arrays require unique field names. Some LiDAR PCD
    # files repeat padding names such as PAD/PAD1. Rename only the internal
    # dtype names while preserving a mapping to the original header names.
    dtype_fields = []
    internal_names = []
    name_counts: Dict[str, int] = {}

    for original_name, size, type_code, count in zip(
        fields, sizes, types, counts
    ):
        base_name = original_name
        occurrence = name_counts.get(base_name.lower(), 0) + 1
        name_counts[base_name.lower()] = occurrence

        internal_name = (
            base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        )
        internal_names.append(internal_name)

        np_dtype = _numpy_dtype_from_pcd(type_code, size)
        if count == 1:
            dtype_fields.append((internal_name, np_dtype))
        else:
            dtype_fields.append((internal_name, np_dtype, (count,)))

    structured_dtype = np.dtype(dtype_fields)

    if data_mode == "binary":
        with open(file_path, "rb") as handle:
            handle.seek(header_size)
            raw = handle.read(point_count * structured_dtype.itemsize)

        expected = point_count * structured_dtype.itemsize
        if len(raw) < expected:
            raise ValueError(
                f"Binary PCD data is shorter than expected: "
                f"{len(raw)} bytes found, {expected} expected."
            )

        structured = np.frombuffer(
            raw[:expected],
            dtype=structured_dtype,
            count=point_count,
        ).copy()

    elif data_mode == "ascii":
        array = np.loadtxt(file_path, skiprows=0, comments="#")

        # np.loadtxt cannot directly skip an unknown header length, so read manually.
        rows = []
        with open(file_path, "r", encoding="ascii", errors="ignore") as handle:
            data_started = False
            for line in handle:
                stripped = line.strip()
                if not data_started:
                    if stripped.upper().startswith("DATA"):
                        data_started = True
                    continue
                if stripped:
                    rows.append(stripped)

        if not rows:
            raise ValueError("ASCII PCD contains no point rows.")

        flat = np.loadtxt(rows)
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)

        total_scalar_columns = sum(counts)
        if flat.shape[1] != total_scalar_columns:
            raise ValueError(
                f"ASCII PCD has {flat.shape[1]} columns, "
                f"but header describes {total_scalar_columns}."
            )

        structured = np.empty(flat.shape[0], dtype=structured_dtype)
        column = 0
        for internal_name, count in zip(internal_names, counts):
            if count == 1:
                structured[internal_name] = flat[:, column]
            else:
                structured[internal_name] = flat[
                    :, column:column + count
                ]
            column += count

        point_count = len(structured)

    elif data_mode == "binary_compressed":
        raise ValueError(
            "binary_compressed PCD is not supported by the custom field reader. "
            "Convert it to binary or ASCII PCD first."
        )
    else:
        raise ValueError(f"Unsupported PCD DATA mode: {data_mode}")

    # Find x, y and z using the original header names, then access their
    # corresponding unique internal dtype names.
    original_to_internal: Dict[str, str] = {}
    for original_name, internal_name in zip(fields, internal_names):
        original_to_internal.setdefault(
            original_name.lower(),
            internal_name,
        )

    required = {"x", "y", "z"}
    if not required.issubset(original_to_internal):
        raise ValueError("PCD must contain x, y and z fields.")

    points = np.column_stack(
        (
            structured[original_to_internal["x"]].astype(np.float64),
            structured[original_to_internal["y"]].astype(np.float64),
            structured[original_to_internal["z"]].astype(np.float64),
        )
    )

    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask].astype(np.float32)

    scalar_fields: Dict[str, np.ndarray] = {}
    for original_name, internal_name in zip(fields, internal_names):
        data = np.asarray(structured[internal_name])

        # Keep duplicate fields accessible with unique names, but avoid
        # overwriting the first occurrence.
        output_name = original_name
        if output_name in scalar_fields:
            duplicate_number = 2
            while f"{output_name}_{duplicate_number}" in scalar_fields:
                duplicate_number += 1
            output_name = f"{output_name}_{duplicate_number}"

        if data.ndim == 1:
            scalar_fields[output_name] = data[valid_mask]
        elif data.ndim == 2 and data.shape[1] == 1:
            scalar_fields[output_name] = data[:, 0][valid_mask]

    colors = None
    if "rgb" in scalar_fields:
        rgb_raw = scalar_fields["rgb"]
        if np.issubdtype(rgb_raw.dtype, np.floating):
            packed = rgb_raw.astype(np.float32).view(np.uint32)
        else:
            packed = rgb_raw.astype(np.uint32)

        r = ((packed >> 16) & 255).astype(np.float32) / 255.0
        g = ((packed >> 8) & 255).astype(np.float32) / 255.0
        b = (packed & 255).astype(np.float32) / 255.0
        colors = np.column_stack((r, g, b))

    elif {"r", "g", "b"}.issubset(scalar_fields):
        colors = np.column_stack(
            (
                scalar_fields["r"].astype(np.float32),
                scalar_fields["g"].astype(np.float32),
                scalar_fields["b"].astype(np.float32),
            )
        )
        if colors.max(initial=0) > 1.0:
            colors /= 255.0

    return PCDData(
        points=valid_points,
        fields=scalar_fields,
        colors=colors,
        total_points=point_count,
        valid_points=len(valid_points),
        removed_points=point_count - len(valid_points),
    )


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)

    result = np.zeros_like(values, dtype=np.float32)
    if not finite.any():
        return result

    valid = values[finite]
    low = np.percentile(valid, 1)
    high = np.percentile(valid, 99)

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    if high <= low:
        result[finite] = 0.5
        return result

    result[finite] = np.clip((valid - low) / (high - low), 0.0, 1.0)
    return result


def color_map(values: np.ndarray) -> np.ndarray:
    t = normalize_values(values)
    rgba = np.ones((len(t), 4), dtype=np.float32)

    rgba[:, 0] = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    rgba[:, 1] = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    rgba[:, 2] = np.clip(1.25 - 1.5 * t, 0.0, 1.0)

    return rgba


class CloudLoader(QThread):
    loaded = Signal(object, object, object, object, object, str)
    failed = Signal(str)

    def __init__(
        self,
        files: List[Path],
        current_index: int,
        past: int,
        future: int,
        voxel_size: float,
        poses: Dict[str, np.ndarray],
        distance_min: float,
        distance_max: float,
    ):
        super().__init__()
        self.files = files
        self.current_index = current_index
        self.past = past
        self.future = future
        self.voxel_size = voxel_size
        self.poses = poses
        self.distance_min = distance_min
        self.distance_max = distance_max

    def run(self):
        try:
            start = max(0, self.current_index - self.past)
            end = min(len(self.files), self.current_index + self.future + 1)
            selected_files = self.files[start:end]
            current_file = self.files[self.current_index]

            # Each pose must be T_world_from_lidar:
            #     p_world = T_world_from_lidar @ p_lidar
            #
            # To express a neighboring scan in the current frame:
            #     p_current =
            #         inverse(T_world_from_current)
            #         @ T_world_from_neighbor
            #         @ p_neighbor
            ref_pose = self.poses.get(current_file.name)

            if len(selected_files) > 1 and ref_pose is None:
                raise ValueError(
                    f"No ego pose was found for the current frame "
                    f"'{current_file.name}'. Multi-frame accumulation requires "
                    f"a pose for the current frame."
                )

            world_to_ref = (
                np.linalg.inv(ref_pose)
                if ref_pose is not None
                else np.eye(4)
            )

            point_parts = []
            rgb_parts = []
            current_mask_parts = []
            frame_index_parts = []
            field_parts: Dict[str, List[np.ndarray]] = {}

            all_have_rgb = True
            total_raw = 0
            total_valid = 0
            total_removed = 0

            for global_index, file_path in enumerate(selected_files, start=start):
                data = read_pcd_fields(file_path)

                total_raw += data.total_points
                total_valid += data.valid_points
                total_removed += data.removed_points

                points = data.points.astype(np.float64)
                fields = data.fields
                rgb = data.colors

                pose = self.poses.get(file_path.name)

                if ref_pose is not None:
                    if pose is None:
                        raise ValueError(
                            f"No ego pose was found for neighboring frame "
                            f"'{file_path.name}'."
                        )

                    transform = world_to_ref @ pose
                    homogeneous = np.column_stack(
                        (points, np.ones(len(points)))
                    )
                    points = (transform @ homogeneous.T).T[:, :3]

                distance = np.linalg.norm(points, axis=1)
                keep = np.isfinite(points).all(axis=1)
                keep &= distance >= self.distance_min
                if self.distance_max > 0:
                    keep &= distance <= self.distance_max

                points = points[keep].astype(np.float32)
                if len(points) == 0:
                    continue

                point_parts.append(points)
                current_mask_parts.append(
                    np.full(len(points), file_path == current_file, dtype=bool)
                )
                frame_index_parts.append(
                    np.full(len(points), global_index, dtype=np.int32)
                )

                for name, values in fields.items():
                    if len(values) == len(keep):
                        field_parts.setdefault(name, []).append(np.asarray(values)[keep])

                if rgb is not None and len(rgb) == len(keep):
                    rgb_parts.append(rgb[keep].astype(np.float32))
                else:
                    all_have_rgb = False

            if not point_parts:
                raise ValueError("No valid points remain after filtering.")

            points = np.concatenate(point_parts, axis=0)
            current_mask = np.concatenate(current_mask_parts, axis=0)
            frame_indices = np.concatenate(frame_index_parts, axis=0)

            rgb = (
                np.concatenate(rgb_parts, axis=0)
                if all_have_rgb and len(rgb_parts) == len(point_parts)
                else None
            )

            merged_fields = {}
            for name, parts in field_parts.items():
                if len(parts) == len(point_parts):
                    merged_fields[name] = np.concatenate(parts, axis=0)

            if self.voxel_size > 0:
                finite = np.isfinite(points).all(axis=1)
                points = points[finite]
                current_mask = current_mask[finite]
                frame_indices = frame_indices[finite]

                if rgb is not None:
                    rgb = rgb[finite]

                for name in list(merged_fields):
                    merged_fields[name] = merged_fields[name][finite]

                scaled = points / self.voxel_size
                int_limit = np.iinfo(np.int64).max // 4
                safe = np.abs(scaled).max(axis=1) < int_limit

                points = points[safe]
                current_mask = current_mask[safe]
                frame_indices = frame_indices[safe]
                scaled = scaled[safe]

                if rgb is not None:
                    rgb = rgb[safe]

                for name in list(merged_fields):
                    merged_fields[name] = merged_fields[name][safe]

                voxels = np.floor(scaled).astype(np.int64)
                _, keep_indices = np.unique(voxels, axis=0, return_index=True)
                keep_indices.sort()

                points = points[keep_indices]
                current_mask = current_mask[keep_indices]
                frame_indices = frame_indices[keep_indices]

                if rgb is not None:
                    rgb = rgb[keep_indices]

                for name in list(merged_fields):
                    merged_fields[name] = merged_fields[name][keep_indices]

            warning = ""

            stats = {
                "raw_points": total_raw,
                "valid_points": total_valid,
                "removed_points": total_removed,
                "displayed_points": len(points),
                "selected_frames": len(selected_files),
            }

            self.loaded.emit(
                points,
                merged_fields,
                rgb,
                current_mask,
                frame_indices,
                warning,
            )
            self.stats = stats

        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LiDAR PCD Multi-Frame Viewer")
        self.resize(1500, 900)

        self.files: List[Path] = []
        self.poses: Dict[str, np.ndarray] = {}

        self.points: Optional[np.ndarray] = None
        self.fields: Dict[str, np.ndarray] = {}
        self.rgb: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.frame_indices: Optional[np.ndarray] = None

        self.scatter: Optional[gl.GLScatterPlotItem] = None
        self.worker: Optional[CloudLoader] = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)

        # Debounce live rebuilding so moving the frame slider does not start
        # many overlapping loading jobs.
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(180)
        self.refresh_timer.timeout.connect(self.refresh_cloud)

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()

        self.setStatusBar(QStatusBar())

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(340)
        controls.setMaximumWidth(450)
        control_layout = QVBoxLayout(controls)

        file_group = QGroupBox("PCD sequence")
        file_form = QFormLayout(file_group)

        load_single_button = QPushButton("Load one PCD…")
        load_single_button.clicked.connect(self.load_single_file)
        file_form.addRow(load_single_button)

        load_multiple_button = QPushButton("Load multiple PCDs…")
        load_multiple_button.clicked.connect(self.load_multiple_files)
        file_form.addRow(load_multiple_button)

        load_folder_button = QPushButton("Load PCD folder…")
        load_folder_button.clicked.connect(self.load_folder)
        file_form.addRow(load_folder_button)

        self.dataset_label = QLabel("No files loaded")
        self.dataset_label.setWordWrap(True)
        file_form.addRow("Dataset:", self.dataset_label)

        self.current_spin = QSpinBox()
        self.current_spin.setRange(1, 1)
        self.current_spin.valueChanged.connect(self.on_frame_control_changed)
        file_form.addRow("Current frame:", self.current_spin)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.valueChanged.connect(self.current_spin.setValue)
        self.current_spin.valueChanged.connect(self.frame_slider.setValue)
        file_form.addRow("Frame slider:", self.frame_slider)

        self.filename_label = QLabel("—")
        self.filename_label.setWordWrap(True)
        file_form.addRow("Filename:", self.filename_label)

        self.range_label = QLabel("—")
        self.range_label.setWordWrap(True)
        file_form.addRow("Accumulation:", self.range_label)

        self.past_spin = QSpinBox()
        self.past_spin.setRange(0, 500)
        self.past_spin.setValue(2)
        self.past_spin.valueChanged.connect(self.on_frame_control_changed)
        file_form.addRow("Past frames N:", self.past_spin)

        self.future_spin = QSpinBox()
        self.future_spin.setRange(0, 500)
        self.future_spin.setValue(2)
        self.future_spin.valueChanged.connect(self.on_frame_control_changed)
        file_form.addRow("Future frames N:", self.future_spin)

        playback_row = QWidget()
        playback_layout = QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_button)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.2, 30.0)
        self.fps_spin.setValue(5.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setSuffix(" FPS")
        self.fps_spin.valueChanged.connect(self.update_playback_interval)
        playback_layout.addWidget(self.fps_spin)

        file_form.addRow("Playback:", playback_row)
        control_layout.addWidget(file_group)

        process_group = QGroupBox("Processing")
        process_form = QFormLayout(process_group)

        pose_button = QPushButton("Load pose CSV…")
        pose_button.clicked.connect(self.load_pose_csv)
        process_form.addRow(pose_button)

        self.pose_label = QLabel("No poses loaded")
        self.pose_label.setWordWrap(True)
        process_form.addRow("Alignment:", self.pose_label)

        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(0.0, 10.0)
        self.voxel_spin.setDecimals(3)
        self.voxel_spin.setSingleStep(0.01)
        self.voxel_spin.setValue(0.05)
        self.voxel_spin.setSuffix(" m")
        process_form.addRow("Voxel size:", self.voxel_spin)

        self.min_distance_spin = QDoubleSpinBox()
        self.min_distance_spin.setRange(0.0, 10000.0)
        self.min_distance_spin.setValue(0.0)
        self.min_distance_spin.setSuffix(" m")
        process_form.addRow("Minimum range:", self.min_distance_spin)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.0, 10000.0)
        self.max_distance_spin.setValue(0.0)
        self.max_distance_spin.setSuffix(" m")
        self.max_distance_spin.setSpecialValueText("No maximum")
        process_form.addRow("Maximum range:", self.max_distance_spin)

        self.auto_refresh = QCheckBox(
            "Live accumulation when frame settings change"
        )
        self.auto_refresh.setChecked(True)
        process_form.addRow(self.auto_refresh)

        self.build_button = QPushButton("Build accumulated cloud")
        self.build_button.clicked.connect(self.refresh_cloud)
        process_form.addRow(self.build_button)

        export_button = QPushButton("Export displayed cloud…")
        export_button.clicked.connect(self.export_cloud)
        process_form.addRow(export_button)

        control_layout.addWidget(process_group)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)

        self.color_combo = QComboBox()
        self.color_combo.addItems(
            [
                "Height",
                "Intensity",
                "Reflectivity",
                "Ring",
                "Ambient",
                "Range",
                "Current vs neighbors",
                "Original RGB",
                "Frame index",
            ]
        )
        self.color_combo.currentTextChanged.connect(self.update_scatter)
        display_form.addRow("Color mode:", self.color_combo)

        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(2)
        self.point_size_slider.valueChanged.connect(self.update_scatter)
        display_form.addRow("Point size:", self.point_size_slider)

        self.grid_checkbox = QCheckBox("Show ground grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self.toggle_grid)
        display_form.addRow(self.grid_checkbox)

        reset_button = QPushButton("Reset camera")
        reset_button.clicked.connect(self.reset_camera)
        display_form.addRow(reset_button)

        control_layout.addWidget(display_group)

        self.stats_label = QLabel("Load a PCD file or folder.")
        self.stats_label.setWordWrap(True)
        control_layout.addWidget(self.stats_label)

        control_layout.addStretch(1)
        splitter.addWidget(controls)

        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((20, 22, 26))
        self.view.opts["distance"] = 30
        self.view.opts["elevation"] = 25
        self.view.opts["azimuth"] = 45

        self.grid = gl.GLGridItem()
        self.grid.setSize(100, 100)
        self.grid.setSpacing(1, 1)
        self.view.addItem(self.grid)

        axis = gl.GLAxisItem()
        axis.setSize(3, 3, 3)
        self.view.addItem(axis)

        viewer_layout.addWidget(self.view)
        splitter.addWidget(viewer_container)
        splitter.setStretchFactor(1, 1)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        single_action = QAction("Load one PCD…", self)
        single_action.triggered.connect(self.load_single_file)
        file_menu.addAction(single_action)

        multiple_action = QAction("Load multiple PCDs…", self)
        multiple_action.triggered.connect(self.load_multiple_files)
        file_menu.addAction(multiple_action)

        folder_action = QAction("Load PCD folder…", self)
        folder_action.triggered.connect(self.load_folder)
        file_menu.addAction(folder_action)

        file_menu.addSeparator()

        export_action = QAction("Export displayed cloud…", self)
        export_action.triggered.connect(self.export_cloud)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_shortcuts(self):
        previous_action = QAction(self)
        previous_action.setShortcut(QKeySequence(Qt.Key_Left))
        previous_action.triggered.connect(self.previous_frame)
        self.addAction(previous_action)

        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_action.triggered.connect(self.next_frame)
        self.addAction(next_action)

        refresh_action = QAction(self)
        refresh_action.setShortcut(QKeySequence(Qt.Key_Space))
        refresh_action.triggered.connect(self.refresh_cloud)
        self.addAction(refresh_action)

    def set_files(self, files: List[Path]):
        self.files = sorted(files, key=natural_key)

        if not self.files:
            QMessageBox.warning(self, "No files", "No PCD files were found.")
            return

        self.current_spin.blockSignals(True)
        self.frame_slider.blockSignals(True)

        self.current_spin.setRange(1, len(self.files))
        self.frame_slider.setRange(1, len(self.files))

        self.current_spin.setValue(1)
        self.frame_slider.setValue(1)

        self.current_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)

        self.dataset_label.setText(
            f"{len(self.files)} file(s)\n{self.files[0].parent}"
        )

        self.update_filename()
        self.update_selected_range_label()
        self.refresh_cloud()

    def load_single_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select PCD file",
            "",
            "Point clouds (*.pcd);;All files (*)",
        )
        if filename:
            self.set_files([Path(filename)])

    def load_multiple_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PCD files",
            "",
            "Point clouds (*.pcd);;All files (*)",
        )
        if filenames:
            self.set_files([Path(name) for name in filenames])

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing PCD files",
        )
        if not folder:
            return

        files = list(Path(folder).glob("*.pcd"))
        self.set_files(files)

    def load_pose_csv(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not filename:
            return

        try:
            poses: Dict[str, np.ndarray] = {}

            with open(filename, "r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)

                for row_number, row in enumerate(reader, start=1):
                    if not row or row[0].strip().startswith("#"):
                        continue

                    first = row[0].strip()
                    if row_number == 1 and first.lower() in {
                        "filename",
                        "file",
                        "frame",
                    }:
                        continue

                    if len(row) != 17:
                        raise ValueError(
                            f"Row {row_number}: expected filename and 16 matrix "
                            f"values, but found {len(row)} columns."
                        )

                    matrix = np.asarray(
                        [float(value) for value in row[1:]],
                        dtype=np.float64,
                    ).reshape(4, 4)

                    poses[Path(first).name] = matrix

            if not poses:
                raise ValueError("No valid poses were found.")

            self.poses = poses
            self.pose_label.setText(f"{len(poses)} pose(s) loaded")

            if self.files:
                self.refresh_cloud()

        except Exception as exc:
            QMessageBox.critical(self, "Pose CSV error", str(exc))

    def on_frame_control_changed(self):
        self.update_filename()
        self.update_selected_range_label()

        if self.auto_refresh.isChecked() and self.files:
            self.refresh_timer.start()

    def update_filename(self):
        if not self.files:
            self.filename_label.setText("—")
            return

        index = self.current_spin.value() - 1
        self.filename_label.setText(self.files[index].name)

    def update_selected_range_label(self):
        if not self.files:
            self.range_label.setText("—")
            return

        current = self.current_spin.value() - 1
        start = max(0, current - self.past_spin.value())
        end = min(
            len(self.files) - 1,
            current + self.future_spin.value(),
        )

        self.range_label.setText(
            f"Frames {start + 1}–{end + 1} → frame {current + 1}"
        )

    def refresh_cloud(self):
        if not self.files:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.build_button.setEnabled(False)
        self.statusBar().showMessage("Loading and accumulating PCD frames…")

        self.worker = CloudLoader(
            files=self.files,
            current_index=self.current_spin.value() - 1,
            past=self.past_spin.value(),
            future=self.future_spin.value(),
            voxel_size=self.voxel_spin.value(),
            poses=self.poses,
            distance_min=self.min_distance_spin.value(),
            distance_max=self.max_distance_spin.value(),
        )

        self.worker.loaded.connect(self.on_cloud_loaded)
        self.worker.failed.connect(self.on_cloud_failed)
        self.worker.finished.connect(lambda: self.build_button.setEnabled(True))
        self.worker.start()

    def on_cloud_loaded(
        self,
        points,
        fields,
        rgb,
        current_mask,
        frame_indices,
        warning,
    ):
        self.points = points
        self.fields = fields
        self.rgb = rgb
        self.current_mask = current_mask
        self.frame_indices = frame_indices

        self.update_scatter()
        self.reset_camera()

        current_name = self.files[self.current_spin.value() - 1].name
        current_index = self.current_spin.value() - 1
        start_index = max(0, current_index - self.past_spin.value())
        end_index = min(
            len(self.files) - 1,
            current_index + self.future_spin.value(),
        )
        actual_frames = end_index - start_index + 1

        info = (
            f"Current: {current_name}\n"
            f"Aligned frames: {start_index + 1}–{end_index + 1} "
            f"into frame {current_index + 1}\n"
            f"Frames used: {actual_frames}\n"
            f"Displayed points: {len(points):,}\n"
            f"Available scalar fields: "
            f"{', '.join(sorted(fields.keys())) or 'none'}"
        )

        if warning:
            info += f"\nWarning: {warning}"
            self.stats_label.setStyleSheet("color: #e0a030;")
        else:
            self.stats_label.setStyleSheet("")

        self.stats_label.setText(info)
        self.statusBar().showMessage(
            f"Displaying {len(points):,} points from frame "
            f"{self.current_spin.value()}."
        )

    def on_cloud_failed(self, message: str):
        self.statusBar().showMessage("Failed to load PCD data.")
        QMessageBox.critical(self, "PCD loading error", message)

    def get_colors(self) -> np.ndarray:
        if self.points is None:
            return np.empty((0, 4), dtype=np.float32)

        mode = self.color_combo.currentText()

        if mode == "Current vs neighbors" and self.current_mask is not None:
            rgba = np.ones((len(self.points), 4), dtype=np.float32)
            rgba[:, :3] = np.array([0.25, 0.65, 1.0], dtype=np.float32)
            rgba[self.current_mask, :3] = np.array(
                [1.0, 0.55, 0.10],
                dtype=np.float32,
            )
            return rgba

        if mode == "Original RGB" and self.rgb is not None:
            rgba = np.ones((len(self.points), 4), dtype=np.float32)
            rgba[:, :3] = np.clip(self.rgb, 0.0, 1.0)
            return rgba

        if mode == "Height":
            return color_map(self.points[:, 2])

        if mode == "Frame index" and self.frame_indices is not None:
            return color_map(self.frame_indices)

        field_map = {
            "Intensity": ["intensity"],
            "Reflectivity": ["reflectivity"],
            "Ring": ["ring"],
            "Ambient": ["ambient"],
            "Range": ["range"],
        }

        for candidate in field_map.get(mode, []):
            if candidate in self.fields:
                return color_map(self.fields[candidate])

        return color_map(self.points[:, 2])

    def update_scatter(self):
        if self.points is None:
            return

        if self.scatter is not None:
            self.view.removeItem(self.scatter)

        self.scatter = gl.GLScatterPlotItem(
            pos=self.points,
            color=self.get_colors(),
            size=float(self.point_size_slider.value()),
            pxMode=True,
        )

        self.view.addItem(self.scatter)

    def reset_camera(self):
        if self.points is None or len(self.points) == 0:
            self.view.setCameraPosition(distance=30, elevation=25, azimuth=45)
            return

        finite = np.isfinite(self.points).all(axis=1)
        points = self.points[finite]

        if len(points) == 0:
            return

        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2.0

        extent = float(np.linalg.norm(maximum - minimum))
        distance = max(5.0, extent * 1.3)

        self.view.opts["center"] = QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

        self.view.setCameraPosition(
            distance=distance,
            elevation=25,
            azimuth=45,
        )

    def toggle_grid(self, visible: bool):
        self.grid.setVisible(visible)

    def export_cloud(self):
        if self.points is None or len(self.points) == 0:
            QMessageBox.warning(
                self,
                "Nothing to export",
                "Load and display a point cloud first.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export displayed point cloud",
            "accumulated_cloud.pcd",
            "PCD files (*.pcd);;PLY files (*.ply)",
        )

        if not filename:
            return

        try:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(
                self.points.astype(np.float64)
            )

            rgba = self.get_colors()
            if len(rgba) == len(self.points):
                cloud.colors = o3d.utility.Vector3dVector(
                    rgba[:, :3].astype(np.float64)
                )

            success = o3d.io.write_point_cloud(
                filename,
                cloud,
                write_ascii=False,
                compressed=False,
            )

            if not success:
                raise RuntimeError("Open3D could not write the output file.")

            self.statusBar().showMessage(f"Saved: {filename}")

        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.update_playback_interval()
            self.play_timer.start()
            self.play_button.setText("Pause")

    def update_playback_interval(self):
        interval_ms = max(1, int(1000.0 / self.fps_spin.value()))
        self.play_timer.setInterval(interval_ms)

    def advance_frame(self):
        if not self.files:
            return

        next_value = self.current_spin.value() + 1
        if next_value > len(self.files):
            next_value = 1

        self.current_spin.setValue(next_value)

    def previous_frame(self):
        if not self.files:
            return

        self.current_spin.setValue(
            max(1, self.current_spin.value() - 1)
        )

    def next_frame(self):
        if not self.files:
            return

        self.current_spin.setValue(
            min(len(self.files), self.current_spin.value() + 1)
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()