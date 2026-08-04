
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass
class PCDData:
    points: np.ndarray


def parse_pcd_header(file_path: Path) -> Tuple[dict, int]:
    header = {}
    data_offset = 0

    with open(file_path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected end of PCD file before DATA line.")

            data_offset += len(line)
            text = line.decode("ascii", errors="ignore").strip()

            if not text or text.startswith("#"):
                continue

            parts = text.split(maxsplit=1)
            key = parts[0].upper()
            value = parts[1] if len(parts) > 1 else ""
            header[key] = value

            if key == "DATA":
                break

    return header, data_offset


def pcd_numpy_dtype(type_code: str, size: int):
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

    key = (type_code.upper(), size)
    if key not in mapping:
        raise ValueError(
            f"Unsupported PCD field type: TYPE={type_code}, SIZE={size}"
        )

    return mapping[key]


def read_pcd(file_path: Path) -> PCDData:
    header, data_offset = parse_pcd_header(file_path)

    fields = header.get("FIELDS", header.get("FIELD", "")).split()
    sizes = [int(value) for value in header.get("SIZE", "").split()]
    types = header.get("TYPE", "").split()

    counts = (
        [int(value) for value in header.get("COUNT", "").split()]
        if "COUNT" in header
        else [1] * len(fields)
    )

    if not fields:
        raise ValueError("PCD has no FIELDS entry.")

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(
            "PCD FIELDS, SIZE, TYPE and COUNT lengths do not match."
        )

    point_count = int(
        header.get(
            "POINTS",
            str(
                int(header.get("WIDTH", "0"))
                * int(header.get("HEIGHT", "1"))
            ),
        )
    )

    internal_names = []
    seen_names = {}
    dtype_description = []

    for field_name, field_size, field_type, field_count in zip(
        fields, sizes, types, counts
    ):
        normalized_name = field_name.lower()
        seen_names[normalized_name] = (
            seen_names.get(normalized_name, 0) + 1
        )

        internal_name = (
            field_name
            if seen_names[normalized_name] == 1
            else f"{field_name}_{seen_names[normalized_name]}"
        )
        internal_names.append(internal_name)

        numpy_type = pcd_numpy_dtype(field_type, field_size)

        if field_count == 1:
            dtype_description.append((internal_name, numpy_type))
        else:
            dtype_description.append(
                (internal_name, numpy_type, (field_count,))
            )

    structured_dtype = np.dtype(dtype_description)
    data_mode = header.get("DATA", "").strip().lower()

    if data_mode == "binary":
        expected_bytes = point_count * structured_dtype.itemsize

        with open(file_path, "rb") as handle:
            handle.seek(data_offset)
            raw_data = handle.read(expected_bytes)

        if len(raw_data) < expected_bytes:
            raise ValueError(
                "Binary PCD contains fewer bytes than expected."
            )

        cloud = np.frombuffer(
            raw_data[:expected_bytes],
            dtype=structured_dtype,
            count=point_count,
        ).copy()

    elif data_mode == "ascii":
        rows = []

        with open(
            file_path,
            "r",
            encoding="ascii",
            errors="ignore",
        ) as handle:
            data_started = False

            for line in handle:
                text = line.strip()

                if not data_started:
                    if text.upper().startswith("DATA"):
                        data_started = True
                    continue

                if text:
                    rows.append(text)

        if not rows:
            raise ValueError("ASCII PCD contains no point rows.")

        matrix = np.loadtxt(rows)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        cloud = np.empty(matrix.shape[0], dtype=structured_dtype)

        column_index = 0
        for internal_name, field_count in zip(
            internal_names, counts
        ):
            if field_count == 1:
                cloud[internal_name] = matrix[:, column_index]
            else:
                cloud[internal_name] = matrix[
                    :,
                    column_index : column_index + field_count,
                ]

            column_index += field_count

    else:
        raise ValueError(
            "Only ASCII and binary PCD files are supported. "
            "binary_compressed PCD is not supported."
        )

    first_name_for_field = {}

    for original_name, internal_name in zip(
        fields, internal_names
    ):
        first_name_for_field.setdefault(
            original_name.lower(),
            internal_name,
        )

    for required_field in ("x", "y", "z"):
        if required_field not in first_name_for_field:
            raise ValueError(
                "The PCD file must contain x, y and z fields."
            )

    points = np.column_stack(
        [
            cloud[first_name_for_field["x"]].astype(np.float64),
            cloud[first_name_for_field["y"]].astype(np.float64),
            cloud[first_name_for_field["z"]].astype(np.float64),
        ]
    )

    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]

    return PCDData(points=points)


def load_calibration(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as handle:
        content = json.load(handle)

    calibration_items = content.get("calibration", [])

    transform = None
    camera_matrix = None
    distortion = None

    for item in calibration_items:
        calibration_name = str(
            item.get("calibration", "")
        ).lower()

        calibration_type = str(
            item.get("calibration_type", "")
        ).lower()

        if calibration_name == "lidar_to_camera" or (
            calibration_type == "extrinsic" and "T" in item
        ):
            transform = np.asarray(
                item["T"],
                dtype=np.float64,
            )

        if calibration_name == "camera" or (
            calibration_type == "intrinsic"
        ):
            if "k" in item:
                camera_matrix = np.asarray(
                    item["k"],
                    dtype=np.float64,
                )

            if "D" in item:
                distortion = np.asarray(
                    item["D"],
                    dtype=np.float64,
                ).reshape(-1)

    if transform is None:
        raise ValueError(
            "No LiDAR-to-camera transformation matrix T was found."
        )

    if transform.shape == (3, 4):
        transform_4x4 = np.eye(4, dtype=np.float64)
        transform_4x4[:3, :] = transform
        transform = transform_4x4

    if transform.shape != (4, 4):
        raise ValueError(
            f"Transformation matrix T must be 3x4 or 4x4, "
            f"but its shape is {transform.shape}."
        )

    if camera_matrix is None or camera_matrix.shape != (3, 3):
        raise ValueError(
            "No valid 3x3 camera intrinsic matrix k was found."
        )

    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)

    return transform, camera_matrix, distortion


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)

    lower = float(np.percentile(depth, 2))
    upper = float(np.percentile(depth, 98))

    if upper <= lower:
        lower = float(np.min(depth))
        upper = float(np.max(depth))

    if upper <= lower:
        return np.full(depth.shape, 0.5, dtype=np.float64)

    return np.clip(
        (depth - lower) / (upper - lower),
        0.0,
        1.0,
    )


def find_pair_files(
    pair_folder: Path,
) -> Tuple[Path, Path]:
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_files = sorted(
        [
            file_path
            for file_path in pair_folder.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in image_extensions
        ]
    )

    pcd_files = sorted(pair_folder.glob("*.pcd"))

    if not image_files:
        raise FileNotFoundError(
            f"No camera image was found in {pair_folder.name}."
        )

    if not pcd_files:
        raise FileNotFoundError(
            f"No PCD file was found in {pair_folder.name}."
        )

    return image_files[0], pcd_files[0]


def project_lidar(
    image_path: Path,
    pcd_path: Path,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    minimum_depth: float,
    maximum_depth: float,
    point_radius: int,
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Could not read camera image: {image_path.name}"
        )

    cloud = read_pcd(pcd_path)
    lidar_points = cloud.points

    if len(lidar_points) == 0:
        raise ValueError("The PCD contains no valid XYZ points.")

    homogeneous_points = np.column_stack(
        [
            lidar_points,
            np.ones(len(lidar_points), dtype=np.float64),
        ]
    )

    camera_points = (
        transform @ homogeneous_points.T
    ).T[:, :3]

    depth = camera_points[:, 2]

    valid = np.isfinite(camera_points).all(axis=1)
    valid &= depth > minimum_depth
    valid &= depth < maximum_depth

    camera_points = camera_points[valid]
    depth = depth[valid]

    if len(camera_points) == 0:
        raise ValueError(
            "No points remain after camera-depth filtering."
        )

    image_points, _ = cv2.projectPoints(
        camera_points.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )

    image_points = image_points.reshape(-1, 2)

    image_height, image_width = image.shape[:2]

    inside_image = (
        np.isfinite(image_points).all(axis=1)
        & (image_points[:, 0] >= 0)
        & (image_points[:, 0] < image_width)
        & (image_points[:, 1] >= 0)
        & (image_points[:, 1] < image_height)
    )

    image_points = image_points[inside_image]
    depth = depth[inside_image]

    if len(image_points) == 0:
        raise ValueError(
            "No transformed LiDAR points project inside the image."
        )

    normalized = normalize_depth(depth)

    colors = cv2.applyColorMap(
        (255 * (1.0 - normalized)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)

    projected_image = image.copy()

    for (u, v), color in zip(image_points, colors):
        cv2.circle(
            projected_image,
            (int(round(u)), int(round(v))),
            point_radius,
            tuple(int(component) for component in color),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    statistics = {
        "original_lidar_points": int(len(lidar_points)),
        "projected_points": int(len(image_points)),
        "minimum_depth_m": float(np.min(depth)),
        "median_depth_m": float(np.median(depth)),
        "average_depth_m": float(np.mean(depth)),
        "maximum_depth_m": float(np.max(depth)),
    }

    return projected_image, statistics


class BatchWorker(QThread):
    progress = Signal(int, int)
    log_message = Signal(str)
    preview_ready = Signal(object)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        pair_folders,
        calibration_path,
        output_folder,
        minimum_depth,
        maximum_depth,
        point_radius,
    ):
        super().__init__()

        self.pair_folders = pair_folders
        self.calibration_path = calibration_path
        self.output_folder = output_folder
        self.minimum_depth = minimum_depth
        self.maximum_depth = maximum_depth
        self.point_radius = point_radius

    def run(self):
        try:
            (
                transform,
                camera_matrix,
                distortion,
            ) = load_calibration(self.calibration_path)

            self.output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary_path = (
                self.output_folder / "projection_summary.csv"
            )

            successful = 0
            failed = 0
            rows = []

            total = len(self.pair_folders)

            for index, pair_folder in enumerate(
                self.pair_folders,
                start=1,
            ):
                try:
                    image_path, pcd_path = find_pair_files(
                        pair_folder
                    )

                    projected_image, statistics = project_lidar(
                        image_path=image_path,
                        pcd_path=pcd_path,
                        transform=transform,
                        camera_matrix=camera_matrix,
                        distortion=distortion,
                        minimum_depth=self.minimum_depth,
                        maximum_depth=self.maximum_depth,
                        point_radius=self.point_radius,
                    )

                    output_name = (
                        f"{pair_folder.name}_projection.png"
                    )

                    output_path = (
                        self.output_folder / output_name
                    )

                    saved = cv2.imwrite(
                        str(output_path),
                        projected_image,
                    )

                    if not saved:
                        raise IOError(
                            f"Could not save {output_name}."
                        )

                    successful += 1

                    rows.append(
                        {
                            "pair": pair_folder.name,
                            "camera_file": image_path.name,
                            "pcd_file": pcd_path.name,
                            **statistics,
                            "status": "success",
                            "error": "",
                        }
                    )

                    self.log_message.emit(
                        f"[{index}/{total}] "
                        f"{pair_folder.name}: "
                        f"{statistics['projected_points']:,} "
                        f"points projected."
                    )

                    self.preview_ready.emit(
                        projected_image
                    )

                except Exception as error:
                    failed += 1

                    rows.append(
                        {
                            "pair": pair_folder.name,
                            "camera_file": "",
                            "pcd_file": "",
                            "original_lidar_points": "",
                            "projected_points": "",
                            "minimum_depth_m": "",
                            "median_depth_m": "",
                            "average_depth_m": "",
                            "maximum_depth_m": "",
                            "status": "failed",
                            "error": str(error),
                        }
                    )

                    self.log_message.emit(
                        f"[{index}/{total}] "
                        f"{pair_folder.name}: FAILED - {error}"
                    )

                self.progress.emit(index, total)

            fieldnames = [
                "pair",
                "camera_file",
                "pcd_file",
                "original_lidar_points",
                "projected_points",
                "minimum_depth_m",
                "median_depth_m",
                "average_depth_m",
                "maximum_depth_m",
                "status",
                "error",
            ]

            with open(
                summary_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()
                writer.writerows(rows)

            self.completed.emit(
                {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "summary_path": str(summary_path),
                    "output_folder": str(self.output_folder),
                }
            )

        except Exception as error:
            self.failed.emit(str(error))


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 550)
        self.setStyleSheet(
            "background-color: #202124; color: white;"
        )
        self.setText("Projection preview")
        self.image_bgr: Optional[np.ndarray] = None

    def set_cv_image(self, image_bgr: np.ndarray):
        self.image_bgr = image_bgr.copy()
        self.refresh()

    def refresh(self):
        if self.image_bgr is None:
            return

        rgb_image = cv2.cvtColor(
            self.image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        height, width, channels = rgb_image.shape

        qimage = QImage(
            rgb_image.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(qimage)

        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Camera-LiDAR Batch Projection Tool"
        )
        self.resize(1500, 900)

        self.matched_folder: Optional[Path] = None
        self.calibration_path: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.worker: Optional[BatchWorker] = None

        self.viewer = ImageViewer()

        self.matched_label = QLabel(
            "No matched-pairs folder selected"
        )
        self.matched_label.setWordWrap(True)

        self.calibration_label = QLabel(
            "No calibration file selected"
        )
        self.calibration_label.setWordWrap(True)

        self.output_label = QLabel(
            "No output folder selected"
        )
        self.output_label.setWordWrap(True)

        self.number_of_pairs = QSpinBox()
        self.number_of_pairs.setRange(0, 100000)
        self.number_of_pairs.setValue(50)
        self.number_of_pairs.setSpecialValueText(
            "All available pairs"
        )

        self.minimum_depth = QDoubleSpinBox()
        self.minimum_depth.setRange(0.01, 500.0)
        self.minimum_depth.setValue(0.1)
        self.minimum_depth.setSuffix(" m")

        self.maximum_depth = QDoubleSpinBox()
        self.maximum_depth.setRange(1.0, 1000.0)
        self.maximum_depth.setValue(100.0)
        self.maximum_depth.setSuffix(" m")

        self.point_radius = QSpinBox()
        self.point_radius.setRange(1, 8)
        self.point_radius.setValue(2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.status_label = QLabel(
            "Select the three required paths."
        )
        self.status_label.setWordWrap(True)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(230)

        self.start_button = QPushButton(
            "Start batch projection"
        )
        self.start_button.clicked.connect(
            self.start_processing
        )

        self._build_ui()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        control_widget = QWidget()
        control_widget.setMaximumWidth(470)
        control_layout = QVBoxLayout(control_widget)

        input_group = QGroupBox("Input and output")
        input_layout = QFormLayout(input_group)

        matched_button = QPushButton(
            "Select matched-pairs folder..."
        )
        matched_button.clicked.connect(
            self.select_matched_folder
        )

        calibration_button = QPushButton(
            "Select calibration.json..."
        )
        calibration_button.clicked.connect(
            self.select_calibration
        )

        output_button = QPushButton(
            "Select output folder..."
        )
        output_button.clicked.connect(
            self.select_output_folder
        )

        input_layout.addRow(matched_button)
        input_layout.addRow(
            "Matched folder:",
            self.matched_label,
        )

        input_layout.addRow(calibration_button)
        input_layout.addRow(
            "Calibration:",
            self.calibration_label,
        )

        input_layout.addRow(output_button)
        input_layout.addRow(
            "Output folder:",
            self.output_label,
        )

        control_layout.addWidget(input_group)

        settings_group = QGroupBox("Processing settings")
        settings_layout = QFormLayout(settings_group)

        settings_layout.addRow(
            "Number of pairs:",
            self.number_of_pairs,
        )

        settings_layout.addRow(
            "Minimum depth:",
            self.minimum_depth,
        )

        settings_layout.addRow(
            "Maximum depth:",
            self.maximum_depth,
        )

        settings_layout.addRow(
            "Point radius:",
            self.point_radius,
        )

        settings_layout.addRow(
            QLabel(
                "Enter 0 to process every available pair."
            )
        )

        control_layout.addWidget(settings_group)

        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)

        process_layout.addWidget(self.start_button)
        process_layout.addWidget(self.progress_bar)
        process_layout.addWidget(self.status_label)
        process_layout.addWidget(self.log_box)

        control_layout.addWidget(process_group)
        control_layout.addStretch(1)

        main_layout.addWidget(control_widget)
        main_layout.addWidget(self.viewer, stretch=1)

    def select_matched_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Select the folder containing pair_ folders",
        )

        if not folder_name:
            return

        selected_folder = Path(folder_name)

        pair_folders = sorted(
            [
                folder
                for folder in selected_folder.iterdir()
                if folder.is_dir()
                and folder.name.lower().startswith("pair_")
            ]
        )

        if not pair_folders:
            QMessageBox.warning(
                self,
                "No pair folders",
                "The selected folder contains no directories "
                "whose names begin with pair_.",
            )
            return

        self.matched_folder = selected_folder

        self.matched_label.setText(
            f"{selected_folder}\n"
            f"{len(pair_folders)} pair folders found"
        )

        self.number_of_pairs.setMaximum(
            max(len(pair_folders), 1)
        )

        if self.number_of_pairs.value() > len(pair_folders):
            self.number_of_pairs.setValue(
                len(pair_folders)
            )

    def select_calibration(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibration.json",
            "",
            "JSON files (*.json);;All files (*)",
        )

        if not file_name:
            return

        selected_file = Path(file_name)

        try:
            load_calibration(selected_file)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Invalid calibration",
                str(error),
            )
            return

        self.calibration_path = selected_file
        self.calibration_label.setText(
            str(selected_file)
        )

    def select_output_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
        )

        if not folder_name:
            return

        self.output_folder = Path(folder_name)
        self.output_label.setText(
            str(self.output_folder)
        )

    def start_processing(self):
        if self.matched_folder is None:
            QMessageBox.warning(
                self,
                "Missing matched folder",
                "Select the matched-pairs folder.",
            )
            return

        if self.calibration_path is None:
            QMessageBox.warning(
                self,
                "Missing calibration",
                "Select calibration.json.",
            )
            return

        if self.output_folder is None:
            QMessageBox.warning(
                self,
                "Missing output folder",
                "Select an output folder.",
            )
            return

        if (
            self.maximum_depth.value()
            <= self.minimum_depth.value()
        ):
            QMessageBox.warning(
                self,
                "Invalid depth range",
                "Maximum depth must be greater than minimum depth.",
            )
            return

        all_pair_folders = sorted(
            [
                folder
                for folder in self.matched_folder.iterdir()
                if folder.is_dir()
                and folder.name.lower().startswith("pair_")
            ]
        )

        requested_pairs = self.number_of_pairs.value()

        if requested_pairs == 0:
            selected_pairs = all_pair_folders
        else:
            selected_pairs = all_pair_folders[
                :requested_pairs
            ]

        if not selected_pairs:
            QMessageBox.warning(
                self,
                "No pairs",
                "There are no pair folders to process.",
            )
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)

        self.status_label.setText(
            f"Processing {len(selected_pairs)} pair(s)..."
        )

        self.worker = BatchWorker(
            pair_folders=selected_pairs,
            calibration_path=self.calibration_path,
            output_folder=self.output_folder,
            minimum_depth=self.minimum_depth.value(),
            maximum_depth=self.maximum_depth.value(),
            point_radius=self.point_radius.value(),
        )

        self.worker.progress.connect(
            self.update_progress
        )

        self.worker.log_message.connect(
            self.log_box.appendPlainText
        )

        self.worker.preview_ready.connect(
            self.viewer.set_cv_image
        )

        self.worker.completed.connect(
            self.processing_completed
        )

        self.worker.failed.connect(
            self.processing_failed
        )

        self.worker.start()

    def update_progress(self, current: int, total: int):
        percentage = int(
            round((current / total) * 100)
        )

        self.progress_bar.setValue(percentage)

        self.status_label.setText(
            f"Processing pair {current} of {total} "
            f"({percentage}%)"
        )

    def processing_completed(self, result: dict):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)

        message = (
            "Batch projection completed.\n\n"
            f"Total pairs: {result['total']}\n"
            f"Successful: {result['successful']}\n"
            f"Failed: {result['failed']}\n\n"
            f"Output folder:\n"
            f"{result['output_folder']}\n\n"
            f"CSV summary:\n"
            f"{result['summary_path']}"
        )

        self.status_label.setText(
            f"Completed: {result['successful']} successful, "
            f"{result['failed']} failed."
        )

        QMessageBox.information(
            self,
            "Processing completed",
            message,
        )

    def processing_failed(self, error_message: str):
        self.start_button.setEnabled(True)

        self.status_label.setText(
            "Processing stopped because of an error."
        )

        QMessageBox.critical(
            self,
            "Processing error",
            error_message,
        )


def main():
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
