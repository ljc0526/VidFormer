import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import dlib
import h5py
import heartpy
import numpy as np
from scipy.interpolate import interp1d as interp1d
from scipy.signal import resample
import scipy.io as sio
import torch

@dataclass
class SplitConfig:
    dataset: str
    root: Path
    window_size: int = 250
    stride: int = 50
    output_size: int = 128
    landmark_model: Optional[Path] = None
    data_manifest: Optional[Path] = None
    label_manifest: Optional[Path] = None
    bbox_root: Optional[Path] = None
    subject_limit: Optional[int] = None
    data_key: Optional[str] = None
    label_key: Optional[str] = None


class FaceCropper:
    def __init__(self, landmark_model: Path) -> None:
        self.predictor = dlib.shape_predictor(str(landmark_model))
        self.detector = dlib.get_frontal_face_detector()
        self.a = np.arange(81).tolist()
        self.b = np.arange(81).tolist()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int, int, int, int]]:
        x1 = y1 = x2 = y2 = 0
        face_roi = np.zeros((128, 128, 3), dtype=frame.dtype)
        img_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)
        if len(faces) == 0:
            return face_roi, 0, (x1, y1, x2, y2)

        landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, faces[0]).parts()])
        for idx, point in enumerate(landmarks):
            self.a[idx] = point[0, 0]
            self.b[idx] = point[0, 1]

        xs = np.array(self.a)
        ys = np.array(self.b)
        x1 = max(int(np.min(xs)), 0)
        y1 = max(int(np.min(ys)), 0)
        x2 = int(np.max(xs))
        y2 = int(np.max(ys))

        center_x = int((x2 + x1) * 0.5)
        center_y = int((y2 + y1) * 0.5)
        y_min = max(center_y - 130, 0)
        y_max = min(center_y + 130, frame.shape[0])
        x_min = max(center_x - 140, 0)
        x_max = min(center_x + 140, frame.shape[1])
        face_roi = frame[y_min:y_max, x_min:x_max]
        return face_roi, len(faces), (x1, y1, x2, y2)


def default_landmark_model() -> Path:
    return (Path(__file__).resolve().parent.parent / "source" / "shape_predictor_81_face_landmarks.dat").resolve()


def default_manifest_path(root: Path, dataset: str, suffix: str) -> Path:
    return (root / f"{dataset}_{suffix}.txt").resolve()


def valid_window_starts(total_length: int, window_size: int, stride: int) -> List[int]:
    if total_length < window_size:
        return []
    return list(range(0, total_length - window_size + 1, stride))


def write_manifest_line(handle, path: Path) -> None:
    handle.write(str(path.resolve()) + "\n")


def resize_face(frame: np.ndarray, output_size: int) -> np.ndarray:
    return cv.resize(frame, (output_size, output_size))


def write_multiline_label_window(output_path: Path, tracks: Sequence[Sequence[float]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for track in tracks:
            handle.write(str(list(track)))
            handle.write("\n")


def write_single_label_window(output_path: Path, signal: Sequence[float]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(str(list(signal)))
        handle.write("\n")


def load_ubfc_label_tracks(label_path: Path) -> List[List[float]]:
    with label_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    tracks = []
    for line in lines:
        tracks.append(list(map(float, line.split())))
    return tracks


def load_pure_waveform(json_path: Path) -> List[float]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [item["Value"]["waveform"] for item in data["/FullPackage"]]


def find_pure_paths(subject_dir: Path) -> Tuple[Path, Path]:
    frame_dir = None
    label_json = None
    for item in sorted(subject_dir.iterdir()):
        if item.is_dir() and frame_dir is None:
            frame_dir = item
        elif item.is_file() and item.suffix.lower() == ".json" and label_json is None:
            label_json = item
    if frame_dir is None or label_json is None:
        raise FileNotFoundError(f"PURE subject folder is incomplete: {subject_dir}")
    return frame_dir, label_json


def open_video(video_path: Path) -> cv.VideoCapture:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    return cap


def write_detection_windows_from_frame_list(
    frame_paths: Sequence[Path],
    starts: Sequence[int],
    output_dir: Path,
    writer_fps: int,
    output_size: int,
    cropper: FaceCropper,
    data_manifest_handle,
    window_size: int,
) -> int:
    processed = 0
    for index, start in enumerate(starts, start=1):
        output_path = output_dir / f"{index}.avi"
        writer = cv.VideoWriter(
            str(output_path),
            cv.VideoWriter_fourcc("X", "V", "I", "D"),
            writer_fps,
            (output_size, output_size),
        )
        complete = True
        for frame_path in frame_paths[start:start + window_size]:
            frame = cv.imread(str(frame_path))
            if frame is None:
                complete = False
                break
            roi, _, _ = cropper.detect(frame)
            if roi.size == 0:
                complete = False
                break
            writer.write(resize_face(roi, output_size))
        writer.release()
        if not complete:
            if output_path.exists():
                output_path.unlink()
            break
        write_manifest_line(data_manifest_handle, output_path)
        processed += 1
    return processed


def resample_cohface_pulse(pulse: np.ndarray) -> np.ndarray:
    # This keeps the original COHFACE label alignment strategy used in the legacy script.
    x = np.arange(0, len(pulse), 1)
    interpolator = interp1d(x, pulse)
    x_new = np.arange(0, len(pulse) - 1, 1 / 40)
    signal = interpolator(x_new)
    return signal[0::256]


def ecg_to_ppg_like(ecg_signal: np.ndarray) -> np.ndarray:
    filtered = heartpy.remove_baseline_wander(ecg_signal, 125)
    filtered = resample(filtered, len(filtered) * 4)
    processed, _ = heartpy.process(heartpy.scale_data(filtered), sample_rate=125 * 4, windowsize=0.75)
    peaks = processed["peaklist"]
    beat_intervals = []
    for idx in range(1, len(peaks)):
        beat_intervals.append(peaks[idx] - peaks[idx - 1])
    half_intervals = np.array(beat_intervals) * 0.5
    result = []
    for idx, peak in enumerate(peaks):
        result.append(filtered[peak])
        if idx < len(peaks) - 1:
            result.append(filtered[peak + int(half_intervals[idx])])
    interpolator = interp1d(np.linspace(0, 1, len(result)), np.array(result), kind="cubic")
    return interpolator(np.linspace(0, 1, int(len(filtered) * 0.25 * 0.24)))


def index_find(data_label: np.ndarray, start_time: float, end_time: float) -> Tuple[int, int]:
    begin_delta = np.abs(data_label[:, 0] - start_time)
    end_delta = np.abs(data_label[:, 0] - end_time)
    begin_index = int(np.argmin(begin_delta))
    end_index = int(np.argmin(end_delta))
    target_length = 7500
    current_length = end_index - begin_index
    if current_length < target_length:
        end_index = end_index + (target_length - current_length)
    elif current_length > target_length:
        end_index = end_index - (current_length - target_length)
    return begin_index, end_index


def parse_ecg_bbox(line: str) -> Tuple[int, int, int, int]:
    values = list(map(float, line.split()))[1:]
    x, y, width, height = map(int, values)
    return max(x, 0), max(y, 0), width, height


def crop_from_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    return frame[y:y + height, x:x + width, :]


def split_ubfc(config: SplitConfig) -> None:
    cropper = FaceCropper(config.landmark_model or default_landmark_model())
    subject_dirs = sorted(path for path in config.root.iterdir() if path.is_dir())
    with config.data_manifest.open("w", encoding="utf-8") as data_handle, config.label_manifest.open("w", encoding="utf-8") as label_handle:
        for subject_dir in subject_dirs:
            video_path = subject_dir / "vid.avi"
            label_path = subject_dir / "ground_truth.txt"
            label_tracks = load_ubfc_label_tracks(label_path)
            frame_cap = open_video(video_path)
            frame_count = int(frame_cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_cap.release()
            video_starts = valid_window_starts(frame_count, config.window_size, config.stride)
            label_starts = valid_window_starts(len(label_tracks[0]), config.window_size, config.stride)
            segment_count = min(len(video_starts), len(label_starts))
            video_starts = video_starts[:segment_count]
            label_starts = label_starts[:segment_count]

            cap = open_video(video_path)
            processed = 0
            for index, start in enumerate(video_starts, start=1):
                output_path = subject_dir / f"{index}.avi"
                writer = cv.VideoWriter(
                    str(output_path),
                    cv.VideoWriter_fourcc("X", "V", "I", "D"),
                    30,
                    (config.output_size, config.output_size),
                )
                cap.set(cv.CAP_PROP_POS_FRAMES, start)
                complete = True
                for _ in range(config.window_size):
                    ret, frame = cap.read()
                    if not ret:
                        complete = False
                        break
                    roi, _, _ = cropper.detect(frame)
                    if roi.size == 0:
                        complete = False
                        break
                    writer.write(resize_face(roi, config.output_size))
                writer.release()
                if not complete:
                    if output_path.exists():
                        output_path.unlink()
                    break
                write_manifest_line(data_handle, output_path)
                processed += 1
            cap.release()

            for index, start in enumerate(label_starts[:processed], start=1):
                output_path = subject_dir / f"{index}.txt"
                tracks = [track[start:start + config.window_size] for track in label_tracks]
                write_multiline_label_window(output_path, tracks)
                write_manifest_line(label_handle, output_path)
            print(f"{subject_dir.name}: {processed} UBFC segments created")


def split_pure(config: SplitConfig) -> None:
    cropper = FaceCropper(config.landmark_model or default_landmark_model())
    subject_dirs = sorted(path for path in config.root.iterdir() if path.is_dir())
    with config.data_manifest.open("w", encoding="utf-8") as data_handle, config.label_manifest.open("w", encoding="utf-8") as label_handle:
        for subject_dir in subject_dirs:
            frame_dir, json_path = find_pure_paths(subject_dir)
            frame_paths = sorted(path for path in frame_dir.iterdir() if path.is_file())
            waveform = load_pure_waveform(json_path)
            max_seconds = min(len(frame_paths) // 30, len(waveform) // 60)
            frame_paths = frame_paths[:max_seconds * 30]
            waveform = waveform[:max_seconds * 60:2]
            starts = valid_window_starts(len(frame_paths), config.window_size, config.stride)
            starts = starts[:len(valid_window_starts(len(waveform), config.window_size, config.stride))]
            processed = write_detection_windows_from_frame_list(
                frame_paths=frame_paths,
                starts=starts,
                output_dir=subject_dir,
                writer_fps=30,
                output_size=config.output_size,
                cropper=cropper,
                data_manifest_handle=data_handle,
                window_size=config.window_size,
            )
            label_starts = valid_window_starts(len(waveform), config.window_size, config.stride)[:processed]
            for index, start in enumerate(label_starts, start=1):
                output_path = subject_dir / f"{index}.txt"
                write_single_label_window(output_path, waveform[start:start + config.window_size])
                write_manifest_line(label_handle, output_path)
            print(f"{subject_dir.name}: {processed} PURE segments created")


def split_cohface(config: SplitConfig) -> None:
    cropper = FaceCropper(config.landmark_model or default_landmark_model())
    subject_dirs = sorted(path for path in config.root.iterdir() if path.is_dir())
    with config.data_manifest.open("w", encoding="utf-8") as data_handle, config.label_manifest.open("w", encoding="utf-8") as label_handle:
        for subject_dir in subject_dirs:
            session_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir())
            for session_dir in session_dirs:
                video_path = session_dir / "data.avi"
                hdf5_path = session_dir / "data.hdf5"
                cap = open_video(video_path)
                frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                video_starts = valid_window_starts(frame_count, config.window_size, config.stride)
                previous_bbox = None
                processed = 0
                for index, start in enumerate(video_starts, start=1):
                    output_path = session_dir / f"{index}.avi"
                    writer = cv.VideoWriter(
                        str(output_path),
                        cv.VideoWriter_fourcc("X", "V", "I", "D"),
                        20,
                        (config.output_size, config.output_size),
                    )
                    cap.set(cv.CAP_PROP_POS_FRAMES, start)
                    complete = True
                    for _ in range(config.window_size):
                        ret, frame = cap.read()
                        if not ret:
                            complete = False
                            break
                        roi, detected, bbox = cropper.detect(frame)
                        if detected == 0:
                            if previous_bbox is None:
                                complete = False
                                break
                            x1, y1, x2, y2 = previous_bbox
                            roi = frame[y1:y2, x1:x2]
                        else:
                            previous_bbox = bbox
                        if roi.size == 0:
                            complete = False
                            break
                        writer.write(resize_face(roi, config.output_size))
                    writer.release()
                    if not complete:
                        if output_path.exists():
                            output_path.unlink()
                        break
                    write_manifest_line(data_handle, output_path)
                    processed += 1
                cap.release()

                with h5py.File(hdf5_path, "r") as handle:
                    pulse = np.array(handle["pulse"][:])
                signal = resample_cohface_pulse(pulse)
                label_window = config.window_size * 2
                label_stride = config.stride * 2
                label_starts = valid_window_starts(len(signal), label_window, label_stride)
                label_starts = label_starts[:processed]
                for index, start in enumerate(label_starts, start=1):
                    output_path = session_dir / f"{index}.txt"
                    write_single_label_window(output_path, signal[start:start + label_window])
                    write_manifest_line(label_handle, output_path)
                print(f"{subject_dir.name}/{session_dir.name}: {len(label_starts)} COHFACE segments created")


def split_ecg_fitness(config: SplitConfig) -> None:
    if config.bbox_root is None:
        raise ValueError("ECG-fitness splitting requires --bbox-root")

    subject_dirs = sorted(path for path in config.root.iterdir() if path.is_dir())
    if config.subject_limit is not None:
        subject_dirs = subject_dirs[:config.subject_limit]

    with config.data_manifest.open("w", encoding="utf-8") as data_handle, config.label_manifest.open("w", encoding="utf-8") as label_handle:
        for subject_dir in subject_dirs:
            session_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir())
            for session_dir in session_dirs:
                bbox_path = config.bbox_root / subject_dir.name / session_dir.name / "c920-1.face"
                video_path = session_dir / "c920-1.avi"
                ecg_csv = session_dir / "viatom-raw.csv"
                timestamp_csv = session_dir / "c920.csv"
                if not bbox_path.exists():
                    print(f"Skip {subject_dir.name}/{session_dir.name}: missing bbox file")
                    continue

                bbox_lines = bbox_path.read_text(encoding="utf-8").splitlines()
                ecg_data = np.loadtxt(ecg_csv.open("r", encoding="utf-8"), delimiter=",", skiprows=1)
                timestamps = np.loadtxt(timestamp_csv.open("r", encoding="utf-8"), delimiter=",")
                start_time = timestamps[0, 0]
                end_time = timestamps[-1, 0]
                begin_index, end_index = index_find(ecg_data, start_time, end_time)
                label_signal = ecg_to_ppg_like(ecg_data[begin_index:end_index, 1])

                cap = open_video(video_path)
                frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                video_starts = valid_window_starts(frame_count, config.window_size, config.stride)
                label_starts = valid_window_starts(len(label_signal), config.window_size, config.stride)
                segment_count = min(len(video_starts), len(label_starts))
                video_starts = video_starts[:segment_count]
                label_starts = label_starts[:segment_count]

                processed = 0
                for index, start in enumerate(video_starts, start=1):
                    output_path = session_dir / f"{index}.avi"
                    writer = cv.VideoWriter(
                        str(output_path),
                        cv.VideoWriter_fourcc("X", "V", "I", "D"),
                        30,
                        (config.output_size, config.output_size),
                    )
                    cap.set(cv.CAP_PROP_POS_FRAMES, start)
                    complete = True
                    for offset in range(config.window_size):
                        ret, frame = cap.read()
                        if not ret:
                            complete = False
                            break
                        bbox_index = start + offset
                        if bbox_index >= len(bbox_lines):
                            complete = False
                            break
                        roi = crop_from_bbox(frame, parse_ecg_bbox(bbox_lines[bbox_index]))
                        if roi.size == 0:
                            complete = False
                            break
                        writer.write(resize_face(roi, config.output_size))
                    writer.release()
                    if not complete:
                        if output_path.exists():
                            output_path.unlink()
                        break
                    write_manifest_line(data_handle, output_path)
                    processed += 1
                cap.release()

                for index, start in enumerate(label_starts[:processed], start=1):
                    output_path = session_dir / f"{index}.txt"
                    write_single_label_window(output_path, label_signal[start:start + config.window_size])
                    write_manifest_line(label_handle, output_path)
                print(f"{subject_dir.name}/{session_dir.name}: {processed} ECG-fitness segments created")


def split_mmpd(config: SplitConfig) -> dict:
    if not config.data_key or not config.label_key:
        raise ValueError("MMPD splitting requires both --data-key and --label-key")

    landmark_model = config.landmark_model or default_landmark_model()
    predictor = dlib.shape_predictor(str(landmark_model))
    detector = dlib.get_frontal_face_detector()
    subject_dirs = sorted(path for path in config.root.iterdir() if path.is_dir())
    saved_summary = {}

    with config.data_manifest.open("w", encoding="utf-8") as data_handle, config.label_manifest.open("w", encoding="utf-8") as label_handle:
        for subject_dir in subject_dirs:
            mat_files = sorted(subject_dir.glob("*.mat"))
            save_index = 1
            created = 0

            for mat_path in mat_files:
                print(f"Split: {mat_path}")
                mat = sio.loadmat(mat_path, variable_names=[config.data_key, config.label_key])
                if config.data_key not in mat or config.label_key not in mat:
                    print(f"Skip {mat_path}: missing key {config.data_key} or {config.label_key}")
                    continue

                data = mat[config.data_key]
                label = mat[config.label_key]
                total_length = data.shape[0]
                if total_length < config.window_size:
                    continue

                face_video, valid_mask = face_detection_full_video(
                    data,
                    detector=detector,
                    predictor=predictor,
                    output_size=config.output_size,
                )

                for start in valid_window_starts(total_length, config.window_size, config.stride):
                    end = start + config.window_size
                    valid_number = int(valid_mask[start:end].sum())
                    if valid_number < config.window_size:
                        print(f"Skip window {start}-{end} in {mat_path}: valid face frames = {valid_number}")
                        continue

                    data_window = face_video[start:end, ...]
                    label_window = label[..., start:end]

                    data_npy_path = subject_dir / f"{save_index}_data.npy"
                    label_npy_path = subject_dir / f"{save_index}_label.npy"

                    np.save(data_npy_path, data_window.astype(np.float32))
                    np.save(label_npy_path, label_window.astype(np.float32))
                    write_manifest_line(data_handle, data_npy_path)
                    write_manifest_line(label_handle, label_npy_path)

                    save_index += 1
                    created += 1

                print(f"Finish split: {mat_path}")

            saved_summary[str(subject_dir)] = created
            print(f"{subject_dir.name}: {created} MMPD segments created")
    return saved_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset splitting for UBFC, PURE, COHFACE, ECG-fitness, and MMPD.")
    parser.add_argument("--dataset", required=True, choices=["ubfc", "pure", "cohface", "ecg-fitness", "mmpd"])
    parser.add_argument("--root", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--window-size", default=250, type=int, help="Number of frames per video window.")
    parser.add_argument("--stride", default=50, type=int, help="Sliding-window stride.")
    parser.add_argument("--output-size", default=128, type=int, help="Output face crop size.")
    parser.add_argument("--landmark-model", type=Path, default=None, help="Path to the dlib 81-point landmark model.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Output path for the split video list.")
    parser.add_argument("--label-manifest", type=Path, default=None, help="Output path for the split label list.")
    parser.add_argument("--bbox-root", type=Path, default=None, help="BBox root for ECG-fitness.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional subject limit for ECG-fitness.")
    parser.add_argument("--data-key", type=str, default=None, help="MAT key for video data, used by MMPD.")
    parser.add_argument("--label-key", type=str, default=None, help="MAT key for waveform label, used by MMPD.")
    return parser


def build_config(args: argparse.Namespace) -> SplitConfig:
    root = args.root.resolve()
    landmark_model = args.landmark_model.resolve() if args.landmark_model else default_landmark_model()
    data_manifest = args.data_manifest.resolve() if args.data_manifest else default_manifest_path(root, args.dataset, "data")
    label_manifest = args.label_manifest.resolve() if args.label_manifest else default_manifest_path(root, args.dataset, "label")
    bbox_root = args.bbox_root.resolve() if args.bbox_root else None
    return SplitConfig(
        dataset=args.dataset,
        root=root,
        window_size=args.window_size,
        stride=args.stride,
        output_size=args.output_size,
        landmark_model=landmark_model,
        data_manifest=data_manifest,
        label_manifest=label_manifest,
        bbox_root=bbox_root,
        subject_limit=args.subject_limit,
        data_key=args.data_key,
        label_key=args.label_key,
    )

def MMPD_split(
    root,
    data_key,
    label_key,
    split_number=250,
    window_length=400,
    output_size=128,
    landmark_model_path="../source/shape_predictor_81_face_landmarks.dat",
):
    config = SplitConfig(
        dataset="mmpd",
        root=Path(root).resolve(),
        window_size=window_length,
        stride=split_number,
        output_size=output_size,
        landmark_model=Path(landmark_model_path).resolve(),
        data_manifest=default_manifest_path(Path(root).resolve(), "mmpd", "data"),
        label_manifest=default_manifest_path(Path(root).resolve(), "mmpd", "label"),
        data_key=data_key,
        label_key=label_key,
    )
    return split_mmpd(config)

def face_detection_full_video(
        video,
        detector,
        predictor,
        output_size=128,
        threshold=-0.5,
):
    """
    Face ROI extraction for a full video. Detect face and landmarks for every frame.

    Args:
        video: numpy array, shape [T, H, W, 3], value range can be [0, 1] or [0, 255].
        detector: dlib face detector.
        predictor: dlib shape predictor.
        output_size: output face size.
        threshold: dlib detector threshold.

    Returns:
        face_roi: shape [T, output_size, output_size, 3]. The value scale follows input video.
        valid_mask: shape [T], 1 if face ROI is valid.
    """
    T, H, W, C = video.shape
    if C != 3:
        raise ValueError(f"Expected RGB video with 3 channels, but got {C} channels.")

    face_roi = np.zeros((T, output_size, output_size, 3), dtype=np.float32)
    valid_mask = np.zeros((T,), dtype=np.int32)
    last_roi = np.zeros((output_size, output_size, 3), dtype=np.float32)

    for j in range(T):
        frame_float = video[j, ...].astype(np.float32)

        # Build uint8 frame for dlib detection robustly.
        if frame_float.max() > 2.0:
            frame_uint8 = np.clip(frame_float, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = (frame_float * 255).clip(0, 255).astype(np.uint8)

        gray = cv.cvtColor(frame_uint8, cv.COLOR_RGB2GRAY)
        faces, scores, _ = detector.run(gray, 0, threshold)

        if len(faces) > 0:
            best_idx = int(np.argmax(scores))
            face = faces[best_idx]

            shape = predictor(gray, face)
            landmark_num = min(81, len(shape.parts()))
            landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(landmark_num)],
                dtype=np.int32,
            )

            x1 = max(0, int(np.min(landmarks[:, 0])))
            x2 = min(W, int(np.max(landmarks[:, 0])))
            y1 = max(0, int(np.min(landmarks[:, 1])))
            y2 = min(H, int(np.max(landmarks[:, 1])))

            if x2 > x1 and y2 > y1:
                crop = frame_float[y1:y2, x1:x2, :]
                if crop.size > 0:
                    crop = cv.resize(crop, (output_size, output_size))
                    face_roi[j, ...] = crop.astype(np.float32)
                    last_roi = crop.astype(np.float32)
                    valid_mask[j] = 1
                else:
                    face_roi[j, ...] = last_roi
            else:
                face_roi[j, ...] = last_roi
        else:
            face_roi[j, ...] = last_roi

    return face_roi, valid_mask


def convert_npz_to_h5(root):
    root = Path(root)
    generated_h5_files = []

    for npz_path in sorted(root.rglob("*.npz")):
        print("Start convert npz to h5: ", npz_path)
        npz_data = np.load(npz_path)
        h5_path = npz_path.with_suffix(".h5")

        with h5py.File(h5_path, "w") as h5_file:
            for key in npz_data.files:
                h5_file.create_dataset(key, data=np.asarray(npz_data[key]))

        generated_h5_files.append(str(h5_path))
        print("Finish convert npz to h5: ", h5_path)

    return {
        "converted_npz_count": len(generated_h5_files),
        "generated_h5_files": generated_h5_files,
    }


def delete_h5_files(root):
    root = Path(root)
    deleted_files = []

    for h5_path in sorted(root.rglob("*.h5")):
        print("Delete: ", h5_path)
        h5_path.unlink()
        deleted_files.append(str(h5_path))
        print("Finish delete: ", h5_path)

    return {
        "deleted_h5_count": len(deleted_files),
        "deleted_files": deleted_files,
    }


def delete_npy_files_with_zero_var(root):
    root = Path(root)
    deleted_files = []

    for npy_path in sorted(root.rglob("*_label.npy")):
        data_path = npy_path.with_name(npy_path.name.replace('_label', '_data'))
        label = np.load(npy_path)
        label = torch.from_numpy(label.copy()).float()
        label_var = torch.var(label, unbiased=False)

        if (not torch.isfinite(label_var)) or label_var <= 1e-12:
            print("Delete: ", npy_path)
            npy_path.unlink()
            if data_path.exists():
                data_path.unlink()
            deleted_files.append(str(npy_path))
            print("Finish delete: ", npy_path)
        else:
            print("check: ", npy_path)

    return {
        "deleted_npy_count": len(deleted_files),
        "deleted_files": deleted_files,
    }


def delete_all_npy_files(root):
    root = Path(root)
    deleted_files = []

    for npy_path in sorted(root.rglob("*.npy")):
        print("Delete: ", npy_path)
        npy_path.unlink()
        deleted_files.append(str(npy_path))
        print("Finish delete: ", npy_path)

    return {
        "deleted_npy_count": len(deleted_files),
        "deleted_files": deleted_files,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)

    if config.dataset == "ubfc":
        split_ubfc(config)
    elif config.dataset == "pure":
        split_pure(config)
    elif config.dataset == "cohface":
        split_cohface(config)
    elif config.dataset == "ecg-fitness":
        split_ecg_fitness(config)
    elif config.dataset == "mmpd":
        split_mmpd(config)



if __name__ == "__main__":
    main()
