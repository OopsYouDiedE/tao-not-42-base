"""Export one Kubric MOVi sample from TFDS to plain local files.

Run this in Colab or any environment that can read
gs://kubric-public/tfds. The output .npz avoids a runtime dependency on
TensorFlow Datasets for local debugging/training checks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow_datasets as tfds


def _decode_uint16(encoded: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    encoded = encoded.astype(np.float32)
    minv, maxv = np.asarray(value_range, dtype=np.float32)
    return encoded / 65535.0 * (maxv - minv) + minv


def _save_contact_sheet(video: np.ndarray, path: Path, frames: int = 6) -> None:
    sample_ids = np.linspace(0, len(video) - 1, min(frames, len(video))).round().astype(int)
    thumbs = []
    for idx in sample_ids:
        frame = video[idx]
        thumbs.append(frame)

    h, w = thumbs[0].shape[:2]
    rows = 2
    cols = int(np.ceil(len(thumbs) / rows))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(thumbs):
        y = (i // cols) * h
        x = (i % cols) * w
        sheet[y : y + h, x : x + w] = frame
    imageio.imwrite(path, sheet)


def export_sample(
    dataset: str,
    split: str,
    index: int,
    out: Path,
    data_dir: str,
) -> None:
    ds = tfds.load(dataset, data_dir=data_dir, split=split, shuffle_files=False)
    sample = next(iter(tfds.as_numpy(ds.skip(index).take(1))))

    metadata = sample["metadata"]
    depth_m = _decode_uint16(sample["depth"], metadata["depth_range"])
    forward_flow_px = _decode_uint16(
        sample["forward_flow"], metadata["forward_flow_range"]
    )
    backward_flow_px = _decode_uint16(
        sample["backward_flow"], metadata["backward_flow_range"]
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        video=sample["video"],
        segmentation=sample["segmentations"][..., 0],
        depth_encoded=sample["depth"],
        depth_m=depth_m[..., 0],
        forward_flow_encoded=sample["forward_flow"],
        forward_flow_px=forward_flow_px,
        backward_flow_encoded=sample["backward_flow"],
        backward_flow_px=backward_flow_px,
        cam_pos=sample["camera"]["positions"],
        cam_quat=sample["camera"]["quaternions"],
        camera_focal_length=np.asarray(sample["camera"]["focal_length"]),
        camera_sensor_width=np.asarray(sample["camera"]["sensor_width"]),
        camera_field_of_view=np.asarray(sample["camera"]["field_of_view"]),
        depth_range=np.asarray(metadata["depth_range"]),
        forward_flow_range=np.asarray(metadata["forward_flow_range"]),
        backward_flow_range=np.asarray(metadata["backward_flow_range"]),
        is_dynamic=np.asarray(sample["instances"]["is_dynamic"]),
        video_name=np.asarray(metadata["video_name"]),
    )

    preview = out.with_suffix(".preview.png")
    _save_contact_sheet(sample["video"], preview)

    print(f"saved_npz={out}")
    print(f"saved_preview={preview}")
    print(f"video_shape={sample['video'].shape} dtype={sample['video'].dtype}")
    print(f"depth_m_range=({float(depth_m.min()):.6f}, {float(depth_m.max()):.6f})")
    print(
        "forward_flow_px_range="
        f"({float(forward_flow_px.min()):.6f}, {float(forward_flow_px.max()):.6f})"
    )
    print(f"cam_quat_first={sample['camera']['quaternions'][0].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movi_e/256x256")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data_dir", default="gs://kubric-public/tfds")
    parser.add_argument("--out", default="movi_e_sample_0000.npz")
    args = parser.parse_args()

    export_sample(
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        out=Path(args.out),
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
