"""PoseStage — driving-signal extraction for AI dance ("pose skeleton" / motion transfer).

Reimplements the OFFICIAL Wan2.2-Animate "animate" (non-retarget) preprocessing using the vendored
Wan modules (``slopmachine.vendor.wan_animate_preprocess``) for the exact skeleton/keypoint format,
but with lean, Windows-friendly I/O (cv2 + imageio) instead of decord + moviepy. It turns a raw
driving video into the ``src_pose.mp4`` (skeleton) + ``src_face.mp4`` that ``WanAnimatePipeline``
expects (the diffusers pipeline does not include this step).

The ONNX pose models run on CPU onnxruntime — the GPU build lacks Blackwell sm_120 kernels, and
pose extraction on a short clip is light. The heavy diffusion runs separately on torch (cu130).

Heavy imports are deferred into methods so importing this module stays cheap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .. import config
from ..registry import get_model
from .base import Stage

# The two ONNX checkpoints, relative to the registry model's `subfolder` (process_checkpoint/).
_DET_REL = ("det", "yolov10m.onnx")
_POSE_REL = ("pose2d", "vitpose_h_wholebody.onnx")


def ensure_pose_assets(spec) -> Path:
    """Fetch only the detector + pose ONNX checkpoints (not SAM2/FLUX) and return their dir."""
    # Set HF_HOME BEFORE huggingface_hub is imported (it reads the cache location at import time),
    # so we hit the repo-local cache regardless of who calls this (CLI sets it early; tests may not).
    config.configure_hf_cache()
    from huggingface_hub import snapshot_download

    sub = (spec.subfolder or "").strip("/")
    patterns = [f"{sub}/{_DET_REL[0]}/*", f"{sub}/{_POSE_REL[0]}/*"] if sub else None
    local = Path(snapshot_download(spec.repo_id, allow_patterns=patterns))
    return local / sub if sub else local


class PoseStage(Stage):
    name = "pose"

    def __init__(self, model_key: Optional[str] = None):
        self.model_key, self.spec = get_model("pose", model_key)
        self._pose2d = None
        self._ckpt: Optional[Path] = None

    def load(self):
        if self._pose2d is not None:
            return self._pose2d
        os.environ.setdefault("MPLBACKEND", "Agg")  # headless skeleton rendering
        from ..vendor.wan_animate_preprocess import Pose2d

        ckpt = ensure_pose_assets(self.spec)
        det = ckpt.joinpath(*_DET_REL)
        pose_ckpt = ckpt.joinpath(*_POSE_REL)  # a directory (external-data ONNX)
        if not det.exists() or not pose_ckpt.exists():
            raise config.SlopError(
                f"Pose models missing under {ckpt}. Run `slop assets download pose`."
            )
        # device="cpu" -> CPUExecutionProvider (no Blackwell sm_120 kernel issues).
        self._pose2d = Pose2d(checkpoint=str(pose_ckpt), detector_checkpoint=str(det), device="cpu")
        self._ckpt = ckpt
        return self._pose2d

    def _read_video_rgb(self, path: Any, target_fps):
        """Read a video as RGB frames resampled to target_fps (emulates the official decord path)."""
        import cv2
        import numpy as np

        from ..vendor.wan_animate_preprocess import get_frame_indices

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise config.SlopError(f"Could not open driving video: {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        bgr_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bgr_frames.append(frame)
        cap.release()
        if not bgr_frames:
            raise config.SlopError(f"Driving video has no frames: {path}")

        out_fps = src_fps if (target_fps in (None, -1)) else float(target_fps)
        n = len(bgr_frames)
        target_n = max(1, int(n / src_fps * out_fps))
        idxs = get_frame_indices(n, src_fps, target_n, out_fps)
        # cv2 gives BGR; convert to RGB so the rest mirrors the official (decord-RGB) pipeline exactly.
        frames = np.stack([cv2.cvtColor(bgr_frames[i], cv2.COLOR_BGR2RGB) for i in idxs])
        return frames, out_fps

    def run(
        self,
        driving_video: Any,
        reference_image: Any,
        out_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import cv2
        import numpy as np

        from ..vendor.wan_animate_preprocess import (
            AAPoseMeta,
            draw_aapose_by_meta_new,
            get_face_bboxes,
            padding_resize,
            resize_by_area,
        )

        pose2d = self.load()

        params = dict(self.spec.gen_defaults)
        params.update({k: v for k, v in kwargs.items() if v is not None})
        max_area = int(params.get("max_area", 480 * 832))
        target_fps = params.get("fps", 30)

        # 1) driving video -> RGB frames at the target fps
        frames, out_fps = self._read_video_rgb(driving_video, target_fps)
        height, width = frames.shape[1], frames.shape[2]

        # 2) reference image -> RGB, resized by area (divisor 16), like the official pipeline
        ref_bgr = cv2.imread(str(reference_image))
        if ref_bgr is None:
            raise config.SlopError(f"Could not read reference image: {reference_image}")
        ref_rgb = resize_by_area(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), max_area, divisor=16)
        ref_h, ref_w = ref_rgb.shape[:2]

        # 3) whole-body keypoints per frame (CPU ONNX). pose2d swaps channels internally,
        #    matching the official behaviour given RGB-ordered input frames.
        metas = pose2d(frames)

        # 4) face crops (512x512) from each frame
        face_images = []
        for idx, meta in enumerate(metas):
            x1, x2, y1, y2 = get_face_bboxes(
                meta["keypoints_face"][:, :2], scale=1.3, image_shape=(height, width)
            )
            crop = frames[idx][y1:y2, x1:x2]
            if crop.size == 0:
                crop = frames[idx]
            face_images.append(cv2.resize(crop, (512, 512)))

        # 5) skeleton frames (non-retarget animate mode): draw on a frame-sized black canvas,
        #    then pad/resize to the reference dimensions — the official else-branch, exactly.
        cond_images = []
        for meta in metas:
            aa = AAPoseMeta.from_humanapi_meta(meta)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cond = draw_aapose_by_meta_new(canvas, aa)
            cond_images.append(padding_resize(cond, ref_h, ref_w))

        # 6) write src_pose.mp4 + src_face.mp4 (imageio treats arrays as RGB, like moviepy did)
        out_dir = Path(out_dir) if out_dir else (config.outputs_dir() / "pose")
        out_dir.mkdir(parents=True, exist_ok=True)
        pose_path = out_dir / "src_pose.mp4"
        face_path = out_dir / "src_face.mp4"
        _write_mp4(pose_path, cond_images, out_fps)
        _write_mp4(face_path, face_images, out_fps)

        return {
            "pose_video": str(pose_path),
            "face_video": str(face_path),
            "reference_image": str(reference_image),
            "fps": int(round(out_fps)),
            "num_frames": len(cond_images),
        }


def _write_mp4(path: Path, frames: list, fps: float) -> None:
    import imageio

    writer = imageio.get_writer(
        str(path), fps=max(1, int(round(fps))), codec="libx264", quality=8, macro_block_size=None
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
