"""Vendored subset of the OFFICIAL Wan2.2-Animate preprocessing (Alibaba Wan Team, Apache-2.0).

Source: https://github.com/Wan-Video/Wan2.2  (wan/modules/animate/preprocess/)
License: Apache-2.0. See NOTICE.txt for full attribution and the exact modification.

Why vendored: diffusers' ``WanAnimatePipeline`` expects ALREADY-preprocessed pose + face videos;
the official diffusers docs say to use the Wan-Animate repo's preprocessing, which is not published
as a pip package. We vendor only the dependency-light, animate-mode modules (person detection +
whole-body keypoints + skeleton rendering) so the pose-skeleton format matches exactly what the
model was trained on. SAM2 (replace mode), FLUX retargeting, ``decord`` and ``moviepy`` are NOT
vendored — :class:`slopmachine.pipeline.pose.PoseStage` does the animate-mode orchestration with
cv2 + imageio instead.

Heavy deps (torch / onnxruntime / matplotlib) are imported by these modules at import time, so
import this package lazily (PoseStage does).
"""

from .human_visualization import draw_aapose_by_meta_new
from .pose2d import Pose2d
from .pose2d_utils import AAPoseMeta, load_pose_metas_from_kp2ds_seq
from .utils import get_face_bboxes, get_frame_indices, padding_resize, resize_by_area

__all__ = [
    "Pose2d",
    "AAPoseMeta",
    "load_pose_metas_from_kp2ds_seq",
    "draw_aapose_by_meta_new",
    "resize_by_area",
    "get_frame_indices",
    "padding_resize",
    "get_face_bboxes",
]
