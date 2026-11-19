"""Paket deteksi dan penghitungan gingerbread 3D kubus secara non-deep learning (Tugas 2)."""

from .baselines import (
    predict_cube_contour_hsv,
    predict_cube_orb_ransac,
    predict_cube_sift_ransac,
)
from .detector import (
    align_and_score_cookie,
    predict_cube_non_dl,
    prepare_reference_cache,
    segment_face_cookies,
)
from .geometry import (
    CANONICAL_FACE_SIZE,
    compute_vanishing_center,
    extract_cube_hexagon,
    fit_background_model,
    rectify_face,
    segment_cube_faces,
)

__all__ = [
    "CANONICAL_FACE_SIZE",
    "align_and_score_cookie",
    "compute_vanishing_center",
    "extract_cube_hexagon",
    "fit_background_model",
    "predict_cube_contour_hsv",
    "predict_cube_non_dl",
    "predict_cube_orb_ransac",
    "predict_cube_sift_ransac",
    "prepare_reference_cache",
    "rectify_face",
    "segment_cube_faces",
    "segment_face_cookies",
]
