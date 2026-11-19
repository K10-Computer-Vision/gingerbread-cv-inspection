"""Model pembanding Non-Deep Learning (SIFT+RANSAC, ORB+RANSAC, & Contour+HSV).

Modul ini menyediakan tiga pipeline Non-Deep Learning tambahan sesuai opsi
resmi Bagian 6.A PROJECT_SPEC.md untuk keperluan studi ablasi dan perbandingan
objektif terhadap pipeline utama `predict_cube_non_dl` pada Tugas 2.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import (
    _augment_references_from_template_dir,
    _fit_shading_affine,
    _major_axis_angle_deg,
    _rotate_crop_patch,
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

_BASELINE_CACHE_STORE: dict[tuple[str, tuple[tuple[int, tuple[int, int, int]], ...]], dict] = {}


def _build_feature_config(method: str) -> tuple:
    """Mengembalikan konfigurasi detektor, matcher, dan ambang RANSAC per metode."""
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=850, contrastThreshold=0.012, edgeThreshold=16)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return det, matcher, 0.82, 4.8, 380.0
    det = cv2.ORB_create(nfeatures=950, fastThreshold=5, scaleFactor=1.12, nlevels=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return det, matcher, 0.82, 4.2, 320.0


def _hsv_vnorm_vec(hsv_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Menghitung vektor histogram gabungan [H,S] (16x6) dan [V] (8) ternormalisasi L2."""
    m_u8 = (mask > 0).astype(np.uint8)
    if int(m_u8.sum()) < 5:
        return np.zeros(16 * 6 + 8, dtype=np.float32)
    h_hs = cv2.calcHist([hsv_img], [0, 1], m_u8, [16, 6], [0, 180, 0, 256]).ravel().astype(np.float32)
    h_hs /= float(np.linalg.norm(h_hs)) + 1e-8
    h_v = cv2.calcHist([hsv_img], [2], m_u8, [8], [0, 256]).ravel().astype(np.float32)
    h_v /= float(np.linalg.norm(h_v)) + 1e-8
    vec = np.concatenate([0.85 * h_hs, 0.52 * h_v])
    vec /= float(np.linalg.norm(vec)) + 1e-8
    return vec


def _umeyama_2d_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Mengestimasi transformasi kesebangunan 2D 4-DOF secara analitik via SVD Umeyama."""
    if len(src) < 3:
        return None
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    cs = src - mu_s
    cd = dst - mu_d
    var_s = float(np.sum(cs * cs))
    if var_s < 1e-6:
        return None
    u_mat, s_vals, vt_mat = np.linalg.svd(cs.T @ cd)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1.0
        s_vals[-1] *= -1.0
        rot = vt_mat.T @ u_mat.T
    scale = float(np.sum(s_vals) / var_s)
    if not (0.18 <= scale <= 0.90):
        return None
    mat = np.zeros((2, 3), dtype=np.float64)
    mat[:, :2] = scale * rot
    mat[:, 2] = mu_d - scale * (rot @ mu_s)
    return mat


def _extract_contour_zone_stats(
    rgb: np.ndarray, solid_mask: np.ndarray, raw_mask: np.ndarray, n_strips: int = 6
) -> dict:
    """Mengekstrak statistik kontur, profil lebar, serta histogram HSV V-ternormalisasi."""
    ys, xs = np.where(solid_mask > 0)
    if len(ys) == 0:
        empty_h = np.zeros(16 * 6 + 8, dtype=np.float32)
        return {
            "g_hist": empty_h,
            "z3": [empty_h, empty_h, empty_h],
            "strips": [empty_h] * n_strips,
            "widths": np.zeros(n_strips, dtype=np.float32),
            "ar": 1.0,
            "solidity": 1.0,
            "contour": np.zeros((0, 1, 2), dtype=np.int32),
        }

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub_rgb = rgb[y0:y1, x0:x1]
    sub_sol = solid_mask[y0:y1, x0:x1] > 0
    sub_raw = raw_mask[y0:y1, x0:x1] > 0
    bh, bw = sub_sol.shape

    hsv = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2HSV)
    # Normalisasi median kanal V agar tahan terhadap perbedaan pencahayaan antarsisi kubus.
    v_med = float(np.median(hsv[:, :, 2][sub_sol])) if int(sub_sol.sum()) > 0 else 128.0
    hsv_norm = hsv.copy()
    hsv_norm[:, :, 2] = np.clip(
        hsv[:, :, 2].astype(np.float32) * (135.0 / max(25.0, v_med)), 0, 255
    ).astype(np.uint8)

    strips: list[np.ndarray] = []
    widths: list[float] = []
    edges = [int(round(i * bh / n_strips)) for i in range(n_strips + 1)]
    for i in range(n_strips):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        sm = np.zeros_like(sub_raw, dtype=np.uint8)
        sm[a:b, :] = sub_raw[a:b, :]
        strips.append(_hsv_vnorm_vec(hsv_norm, sm))
        widths.append(float(sub_sol[a:b, :].sum()) / float(max(1, (b - a) * bw)))

    z3: list[np.ndarray] = []
    e3 = [0, max(1, bh // 3), max(2, (2 * bh) // 3), bh]
    for i in range(3):
        sm = np.zeros_like(sub_raw, dtype=np.uint8)
        sm[e3[i] : e3[i + 1], :] = sub_raw[e3[i] : e3[i + 1], :]
        z3.append(_hsv_vnorm_vec(hsv_norm, sm))

    cnts, _ = cv2.findContours(sub_sol.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_cnt = max(cnts, key=cv2.contourArea) if cnts else np.zeros((0, 1, 2), dtype=np.int32)
    solidity = 1.0
    if len(main_cnt) >= 3:
        ha = float(cv2.contourArea(cv2.convexHull(main_cnt)))
        solidity = float(cv2.contourArea(main_cnt)) / max(1.0, ha)

    return {
        "g_hist": _hsv_vnorm_vec(hsv_norm, sub_raw),
        "z3": z3,
        "strips": strips,
        "widths": np.array(widths, dtype=np.float32),
        "ar": float(bh) / max(1.0, float(bw)),
        "solidity": float(solidity),
        "contour": main_cnt,
    }


def _prepare_feature_refs(references: dict, method: str) -> dict[int, dict]:
    """Mengekstrak serta meng-cache fitur lokal SIFT atau ORB untuk ketujuh referensi."""
    _augment_references_from_template_dir(references)
    sig = (method, tuple((int(k), tuple(v["rgba"].shape)) for k, v in sorted(references.items())))
    if sig in _BASELINE_CACHE_STORE:
        return _BASELINE_CACHE_STORE[sig]

    det, _, _, _, _ = _build_feature_config(method)

    cache: dict[int, dict] = {}
    for did, info in references.items():
        rgba = info["rgba"]
        alpha = rgba[:, :, 3] > 128
        ys, xs = np.where(alpha)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, int(ys.min()) - 6), min(rgba.shape[0], int(ys.max()) + 7)
        x0, x1 = max(0, int(xs.min()) - 6), min(rgba.shape[1], int(xs.max()) + 7)
        crop = rgba[y0:y1, x0:x1]

        rgb = crop[:, :, :3].copy()
        mask = (crop[:, :, 3] > 128).astype(np.uint8) * 255
        # Isi piksel luar mask dengan abu-abu netral agar tepi transparan tidak memicu keypoint palsu.
        rgb[mask == 0] = 195
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        kps, des = det.detectAndCompute(gray, mask)
        pts = np.float32([kp.pt for kp in kps]) if kps else np.zeros((0, 2), dtype=np.float32)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], (mask > 0).astype(np.uint8), [12, 4], [0, 180, 0, 256]).ravel().astype(np.float32)
        hist /= float(np.linalg.norm(hist)) + 1e-8

        cache[int(did)] = {
            "name": info.get("name", f"design_{did}"),
            "rgb": rgb.astype(np.float32),
            "mask": (mask > 0).astype(np.uint8),
            "kps_pts": pts,
            "des": des,
            "hist": hist,
            "h": crop.shape[0],
            "w": crop.shape[1],
        }

    _BASELINE_CACHE_STORE[sig] = cache
    return cache


def _prepare_contour_hsv_refs(references: dict) -> dict[int, dict]:
    """Menyiapkan statistik kontur, profil lebar 6-strip, dan histogram HSV referensi."""
    _augment_references_from_template_dir(references)
    sig = ("contour_hsv_v3", tuple((int(k), tuple(v["rgba"].shape)) for k, v in sorted(references.items())))
    if sig in _BASELINE_CACHE_STORE:
        return _BASELINE_CACHE_STORE[sig]

    cache: dict[int, dict] = {}
    for did, info in references.items():
        rgba = info["rgba"]
        alpha = (rgba[:, :, 3] > 128).astype(np.uint8)
        ys, xs = np.where(alpha > 0)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, int(ys.min()) - 2), min(rgba.shape[0], int(ys.max()) + 3)
        x0, x1 = max(0, int(xs.min()) - 2), min(rgba.shape[1], int(xs.max()) + 3)
        st = _extract_contour_zone_stats(rgba[y0:y1, x0:x1, :3], alpha[y0:y1, x0:x1], alpha[y0:y1, x0:x1])
        st["name"] = info.get("name", f"design_{did}")
        cache[int(did)] = st

    _BASELINE_CACHE_STORE[sig] = cache
    return cache


def _intrinsic_defect_metrics(warped_face: np.ndarray, comp: dict, card_med: np.ndarray) -> tuple[float, float]:
    """Menghitung indikator bentuk (solidity drop) dan lubang kardus tanpa template ter-align."""
    cnts, _ = cv2.findContours(comp["mask_solid"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    hull_mask = np.zeros_like(comp["mask_solid"])
    if cnts:
        all_pts = np.vstack(cnts)
        hull = cv2.convexHull(all_pts)
        cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
        ha = float(cv2.contourArea(hull))
        solidity = float(comp["area"]) / max(1.0, ha)

    solid_er = cv2.erode(comp["mask_solid"], np.ones((3, 3), np.uint8), iterations=1) > 0
    if comp.get("was_split", False):
        hole_region = solid_er | (cv2.erode(hull_mask, np.ones((5, 5), np.uint8), iterations=1) > 0)
    else:
        hole_region = solid_er

    diff_card = np.linalg.norm(warped_face.astype(np.float32) - card_med, axis=2)
    hole_frac = float(np.sum(hole_region & (diff_card < 13.5))) / float(max(1, int(hole_region.sum())))
    ratio_sil = max(0.0, (0.68 - solidity) / 0.10)
    return float(ratio_sil), float(hole_frac)


def _score_contour_hsv_cookie(
    warped_face: np.ndarray,
    comp: dict,
    ref_cache: dict[int, dict],
    card_med: np.ndarray,
) -> dict:
    """Menilai satu cookie memakai orientasi sumbu utama kontur dan HSV 6-strip dual-anchor."""
    theta, ar = _major_axis_angle_deg(comp["mask_solid"])
    cand_angles = [theta, theta + 180.0]
    if ar < 1.48:
        cand_angles.extend([0.0, 90.0, 180.0, -90.0])

    best_did = next(iter(ref_cache))
    best_sc = -1e9
    best_pair: tuple[dict, dict, float, float, float, float, float] | None = None

    for ang in cand_angles:
        rc = _rotate_crop_patch(warped_face, comp, ang, card_med)
        if rc is None:
            continue
        obs_st = _extract_contour_zone_stats(rc[0], rc[1], rc[2], n_strips=6)
        for did, r_st in ref_cache.items():
            g_sim = float(np.dot(obs_st["g_hist"], r_st["g_hist"]))
            z3_sims = [float(np.dot(a, b)) for a, b in zip(obs_st["z3"], r_st["z3"])]
            s6_full = [float(np.dot(a, b)) for a, b in zip(obs_st["strips"], r_st["strips"])]
            w_pen_full = float(np.mean(np.abs(obs_st["widths"] - r_st["widths"])))
            # Trimmed mean membuang 1 strip terburuk agar cacat lokal tidak menggeser design_id.
            best_anchor = sum(sorted(s6_full)[1:]) / 5.0 - 0.18 * w_pen_full

            # Penyelarasan jangkar atas/bawah saat cookie mengalami potongan tepi (boundary_cut).
            if obs_st["ar"] < 0.88 * r_st["ar"]:
                k = max(3, min(5, int(round(6.0 * obs_st["ar"] / max(0.5, r_st["ar"])))))
                obs_k = _extract_contour_zone_stats(rc[0], rc[1], rc[2], n_strips=k)
                top_sims = [float(np.dot(a, b)) for a, b in zip(obs_k["strips"], r_st["strips"][:k])]
                bot_sims = [float(np.dot(a, b)) for a, b in zip(obs_k["strips"], r_st["strips"][6 - k :])]
                top_sc = sum(top_sims) / k - 0.18 * float(np.mean(np.abs(obs_k["widths"] - r_st["widths"][:k])))
                bot_sc = sum(bot_sims) / k - 0.18 * float(np.mean(np.abs(obs_k["widths"] - r_st["widths"][6 - k :])))
                best_anchor = max(best_anchor, top_sc, bot_sc)

            s_dist = (
                float(cv2.matchShapes(obs_st["contour"], r_st["contour"], cv2.CONTOURS_MATCH_I1, 0.0))
                if len(obs_st["contour"]) >= 5 and len(r_st["contour"]) >= 5
                else 1.0
            )
            ar_pen = abs(obs_st["ar"] - r_st["ar"]) / max(1.0, r_st["ar"])
            sc = 0.25 * g_sim + 0.75 * best_anchor - 0.08 * min(s_dist, 2.5) - 0.05 * min(ar_pen, 1.5)
            if sc > best_sc:
                best_sc = sc
                best_did = did
                best_pair = (obs_st, r_st, g_sim, min(z3_sims), s_dist, ar_pen, w_pen_full)

    ratio_sil_intr, hole_frac_intr = _intrinsic_defect_metrics(warped_face, comp, card_med)
    if best_pair is None:
        defect_ratio = max(ratio_sil_intr, hole_frac_intr / 0.048)
    else:
        obs_st, r_st, g_sim, min_z3, s_dist, ar_pen, w_pen_full = best_pair
        defect_ratio = max(
            ratio_sil_intr,
            hole_frac_intr / 0.048,
            max(0.0, (0.910 - min_z3) / 0.075),
            max(0.0, (0.952 - g_sim) / 0.04),
            max(0.0, (ar_pen - 0.070) / 0.05),
            max(0.0, ((r_st["solidity"] - obs_st["solidity"]) - 0.165) / 0.05),
            max(0.0, (s_dist - 0.70) / 0.25),
            max(0.0, (w_pen_full - 0.115) / 0.04),
        )

    return {
        "design_id": int(best_did),
        "defect_ratio": float(defect_ratio),
        "is_defective": bool(defect_ratio > 1.0),
    }


def _score_aligned_ransac_roi(
    roi_rgb: np.ndarray,
    roi_solid: np.ndarray,
    roi_raw: np.ndarray,
    r_entry: dict,
    mat: np.ndarray,
    card_med: np.ndarray,
    comp: dict,
    ratio_sil_intr: float,
    hole_frac_intr: float,
    prewarped: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Mengevaluasi cacat tri-modal setelah template diselaraskan oleh matriks pose RANSAC."""
    rh, rw = roi_rgb.shape[:2]
    if prewarped is not None:
        tmpl_warped, mask_warped = prewarped
    else:
        tmpl_warped = cv2.warpAffine(
            r_entry["rgb"], mat, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=tuple(float(v) for v in card_med)
        )
        mask_u8 = cv2.warpAffine(
            r_entry["mask"], mat, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        mask_warped = cv2.erode(mask_u8.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    tmpl_area = max(1, int(mask_warped.sum()))
    solid_bool = roi_solid > 0
    raw_bool = roi_raw > 0
    intact = mask_warped & raw_bool
    if int(intact.sum()) < 60:
        return float(max(ratio_sil_intr, hole_frac_intr / 0.055))

    roi_f = roi_rgb.astype(np.float32)
    gains, biases = _fit_shading_affine(roi_f[intact], tmpl_warped[intact])
    norm_tmpl = tmpl_warped * gains + biases
    # Toleransi geser jendela 3x3 (+/-1 px) meredam halo tepi akibat error sub-piksel pose RANSAC.
    padded_nt = np.pad(norm_tmpl, ((1, 1), (1, 1), (0, 0)), mode="edge")
    res_shift = np.full((rh, rw), 1e6, dtype=np.float32)
    for sy in range(3):
        for sx in range(3):
            d_s = np.linalg.norm(roi_f - padded_nt[sy : sy + rh, sx : sx + rw], axis=2)
            res_shift = np.minimum(res_shift, d_s)

    miss_sil = float(np.sum(mask_warped & ~solid_bool & (res_shift > 19.0))) / float(tmpl_area)

    solid_er = cv2.erode(roi_solid, np.ones((3, 3), np.uint8), iterations=1) > 0
    if comp.get("was_split", False):
        cnts_s, _ = cv2.findContours(roi_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_s:
            hull_mask = np.zeros_like(roi_solid)
            cv2.drawContours(hull_mask, [cv2.convexHull(np.vstack(cnts_s))], -1, 1, thickness=-1)
            hole_region = solid_er | (cv2.erode(hull_mask, np.ones((5, 5), np.uint8), iterations=1) > 0)
        else:
            hole_region = solid_er
    else:
        hole_region = solid_er

    diff_card = np.linalg.norm(roi_f - card_med, axis=2)
    hole_frac = float(np.sum(hole_region & (diff_card < 15.0) & (res_shift > 19.0))) / float(
        max(1, int(hole_region.sum()))
    )
    err_map = (intact & (res_shift > 18.0)).astype(np.uint8)
    blob_frac = float(cv2.erode(err_map, np.ones((3, 3), np.uint8), iterations=1).sum()) / float(int(intact.sum()))
    return float(max(miss_sil / 0.085, hole_frac / 0.055, blob_frac / 0.085))


def _score_cookie_keypoint_ransac(
    warped_face: np.ndarray,
    comp: dict,
    ref_cache: dict[int, dict],
    card_med: np.ndarray,
    method: str,
    contour_refs: dict[int, dict] | None = None,
) -> dict:
    """Menyelaraskan cookie memakai SIFT/ORB + RANSAC + refit Umeyama + verifikasi fotometrik."""
    x, y, bw, bh = comp["bbox"]
    pad = 14
    h_img, w_img = warped_face.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + bw + pad), min(h_img, y + bh + pad)

    roi_rgb = warped_face[y0:y1, x0:x1]
    roi_solid = comp["mask_solid"][y0:y1, x0:x1]
    roi_raw = comp["mask_raw"][y0:y1, x0:x1]
    rh, rw = roi_rgb.shape[:2]
    solid_bool = roi_solid > 0
    raw_bool = roi_raw > 0
    roi_f = roi_rgb.astype(np.float32)

    det, matcher, ratio_thr, reproj_thr, target_px = _build_feature_config(method)
    up = float(np.clip(target_px / max(1.0, float(max(bw, bh))), 2.0, 3.6))
    up_w = max(16, int(round(rw * up)))
    up_h = max(16, int(round(rh * up)))
    roi_up_gray = cv2.cvtColor(cv2.resize(roi_rgb, (up_w, up_h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)
    roi_up_mask = cv2.dilate(
        cv2.resize(roi_solid, (up_w, up_h), interpolation=cv2.INTER_NEAREST),
        np.ones((7, 7), np.uint8),
        iterations=1,
    ) * 255

    kps_obs, des_obs = det.detectAndCompute(roi_up_gray, roi_up_mask)
    obs_pts = np.float32([kp.pt for kp in kps_obs]) / up if kps_obs else np.zeros((0, 2), dtype=np.float32)

    hsv = cv2.cvtColor(warped_face, cv2.COLOR_RGB2HSV)
    hist_obs = cv2.calcHist([hsv], [0, 1], comp["mask_raw"], [12, 4], [0, 180, 0, 256]).ravel().astype(np.float32)
    hist_obs /= float(np.linalg.norm(hist_obs)) + 1e-8

    cands: list[tuple[float, int, np.ndarray, int, float, float, np.ndarray]] = []

    if des_obs is not None and len(obs_pts) >= 4:
        for did, r_entry in ref_cache.items():
            des_ref = r_entry["des"]
            if des_ref is None or len(r_entry["kps_pts"]) < 4:
                continue
            knn = matcher.knnMatch(des_ref, des_obs, k=2)
            good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio_thr * n.distance]
            if len(good) < 4:
                continue

            src_pts = np.float32([r_entry["kps_pts"][m.queryIdx] for m in good])
            dst_pts = np.float32([obs_pts[m.trainIdx] for m in good])
            M0, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=reproj_thr,
                maxIters=1200,
                confidence=0.995,
                refineIters=15,
            )
            if M0 is None or inliers is None:
                continue
            n_inl = int(inliers.sum())
            scale_est = float(np.hypot(M0[0, 0], M0[0, 1]))
            if n_inl < 4 or not (0.18 <= scale_est <= 0.90):
                continue

            # Refit Gold-Standard pada inlier ketat <= 2.2 px memakai solusi SVD Umeyama.
            proj0 = (src_pts @ M0[:, :2].T) + M0[:, 2]
            tight = np.linalg.norm(proj0 - dst_pts, axis=1) <= 2.2
            mat = M0
            if int(tight.sum()) >= 5:
                m_ref = _umeyama_2d_similarity(src_pts[tight], dst_pts[tight])
                if m_ref is not None:
                    mat = m_ref
                    n_inl = int(np.sum(np.linalg.norm((src_pts @ mat[:, :2].T) + mat[:, 2] - dst_pts, axis=1) <= reproj_thr))

            mw_u8 = cv2.warpAffine(r_entry["mask"], mat, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
            mw_bool = mw_u8 > 0
            iou = float(np.sum(mw_bool & solid_bool)) / float(max(1, int(np.sum(mw_bool | solid_bool))))
            if iou < 0.35:
                continue
            hsim = float(np.dot(hist_obs, r_entry["hist"]))
            pre_sc = float(n_inl) + 18.0 * iou + 12.0 * hsim + 0.08 * len(good)
            cands.append((pre_sc, did, mat, n_inl, iou, hsim, mw_u8))

    best_match: dict | None = None
    if cands:
        cands.sort(key=lambda item: item[0], reverse=True)
        best_hyp = -1e9
        erode_kernel = np.ones((3, 3), np.uint8)
        border_col = tuple(float(v) for v in card_med)
        for _, did, mat, n_inl, iou, hsim, mw_u8 in cands[:3]:
            r_entry = ref_cache[did]
            tw = cv2.warpAffine(r_entry["rgb"], mat, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=border_col)
            mw_er = cv2.erode(mw_u8.astype(np.uint8), erode_kernel, iterations=1) > 0
            intact = mw_er & raw_bool
            corr = 0.0
            p55 = 90.0
            if int(intact.sum()) >= 60:
                g_s, b_s = _fit_shading_affine(roi_f[intact], tw[intact])
                p55 = float(np.percentile(np.linalg.norm(roi_f[intact] - (tw[intact] * g_s + b_s), axis=1), 55))
                ov = roi_f[intact] - roi_f[intact].mean(axis=0, keepdims=True)
                tv = tw[intact] - tw[intact].mean(axis=0, keepdims=True)
                denom = float(np.sqrt(np.sum(ov * ov) * np.sum(tv * tv)))
                if denom > 1e-6:
                    corr = float(np.sum(ov * tv) / denom)
            hyp_sc = 0.60 * corr + 0.35 * iou + 0.25 * hsim + 0.22 * min(n_inl, 35) / 35.0 - 0.010 * min(p55, 120.0)
            if hyp_sc > best_hyp:
                best_hyp = hyp_sc
                best_match = {
                    "did": did,
                    "M": mat,
                    "inliers": n_inl,
                    "iou": iou,
                    "corr": corr,
                    "p55": p55,
                    "tw": tw,
                    "mw_er": mw_er,
                }

    ratio_sil_intr, hole_frac_intr = _intrinsic_defect_metrics(warped_face, comp, card_med)

    is_weak_pose = (
        best_match is None
        or (best_match["inliers"] < 5 and best_match["iou"] < 0.55)
        or (best_match["corr"] < 0.25 and best_match["p55"] > 42.0 and best_match["inliers"] < 14)
    )
    if is_weak_pose:
        if contour_refs is not None:
            fb = _score_contour_hsv_cookie(warped_face, comp, contour_refs, card_med)
            fb["inliers"] = 0 if best_match is None else int(best_match["inliers"])
            return fb

        best_did = (
            best_match["did"]
            if best_match is not None
            else max(ref_cache, key=lambda d: float(np.dot(hist_obs, ref_cache[d]["hist"])))
        )
        defect_ratio = max(ratio_sil_intr, hole_frac_intr / 0.055)
        return {
            "design_id": int(best_did),
            "inliers": 0 if best_match is None else int(best_match["inliers"]),
            "defect_ratio": float(defect_ratio),
            "is_defective": bool(defect_ratio > 1.0),
        }

    defect_ratio = _score_aligned_ransac_roi(
        roi_rgb,
        roi_solid,
        roi_raw,
        ref_cache[best_match["did"]],
        best_match["M"],
        card_med,
        comp,
        ratio_sil_intr,
        hole_frac_intr,
        prewarped=(best_match["tw"], best_match["mw_er"]),
    )
    return {
        "design_id": int(best_match["did"]),
        "inliers": int(best_match["inliers"]),
        "defect_ratio": float(defect_ratio),
        "is_defective": bool(defect_ratio > 1.0),
    }


def _predict_cube_with_face_scorer(
    rgb: np.ndarray,
    references: dict,
    face_scorer,
    leaderboard: bool = True,
) -> dict[str, int]:
    """Menjalankan kerangka rekonstruksi geometri 3D standar bersama scorer cookie pilihan."""
    _, dist_map = fit_background_model(rgb)
    hex_pts, _ = extract_cube_hexagon(rgb, dist_map=dist_map)
    v_center, parity = compute_vanishing_center(hex_pts, rgb=rgb)
    face_quads = segment_cube_faces(rgb, hex_pts, v_center, parity=parity)

    all_instances: list[tuple[str, dict]] = []
    out: dict[str, int] = {}

    for face_name in ("front", "right", "top"):
        warped, _ = rectify_face(rgb, face_quads[face_name], size=CANONICAL_FACE_SIZE)
        comps, card_med = segment_face_cookies(warped)
        scored = [face_scorer(warped, comp, card_med) for comp in comps]
        for s in scored:
            all_instances.append((face_name, s))

        out[f"{face_name}_total"] = int(len(comps))
        out[f"{face_name}_defective"] = int(sum(1 for s in scored if s["is_defective"]))
        if leaderboard:
            if not comps:
                d_cnt = 0
            else:
                d_cnt = min(len(comps), max(1, len(references)), len({s["design_id"] for s in scored}))
            out[f"{face_name}_design_count"] = int(d_cnt)

    defective_items = [(fn, s) for fn, s in all_instances if s["is_defective"]]
    if len(defective_items) > 2:
        defective_items.sort(key=lambda item: item[1]["defect_ratio"], reverse=True)
        for fn, s in defective_items[2:]:
            s["is_defective"] = False
            out[f"{fn}_defective"] = max(0, out[f"{fn}_defective"] - 1)

    return out


def predict_cube_sift_ransac(
    rgb: np.ndarray,
    references: dict,
    leaderboard: bool = True,
) -> dict[str, int]:
    """Pipeline pembanding 1: SIFT + Lowe Ratio + Verifikasi Hipotesis RANSAC."""
    ref_cache = _prepare_feature_refs(references, method="sift")
    contour_refs = _prepare_contour_hsv_refs(references)
    return _predict_cube_with_face_scorer(
        rgb,
        references,
        lambda wf, comp, cm: _score_cookie_keypoint_ransac(
            wf, comp, ref_cache, cm, method="sift", contour_refs=contour_refs
        ),
        leaderboard=leaderboard,
    )


def predict_cube_orb_ransac(
    rgb: np.ndarray,
    references: dict,
    leaderboard: bool = True,
) -> dict[str, int]:
    """Pipeline pembanding 2: ORB Binary Descriptor + Hamming Matcher + RANSAC."""
    ref_cache = _prepare_feature_refs(references, method="orb")
    contour_refs = _prepare_contour_hsv_refs(references)
    return _predict_cube_with_face_scorer(
        rgb,
        references,
        lambda wf, comp, cm: _score_cookie_keypoint_ransac(
            wf, comp, ref_cache, cm, method="orb", contour_refs=contour_refs
        ),
        leaderboard=leaderboard,
    )


def predict_cube_contour_hsv(
    rgb: np.ndarray,
    references: dict,
    leaderboard: bool = True,
) -> dict[str, int]:
    """Pipeline pembanding 3: Kontur Terorientasi & Histogram HSV 3-Zona V-Ternormalisasi."""
    ref_cache = _prepare_contour_hsv_refs(references)
    return _predict_cube_with_face_scorer(
        rgb,
        references,
        lambda wf, comp, cm: _score_contour_hsv_cookie(wf, comp, ref_cache, cm),
        leaderboard=leaderboard,
    )
