"""Modul rekonstruksi geometri 3D kubus secara analitis (non-deep learning).

Pada modul ini, kami memisahkan citra kemasan kubus dari latar belakang,
mengekstraksi kontur heksagon luar pada presisi sub-piksel, menentukan titik
pertemuan rusuk dalam melalui proyeksi vanishing point, dan menormalisasi
setiap sisi kubus (front, right, top) ke bidang persegi kanonik memakai
transformasi homography.
"""

from __future__ import annotations

import cv2
import numpy as np

CANONICAL_FACE_SIZE = 384


def fit_background_model(
    rgb: np.ndarray, border: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Memodelkan iluminasi latar belakang dan menghitung peta jarak warna.

    Kami menyampel piksel tepi luar citra untuk meregresikan model latar
    kuadratik-radial `[1, u, v, u^2 + v^2]` per kanal RGB dengan penyaringan
    outlier dua tahap agar sudut kubus tidak mengganggu estimasi latar.
    """
    h, w = rgb.shape[:2]
    img_f = rgb.astype(np.float32)
    scale = float(max(h, w))

    xs = (np.arange(w, dtype=np.float32) - w * 0.5) / scale
    ys = (np.arange(h, dtype=np.float32) - h * 0.5) / scale
    uu, vv = np.meshgrid(xs, ys)
    rr = uu * uu + vv * vv

    b = max(4, min(border, min(h, w) // 8))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True

    feats = np.stack([np.ones_like(uu), uu, vv, rr], axis=-1)
    x_border = feats[border_mask]
    y_border = img_f[border_mask]

    coef, _, _, _ = np.linalg.lstsq(x_border, y_border, rcond=None)
    resid = np.linalg.norm(y_border - x_border @ coef, axis=1)

    # Kami melakukan fit ulang pada inlier jika ada objek/bayangan menyentuh tepi.
    if np.sum(resid > 15.0) > 16:
        thresh = max(10.0, float(np.percentile(resid, 85)))
        inlier = resid <= thresh
        if np.sum(inlier) >= 32:
            coef, _, _, _ = np.linalg.lstsq(
                x_border[inlier], y_border[inlier], rcond=None
            )

    bg_model = (feats.reshape(-1, 4) @ coef).reshape(h, w, 3).astype(np.float32)
    diff_smooth = cv2.GaussianBlur(img_f - bg_model, (5, 5), 1.2)
    dist_map = np.linalg.norm(diff_smooth, axis=2).astype(np.float32)
    return bg_model, dist_map


def _min_edge_length(poly: np.ndarray) -> float:
    """Menghitung panjang sisi terpendek dari sebuah poligon tertutup."""
    diffs = np.roll(poly, -1, axis=0) - poly
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def _collapse_short_edges(pts: np.ndarray, min_len: float = 24.0) -> np.ndarray:
    """Menggabungkan pasangan sudut yang jaraknya lebih dekat dari min_len piksel."""
    cur = pts.astype(np.float32).copy()
    while len(cur) > 6:
        diffs = np.roll(cur, -1, axis=0) - cur
        lens = np.linalg.norm(diffs, axis=1)
        min_idx = int(np.argmin(lens))
        if float(lens[min_idx]) >= min_len:
            break
        nxt = (min_idx + 1) % len(cur)
        cur[min_idx] = 0.5 * (cur[min_idx] + cur[nxt])
        cur = np.delete(cur, nxt, axis=0)
    return cur


def _order_polygon_cw(pts: np.ndarray) -> np.ndarray:
    """Mengurutkan titik poligon searah jarum jam pada koordinat citra."""
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    # Pada koordinat citra (y ke bawah), sudut membesar searah jarum jam.
    order = np.argsort(angles)
    return pts[order].astype(np.float32)


def _reduce_polygon_to_hex(pts: np.ndarray) -> np.ndarray:
    """Merampingkan poligon > 6 titik hingga tepat menyisakan 6 sudut heksagon."""
    cur = _collapse_short_edges(pts.astype(np.float32), min_len=28.0)
    while len(cur) > 6:
        n = len(cur)
        best_idx = 0
        best_score = float("inf")
        for i in range(n):
            p_prev = cur[(i - 1) % n]
            p_curr = cur[i]
            p_next = cur[(i + 1) % n]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                best_idx = i
                break
            cos_turn = float(np.dot(v1, v2) / (n1 * n2))
            # Semakin lurus (cos_turn -> 1) dan pendek segmen, semakin aman digabung.
            deviation = max(1e-4, 1.0 - cos_turn)
            score = deviation * min(n1, n2)
            if score < best_score:
                best_score = score
                best_idx = i
        cur = np.delete(cur, best_idx, axis=0)
    return cur


def _fit_line_huber(points: np.ndarray) -> np.ndarray:
    """Menyesuaikan garis homogen [a, b, c] dengan norm (a, b)=1 memakai fitLine."""
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_HUBER, 0.0, 0.01, 0.01
    ).ravel()
    a = -float(vy)
    b = float(vx)
    norm = float(np.hypot(a, b)) + 1e-12
    a /= norm
    b /= norm
    c = -(a * float(x0) + b * float(y0))
    return np.array([a, b, c], dtype=np.float64)


def _intersect_2d_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """Menghitung perpotongan 2D dari dua garis homogen [a, b, c]."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-8:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def extract_cube_hexagon(
    rgb: np.ndarray,
    dist_map: np.ndarray | None = None,
    border: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Mengekstraksi 6 sudut terluar kubus pada presisi sub-piksel dan mask foreground.

    Kami menyegmentasi kubus menggunakan jarak warna latar, mengekstraksi
    convex hull, menyederhanakan ke 6 sudut kasar, lalu memperhalus setiap sisi
    dengan cv2.fitLine berbasis jarak Huber untuk memotong ulang sudut heksagon.
    """
    if dist_map is None:
        _, dist_map = fit_background_model(rgb, border=border)

    h, w = rgb.shape[:2]
    b = max(4, min(border, min(h, w) // 8))
    bg_border = np.concatenate(
        [
            dist_map[:b, :].ravel(),
            dist_map[-b:, :].ravel(),
            dist_map[:, :b].ravel(),
            dist_map[:, -b:].ravel(),
        ]
    )
    # Kami memakai ambang adaptif berbasis persentil tepi agar sisi ber-kontras tipis tetap utuh.
    fg_thresh = max(4.2, min(12.0, float(np.percentile(bg_border, 98.5)) + 1.8))
    fg = (dist_map > fg_thresh).astype(np.uint8)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    if num_labels <= 1:
        raise ValueError("Gagal menemukan komponen foreground kubus pada citra.")

    largest_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg_clean = (labels == largest_lbl).astype(np.uint8)
    for lbl in range(1, num_labels):
        if lbl != largest_lbl and stats[lbl, cv2.CC_STAT_AREA] >= 2500:
            cx_l, cy_l = centroids[lbl]
            if 0.05 * w <= cx_l <= 0.95 * w and 0.05 * h <= cy_l <= 0.95 * h:
                fg_clean[labels == lbl] = 1

    contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Kontur terluar kubus kosong.")

    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_mask, [hull], -1, 1, thickness=-1)
    perim = float(cv2.arcLength(hull, True))

    coarse_hex = None
    best_over6 = None
    fallback_hex = None
    for frac in np.linspace(0.004, 0.060, 113):
        approx = cv2.approxPolyDP(hull, frac * perim, True).reshape(-1, 2)
        if len(approx) == 6:
            if _min_edge_length(approx.astype(np.float32)) >= 35.0:
                coarse_hex = approx
                break
            if fallback_hex is None:
                fallback_hex = approx
        elif len(approx) > 6:
            best_over6 = approx

    if coarse_hex is None:
        if best_over6 is not None:
            reduced = _reduce_polygon_to_hex(best_over6)
            if len(reduced) == 6 and _min_edge_length(reduced) >= 30.0:
                coarse_hex = reduced
        if coarse_hex is None:
            coarse_hex = (
                fallback_hex
                if fallback_hex is not None
                else _reduce_polygon_to_hex(hull.reshape(-1, 2))
            )

    coarse_hex = _order_polygon_cw(coarse_hex)
    cnt_pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)

    # Kami memilih titik kontur di tengah setiap segmen tepi untuk cv2.fitLine.
    edge_lines: list[np.ndarray] = []
    for i in range(6):
        p1 = coarse_hex[i].astype(np.float64)
        p2 = coarse_hex[(i + 1) % 6].astype(np.float64)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            edge_lines.append(np.array([0.0, 1.0, -p1[1]], dtype=np.float64))
            continue
        u = vec / length
        n_vec = np.array([-u[1], u[0]], dtype=np.float64)
        rel = cnt_pts - p1
        proj = (rel @ u) / length
        perp = np.abs(rel @ n_vec)
        sel = (proj >= 0.10) & (proj <= 0.90) & (perp <= 6.0)
        if int(np.sum(sel)) >= 8:
            line = _fit_line_huber(cnt_pts[sel])
        else:
            line = _fit_line_huber(np.vstack([p1, p2]))
        edge_lines.append(line)

    refined = np.zeros((6, 2), dtype=np.float32)
    for i in range(6):
        inter = _intersect_2d_lines(edge_lines[(i - 1) % 6], edge_lines[i])
        if inter is None or np.linalg.norm(inter - coarse_hex[i]) > 18.0:
            refined[i] = coarse_hex[i]
        else:
            refined[i] = inter

    # Mulai rotasi dari sudut paling atas (y minimum) searah jarum jam.
    refined = _order_polygon_cw(refined)
    top_idx = int(np.argmin(refined[:, 1]))
    refined = np.roll(refined, -top_idx, axis=0)
    return refined.astype(np.float32), filled_mask


def _solve_center_for_parity(
    hex_pts: np.ndarray, parity: int, img_shape: tuple[int, int]
) -> np.ndarray:
    """Menyelesaikan titik temu rusuk dalam V_center via vanishing point pada paritas tertentu.

    Pada heksagon proyeksi kubus, tiga rusuk dalam bertemu di V_center menuju
    simpul berindeks sama-paritas ({1, 3, 5} atau {0, 2, 4}). Dua sisi luar yang
    sejajar secara 3D bertemu di vanishing point homogen untuk mengarahkan tiap rusuk.
    """
    h, w = img_shape
    scale = float(max(h, w))
    cx, cy = w * 0.5, h * 0.5
    pts_norm = np.column_stack(
        [(hex_pts[:, 0] - cx) / scale, (hex_pts[:, 1] - cy) / scale, np.ones(6)]
    )

    rib_indices = [i for i in range(6) if (i % 2) == (parity % 2)]
    rib_lines: list[np.ndarray] = []
    for k in rib_indices:
        # Sisi luar yang sejajar 3D dengan rusuk k adalah (k-2 -> k-1) dan (k+2 -> k+1).
        l_left = np.cross(pts_norm[(k - 2) % 6], pts_norm[(k - 1) % 6])
        l_right = np.cross(pts_norm[(k + 2) % 6], pts_norm[(k + 1) % 6])
        vp = np.cross(l_left, l_right)
        if float(np.linalg.norm(vp)) < 1e-12:
            # Cadangan arah sejajar langsung bila kedua sisi benar-benar kolinear.
            d = pts_norm[(k - 1) % 6] - pts_norm[(k - 2) % 6]
            vp = np.array([d[0], d[1], 0.0], dtype=np.float64)
        rib = np.cross(pts_norm[k], vp)
        nr = float(np.hypot(rib[0], rib[1]))
        if nr > 1e-12:
            rib_lines.append(rib / nr)

    if len(rib_lines) < 2:
        return hex_pts[rib_indices].mean(axis=0).astype(np.float32)

    mat = np.array(rib_lines, dtype=np.float64)
    A = mat[:, :2]
    b = -mat[:, 2]
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([sol[0] * scale + cx, sol[1] * scale + cy], dtype=np.float32)


def _point_in_triangle(p: np.ndarray, tri: np.ndarray) -> bool:
    """Memeriksa apakah titik 2D p berada di dalam segitiga tri (3x2)."""
    v0 = tri[2] - tri[0]
    v1 = tri[1] - tri[0]
    v2 = p - tri[0]
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-9:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u > 0.02) and (v > 0.02) and (u + v < 0.98)


def _eval_partition_variance(rgb: np.ndarray, quads: list[np.ndarray]) -> float:
    """Menghitung median variansi warna intra-kuadran untuk disambiguasi paritas."""
    h, w = rgb.shape[:2]
    img_f = rgb.astype(np.float32)
    variances: list[float] = []
    for q in quads:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(q).astype(np.int32)], 1)
        # Kami memakai jarak L1 dari median warna kuadran agar tahan terhadap cookies.
        pix = img_f[mask == 1]
        if len(pix) < 100:
            return 1e9
        med = np.median(pix, axis=0)
        dev = np.linalg.norm(pix - med, axis=1)
        mad = float(np.percentile(dev, 75))
        variances.append(mad)
    return float(np.mean(variances))


def compute_vanishing_center(
    hex_pts: np.ndarray, rgb: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Menentukan koordinat V_center serta paritas simpul rusuk dalam (1 atau 0).

    Secara baku untuk heksagon yang dimulai dari simpul paling atas, simpul rusuk
    berada pada paritas ganjil {1, 3, 5}. Jika kubus mengalami rotasi ekstrem,
    kami mengevaluasi paritas genap {0, 2, 4} dan hanya beralih jika V_center
    genap valid di dalam segitiga simpul serta variasi warna antar kuadran jauh
    lebih seragam (< 0.65 kali paritas ganjil).
    """
    h, w = rgb.shape[:2] if rgb is not None else (768, 768)
    v_odd = _solve_center_for_parity(hex_pts, 1, (h, w))
    if rgb is None:
        return v_odd, 1

    tri_odd = hex_pts[[1, 3, 5]]
    odd_valid = _point_in_triangle(v_odd, tri_odd)

    v_even = _solve_center_for_parity(hex_pts, 0, (h, w))
    tri_even = hex_pts[[0, 2, 4]]
    even_valid = _point_in_triangle(v_even, tri_even)

    if not even_valid:
        return v_odd, 1
    if not odd_valid and even_valid:
        return v_even, 0

    quads_odd = [
        np.array([v_odd, hex_pts[5], hex_pts[0], hex_pts[1]], dtype=np.float32),
        np.array([v_odd, hex_pts[1], hex_pts[2], hex_pts[3]], dtype=np.float32),
        np.array([v_odd, hex_pts[3], hex_pts[4], hex_pts[5]], dtype=np.float32),
    ]
    quads_even = [
        np.array([v_even, hex_pts[0], hex_pts[1], hex_pts[2]], dtype=np.float32),
        np.array([v_even, hex_pts[2], hex_pts[3], hex_pts[4]], dtype=np.float32),
        np.array([v_even, hex_pts[4], hex_pts[5], hex_pts[0]], dtype=np.float32),
    ]

    var_odd = _eval_partition_variance(rgb, quads_odd)
    var_even = _eval_partition_variance(rgb, quads_even)
    if var_even < 0.65 * var_odd:
        return v_even, 0
    return v_odd, 1


def segment_cube_faces(
    rgb: np.ndarray,
    hex_pts: np.ndarray,
    v_center: np.ndarray,
    parity: int = 1,
) -> dict[str, np.ndarray]:
    """Membagi heksagon kubus menjadi tiga kuadran berlabel 'top', 'right', 'front'.

    Kami membentuk ketiga segmen kuadran searah jarum jam mengelilingi V_center.
    Sisi paling atas atau paling terang ditetapkan sebagai 'top', lalu kedua
    sisi berikutnya sepanjang arah jarum jam otomatis menjadi 'right' dan 'front'.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k_list = [i for i in range(6) if (i % 2) == (parity % 2)]
    # Setiap kuadran i: [V_center, hex[k_i], hex[(k_i + 1)%6], hex[(k_i + 2)%6]]
    quads: list[np.ndarray] = []
    centroids_y: list[float] = []
    medians_luma: list[float] = []

    for idx in range(3):
        k0 = k_list[idx]
        k1 = (k0 + 1) % 6
        k2 = (k0 + 2) % 6
        q = np.array(
            [v_center, hex_pts[k0], hex_pts[k1], hex_pts[k2]], dtype=np.float32
        )
        quads.append(q)
        centroids_y.append(float(q[:, 1].mean()))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(q).astype(np.int32)], 1)
        pix = gray[mask == 1]
        medians_luma.append(float(np.percentile(pix, 75)) if len(pix) > 0 else 0.0)

    # Sisi 'top' secara konsisten lebih tinggi (y kecil) dan lebih terang (p75 tinggi).
    y_order = np.argsort(centroids_y)
    # Jika selisih tinggi dua sisi atas < 18 piksel (rotasi roll), pilih yang lebih terang.
    if abs(centroids_y[int(y_order[0])] - centroids_y[int(y_order[1])]) < 18.0:
        top_slot = (
            int(y_order[0])
            if medians_luma[int(y_order[0])] >= medians_luma[int(y_order[1])]
            else int(y_order[1])
        )
    else:
        top_slot = int(y_order[0])

    # Urutan sudut kuadran sudah searah jarum jam: top -> right -> front.
    right_slot = (top_slot + 1) % 3
    front_slot = (top_slot + 2) % 3

    return {
        "top": quads[top_slot],
        "right": quads[right_slot],
        "front": quads[front_slot],
    }


def _orient_quad_upright(quad: np.ndarray) -> np.ndarray:
    """Mengatur urutan sudut kuadran ke [top_left, top_right, bottom_right, bottom_left]."""
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    # Urutkan searah jarum jam lalu mulai dari kuadran kiri-atas (-3*pi/4).
    cw = quad[np.argsort(angles)]
    cw_angles = np.sort(angles)
    tl_idx = int(np.argmin(np.abs(((cw_angles + 0.75 * np.pi + np.pi) % (2 * np.pi)) - np.pi)))
    return np.roll(cw, -tl_idx, axis=0).astype(np.float32)


def rectify_face(
    rgb: np.ndarray,
    quad: np.ndarray,
    size: int = CANONICAL_FACE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Meluruskan sisi kubus ke citra persegi size x size menggunakan homography.

    Kami menyusun 4 sudut tujuan searah jarum jam mulai dari kiri-atas sehingga
    orientasi dasar setiap sisi siap diproses oleh modul segmentasi cookie.
    """
    src = _orient_quad_upright(quad)
    dst = np.array(
        [[0.0, 0.0], [size - 1.0, 0.0], [size - 1.0, size - 1.0], [0.0, size - 1.0]],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgb, h_mat, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return warped, h_mat
