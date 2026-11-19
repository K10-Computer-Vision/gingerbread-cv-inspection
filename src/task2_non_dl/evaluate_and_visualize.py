"""Skrip evaluasi lengkap, pembuatan visualisasi diagnostik, dan inferensi hidden (Tugas 2 Non-DL).

Kami mengevaluasi pipeline Non-Deep Learning pada seluruh 100 citra kubus
dataset training (`train`, `val`, `test`, dan gabungan), menghasilkan tiga gambar
diagnostik ke `reports/figures/`, serta menjalankan inferensi tervalidasi pada
200 citra tersembunyi `hidden_students.zip` ke `submissions/`.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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

FACES = ("front", "right", "top")


def cube_columns(leaderboard: bool = False) -> list[str]:
    """Membentuk urutan kolom standar sesuai starter notebook Tugas 2."""
    fields = ["total", "defective"] + (["design_count"] if leaderboard else [])
    return ["image_id"] + [f"{face}_{field}" for field in fields for face in FACES]


def load_zip_references(zf: zipfile.ZipFile) -> dict[int, dict]:
    """Memuat kamus desain referensi dari arsip training dan melengkapinya dari template_all."""
    records = json.loads(zf.read("references/designs.json"))
    refs: dict[int, dict] = {}
    for item in records:
        with Image.open(io.BytesIO(zf.read("references/" + item["reference_file"]))) as im:
            rgba = np.asarray(im.convert("RGBA"))
        refs[int(item["design_id"])] = {
            "name": item["name"],
            "rgba": rgba,
            "mask": rgba[:, :, 3] > 0,
        }

    # Lengkapi seluruh 7 desain apabila folder template_all tersedia di repositori.
    tdir = next(
        (p for p in (Path("template_all"), Path(__file__).resolve().parents[2] / "template_all") if p.is_dir()),
        None,
    )
    if tdir is not None:
        designs_file = tdir / "designs.json"
        if designs_file.exists():
            t_records = json.loads(designs_file.read_text(encoding="utf-8"))
        else:
            t_records = [
                {"design_id": i, "name": p.stem, "reference_file": p.name}
                for i, p in enumerate(sorted(tdir.glob("*.png")), start=1)
            ]
        for item in t_records:
            did = int(item["design_id"])
            if did in refs:
                continue
            p_ref = tdir / item["reference_file"]
            if p_ref.exists():
                with Image.open(p_ref) as im:
                    rgba = np.asarray(im.convert("RGBA"))
                refs[did] = {
                    "name": item["name"],
                    "rgba": rgba,
                    "mask": rgba[:, :, 3] > 0,
                }
    return refs


def validate_cube_predictions(
    pred: pd.DataFrame, expected_ids: list[str], num_refs: int, leaderboard: bool = False
) -> pd.DataFrame:
    """Memvalidasi integritas kolom, tipe integer, batas cacat, dan desain unik."""
    cols = cube_columns(leaderboard)
    if list(pred.columns) != cols:
        raise ValueError(f"Nama/urutan kolom wajib: {cols}, dapat: {list(pred.columns)}")
    if pred.image_id.isna().any() or pred.image_id.duplicated().any():
        raise ValueError("image_id kosong atau duplikat.")
    if set(pred.image_id) != set(expected_ids) or len(pred) != len(expected_ids):
        raise ValueError("Daftar image_id tidak sesuai ekspektasi.")

    values = pred[cols[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prediksi mengandung NaN/inf.")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Semua hitungan wajib integer nonnegatif.")

    for f in FACES:
        if (pred[f"{f}_defective"] > pred[f"{f}_total"]).any():
            raise ValueError(f"Jumlah cacat melebihi jumlah total pada sisi {f}.")
        if leaderboard:
            d, t = pred[f"{f}_design_count"], pred[f"{f}_total"]
            if ((d > t) | (d > num_refs) | ((t > 0) & (d == 0))).any():
                raise ValueError(f"Konsistensi design_count gagal pada sisi {f}.")
    return pred


def evaluate_cube_dataframe(
    pred: pd.DataFrame, truth: pd.DataFrame, leaderboard: bool = False
) -> dict:
    """Menghitung metrik MAE dan Exact Match per sisi maupun per citra kubus."""
    fields = ["total", "defective"] + (["design_count"] if leaderboard else [])
    cols = cube_columns(leaderboard)[1:]
    indexed = truth.set_index("image_id")
    y = indexed.loc[pred.image_id, cols].to_numpy(dtype=float)
    p = pred[cols].to_numpy(dtype=float)

    errors = np.abs(p - y).reshape(len(p), len(fields), 3)
    res: dict[str, float | None] = {}
    for k, field in enumerate(fields):
        res[f"mae_{field}"] = float(errors[:, k, :].mean())
        for j, face in enumerate(FACES):
            res[f"mae_{face}_{field}"] = float(errors[:, k, j].mean())
    res["exact_match_face_pct"] = float((errors == 0).all(axis=1).mean() * 100.0)
    res["exact_match_image_pct"] = float((errors == 0).all(axis=(1, 2)).mean() * 100.0)
    bad_truth = y.reshape(len(p), len(fields), 3)[:, 1, :]
    pos = bad_truth > 0
    res["mae_defective_on_defective_faces"] = (
        float(errors[:, 1, :][pos].mean()) if pos.any() else None
    )
    return res


def run_training_benchmark(train_zip: Path, refs: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Menjalankan evaluasi menyeluruh pada seluruh 100 citra training berlabel."""
    records_pred: list[dict] = []
    records_truth: list[dict] = []
    splits_map: dict[str, str] = {}

    with zipfile.ZipFile(train_zip) as zf:
        anns = json.loads(zf.read("cube_annotations.json"))
        t0 = time.time()
        for ann in anns:
            img_id = Path(ann["image"]).stem
            with Image.open(io.BytesIO(zf.read(ann["image"]))) as im:
                rgb = np.asarray(im.convert("RGB"))
            pred = predict_cube_non_dl(rgb, refs, leaderboard=True)
            records_pred.append({"image_id": img_id, **pred})

            gt_row: dict[str, str | int] = {"image_id": img_id, "split": ann["split"]}
            for f in ann["faces"]:
                fn = f["name"]
                gt_row[f"{fn}_total"] = int(f["total_instances"])
                gt_row[f"{fn}_defective"] = int(len(f["defective_instance_ids"]))
                gt_row[f"{fn}_design_count"] = int(len(f["counts_by_design"]))
            records_truth.append(gt_row)
            splits_map[img_id] = ann["split"]
        elapsed = time.time() - t0

    df_pred = pd.DataFrame(records_pred, columns=cube_columns(True))
    df_truth = pd.DataFrame(records_truth)
    validate_cube_predictions(df_pred, df_truth.image_id.tolist(), len(refs), leaderboard=True)

    summary: dict[str, dict] = {
        "_meta": {
            "total_images": len(df_pred),
            "elapsed_seconds": round(elapsed, 2),
            "sec_per_image": round(elapsed / max(1, len(df_pred)), 3),
        }
    }
    for sp in ("train", "val", "test", "all"):
        if sp == "all":
            sub_p, sub_t = df_pred, df_truth
        else:
            mask = df_truth["split"] == sp
            ids = set(df_truth.loc[mask, "image_id"])
            sub_p = df_pred[df_pred["image_id"].isin(ids)]
            sub_t = df_truth[mask]
        m_main = evaluate_cube_dataframe(sub_p[cube_columns(False)], sub_t, leaderboard=False)
        m_lb = evaluate_cube_dataframe(sub_p, sub_t, leaderboard=True)
        summary[sp] = {
            "n": int(len(sub_p)),
            "main": m_main,
            "leaderboard": m_lb,
        }
    return df_pred, df_truth, summary


def render_pipeline_figure(train_zip: Path, refs: dict[int, dict], out_path: Path) -> None:
    """Membuat visualisasi tahapan geometri dan segmentasi cookie (reports/figures/)."""
    ref_cache = prepare_reference_cache(refs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(train_zip) as zf:
        anns = json.loads(zf.read("cube_annotations.json"))
        sample_indices = [0, 26]  # Contoh normal & contoh lubang terbelah + boundary/feature defect
        fig, axes = plt.subplots(len(sample_indices), 6, figsize=(19, 6.4))

        for row_idx, ann_idx in enumerate(sample_indices):
            ann = anns[ann_idx]
            with Image.open(io.BytesIO(zf.read(ann["image"]))) as im:
                rgb = np.asarray(im.convert("RGB"))
            _, dist_map = fit_background_model(rgb)
            hex_pts, _ = extract_cube_hexagon(rgb, dist_map=dist_map)
            v_center, parity = compute_vanishing_center(hex_pts, rgb=rgb)
            face_quads = segment_cube_faces(rgb, hex_pts, v_center, parity=parity)

            # Panel 0: Citra input
            axes[row_idx, 0].imshow(rgb)
            axes[row_idx, 0].set_title(f"{Path(ann['image']).stem}\nInput RGB")

            # Panel 1: Peta jarak latar + heksagon
            axes[row_idx, 1].imshow(dist_map, cmap="magma", vmin=0, vmax=65)
            hx_loop = np.vstack([hex_pts, hex_pts[0]])
            axes[row_idx, 1].plot(hx_loop[:, 0], hx_loop[:, 1], "c-", lw=2)
            axes[row_idx, 1].set_title("Background Residual\n& Sub-px Hexagon")

            # Panel 2: Kuadran rusuk proyeksi
            overlay = rgb.copy()
            colors = {"front": (30, 144, 255), "right": (46, 204, 113), "top": (241, 196, 15)}
            for fn in FACES:
                poly = np.round(face_quads[fn]).astype(np.int32)
                cv2.polylines(overlay, [poly], True, colors[fn], 4)
            vc_i = (int(round(v_center[0])), int(round(v_center[1])))
            cv2.circle(overlay, vc_i, 9, (231, 76, 60), -1)
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title(f"Vanishing Center\n(front=B, right=G, top=Y)")

            # Panel 3..5: Tiga sisi ter-rektifikasi + kotak deteksi
            for col_off, fn in enumerate(FACES):
                warped, _ = rectify_face(rgb, face_quads[fn], size=CANONICAL_FACE_SIZE)
                comps, card_med = segment_face_cookies(warped)
                vis = warped.copy()
                ref0 = next(iter(ref_cache.values()))
                peer = (
                    float(np.median([max(c["bbox"][2] / ref0["w"], c["bbox"][3] / ref0["h"]) for c in comps]))
                    if len(comps) >= 2
                    else None
                )
                n_def = 0
                for c in comps:
                    sc = align_and_score_cookie(warped, c, ref_cache, card_med, peer_scale=peer)
                    if sc["is_defective"]:
                        n_def += 1
                    col = (220, 38, 38) if sc["is_defective"] else (22, 163, 74)
                    x, y, bw, bh = c["bbox"]
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 3)
                    label = f"r={sc['defect_ratio']:.2f}"
                    cv2.putText(vis, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                axes[row_idx, 3 + col_off].imshow(vis)
                axes[row_idx, 3 + col_off].set_title(f"{fn} ({len(comps)} tot, {n_def} def)")

            for c_ax in axes[row_idx]:
                c_ax.axis("off")

    plt.suptitle("Pipeline Tugas 2 Non-Deep Learning: Geometri 3D & Inspeksi per Sisi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_defect_modes_figure(train_zip: Path, refs: dict[int, dict], out_path: Path) -> None:
    """Memvisualisasikan contoh nyata tiga mode cacat vs cookie sempurna."""
    ref_cache = prepare_reference_cache(refs)
    ref0 = next(iter(ref_cache.values()))

    candidates: dict[str, dict] = {}
    with zipfile.ZipFile(train_zip) as zf:
        anns = json.loads(zf.read("cube_annotations.json"))
        for ann in anns[:35]:
            with Image.open(io.BytesIO(zf.read(ann["image"]))) as im:
                rgb = np.asarray(im.convert("RGB"))
            _, dist_map = fit_background_model(rgb)
            hex_pts, _ = extract_cube_hexagon(rgb, dist_map=dist_map)
            v_center, parity = compute_vanishing_center(hex_pts, rgb=rgb)
            face_quads = segment_cube_faces(rgb, hex_pts, v_center, parity=parity)

            for fn in FACES:
                warped, _ = rectify_face(rgb, face_quads[fn], size=CANONICAL_FACE_SIZE)
                comps, card_med = segment_face_cookies(warped)
                peer = (
                    float(np.median([max(c["bbox"][2] / ref0["w"], c["bbox"][3] / ref0["h"]) for c in comps]))
                    if len(comps) >= 2
                    else None
                )
                for c in comps:
                    sc = align_and_score_cookie(warped, c, ref_cache, card_med, peer_scale=peer)
                    x, y, bw, bh = c["bbox"]
                    x0, y0 = max(0, x - 12), max(0, y - 12)
                    x1, y1 = min(warped.shape[1], x + bw + 12), min(warped.shape[0], y + bh + 12)
                    crop_rgb = warped[y0:y1, x0:x1]
                    crop_mask = (c["mask_solid"][y0:y1, x0:x1] * 2 + c["mask_raw"][y0:y1, x0:x1])

                    if (not sc["is_defective"]) and "perfect" not in candidates and sc["defect_ratio"] < 0.45:
                        candidates["perfect"] = {"rgb": crop_rgb, "mask": crop_mask, "sc": sc, "tag": f"{Path(ann['image']).stem} ({fn})"}
                    elif sc["miss_sil"] > 0.14 and "boundary_cut" not in candidates:
                        candidates["boundary_cut"] = {"rgb": crop_rgb, "mask": crop_mask, "sc": sc, "tag": f"{Path(ann['image']).stem} ({fn})"}
                    elif sc["hole_frac"] > 0.08 and "interior_hole" not in candidates:
                        candidates["interior_hole"] = {"rgb": crop_rgb, "mask": crop_mask, "sc": sc, "tag": f"{Path(ann['image']).stem} ({fn})"}
                    elif sc["blob_frac"] > 0.15 and "missing_feature" not in candidates:
                        candidates["missing_feature"] = {"rgb": crop_rgb, "mask": crop_mask, "sc": sc, "tag": f"{Path(ann['image']).stem} ({fn})"}

            if len(candidates) == 4:
                break

    order = ["perfect", "boundary_cut", "interior_hole", "missing_feature"]
    titles = {
        "perfect": "Sempurna (Normal)",
        "boundary_cut": "Cacat Tepi (boundary_cut)",
        "interior_hole": "Cacat Lubang (interior_hole)",
        "missing_feature": "Cacat Hiasan (missing_feature)",
    }

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for col, key in enumerate(order):
        item = candidates.get(key)
        if item is None:
            continue
        sc = item["sc"]
        axes[0, col].imshow(item["rgb"])
        axes[0, col].set_title(f"{titles[key]}\n{item['tag']}", fontsize=10)
        axes[1, col].imshow(item["mask"], cmap="viridis")
        axes[1, col].set_title(
            f"miss={sc['miss_sil']:.3f} | hole={sc['hole_frac']:.3f}\nblob={sc['blob_frac']:.3f} | ratio={sc['defect_ratio']:.2f}",
            fontsize=9.5,
        )
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    plt.suptitle("Evaluasi Tri-Modal Fitur Non-DL per Kategori Kondisi Cookie", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_hidden_inference(
    hidden_zip: Path,
    refs: dict[int, dict],
    out_lb_csv: Path,
    out_main_csv: Path,
    fig_path: Path,
) -> pd.DataFrame:
    """Menjalankan inferensi 200 citra hidden_students.zip dan memvalidasi format submisi."""
    with zipfile.ZipFile(hidden_zip) as zf:
        sample_df = pd.read_csv(io.BytesIO(zf.read("sample_submission_task2.csv")), dtype={"image_id": str})
        test_df = pd.read_csv(io.BytesIO(zf.read("test_cube.csv")), dtype={"image_id": str})
        path_map = dict(zip(test_df["image_id"], test_df["image"]))

        rows: list[dict] = []
        vis_samples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict, dict]] = []
        vis_indices = {0, 2, 15, 35, 70, 120}

        t0 = time.time()
        for idx, img_id in enumerate(sample_df["image_id"]):
            img_path = path_map[img_id]
            with Image.open(io.BytesIO(zf.read(img_path))) as im:
                rgb = np.asarray(im.convert("RGB"))
            pred = predict_cube_non_dl(rgb, refs, leaderboard=True)
            rows.append({"image_id": img_id, **pred})

            if idx in vis_indices:
                _, dmap = fit_background_model(rgb)
                hx, _ = extract_cube_hexagon(rgb, dist_map=dmap)
                vc, par = compute_vanishing_center(hx, rgb=rgb)
                fquads = segment_cube_faces(rgb, hx, vc, parity=par)
                vis_samples.append((img_id, rgb, hx, vc, fquads, pred))
        elapsed = time.time() - t0

    df_lb = pd.DataFrame(rows, columns=cube_columns(True))
    validate_cube_predictions(df_lb, sample_df["image_id"].tolist(), len(refs), leaderboard=True)

    df_main = df_lb[cube_columns(False)].copy()
    validate_cube_predictions(df_main, sample_df["image_id"].tolist(), len(refs), leaderboard=False)

    out_lb_csv.parent.mkdir(parents=True, exist_ok=True)
    df_lb.to_csv(out_lb_csv, index=False)
    df_main.to_csv(out_main_csv, index=False)

    # Simpan grid inspeksi visual sampel hidden untuk pemeriksaan kualitas geometri.
    if vis_samples:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        colors = {"front": (30, 144, 255), "right": (46, 204, 113), "top": (241, 196, 15)}
        for ax, (img_id, rgb, hx, vc, fquads, pred) in zip(axes.ravel(), vis_samples):
            vis = rgb.copy()
            for fn in FACES:
                cv2.polylines(vis, [np.round(fquads[fn]).astype(np.int32)], True, colors[fn], 4)
            cv2.circle(vis, (int(round(vc[0])), int(round(vc[1]))), 8, (231, 76, 60), -1)
            ax.imshow(vis)
            ax.set_title(
                f"{img_id[:16]}\n"
                f"F:{pred['front_total']}/{pred['front_defective']}  "
                f"R:{pred['right_total']}/{pred['right_defective']}  "
                f"T:{pred['top_total']}/{pred['top_defective']}",
                fontsize=10,
            )
            ax.axis("off")
        plt.suptitle(
            f"Sanity Check Dataset Tersembunyi (200 citra selesai dalam {elapsed:.1f}s)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(
        f"[Hidden] Validated {len(df_lb)} rows in {elapsed:.2f}s "
        f"({elapsed / len(df_lb):.3f}s/img) -> saved {out_lb_csv.name} & {out_main_csv.name}"
    )
    return df_lb


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi dan visualisasi Tugas 2 Non-Deep Learning")
    parser.add_argument(
        "--train-zip",
        type=Path,
        default=Path("data/gingerbread_seed2026_students_training.zip"),
    )
    parser.add_argument(
        "--hidden-zip",
        type=Path,
        default=Path("data/hidden_students.zip"),
    )
    parser.add_argument("--skip-hidden", action="store_true")
    args = parser.parse_args()

    with zipfile.ZipFile(args.train_zip) as zf:
        refs = load_zip_references(zf)

    print(f"Loaded {len(refs)} reference design(s): {[v['name'] for v in refs.values()]}")
    _, _, summary = run_training_benchmark(args.train_zip, refs)

    meta = summary["_meta"]
    print(
        f"Evaluated {meta['total_images']} training cubes in {meta['elapsed_seconds']}s "
        f"({meta['sec_per_image']}s/img):"
    )
    for sp in ("train", "val", "test", "all"):
        item = summary[sp]
        mm = item["main"]
        ml = item["leaderboard"]
        print(
            f"  [{sp:5s} n={item['n']:3d}] "
            f"MAE_total={mm['mae_total']:.4f}, MAE_def={mm['mae_defective']:.4f}, MAE_des={ml['mae_design_count']:.4f} | "
            f"EM_Face={mm['exact_match_face_pct']:.1f}%, EM_Main={mm['exact_match_image_pct']:.1f}%, "
            f"EM_LB={ml['exact_match_image_pct']:.1f}%"
        )

    render_pipeline_figure(
        args.train_zip, refs, Path("reports/figures/task2_non_dl_pipeline_stages.png")
    )
    render_defect_modes_figure(
        args.train_zip, refs, Path("reports/figures/task2_non_dl_defect_types.png")
    )
    print("Saved diagnostic figures to reports/figures/.")

    if not args.skip_hidden and args.hidden_zip.exists():
        run_hidden_inference(
            args.hidden_zip,
            refs,
            Path("submissions/task2_ncc_lb.csv"),
            Path("submissions/task2_ncc_main.csv"),
            Path("reports/figures/task2_non_dl_hidden_samples.png"),
        )


if __name__ == "__main__":
    main()
