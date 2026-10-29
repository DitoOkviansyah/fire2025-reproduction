"""Reproduksi Fire 2025. Tidak memasang dependensi atau mengunduh bobot/data.

Jalankan dari notebook pendamping pada environment HPC yang sudah tersedia.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "False"  # Ultralytics must never install missing packages.

PAPER = "https://www.mdpi.com/2571-6255/8/1/26"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_MODELS = ("yolov9t", "yolov10n", "yolo11n")
ALL_MODELS = SMALL_MODELS + ("yolo11x",)
PAPER_SCORES = {
    "t4_yolo11n_50": (0.766005, 0.709466, 0.7772, 0.467005),
    "t4_yolo11n_100": (0.803112, 0.758146, 0.824964, 0.514914),
    "t4_yolov10n_50": (0.740179, 0.690027, 0.755201, 0.453826),
    "t4_yolov10n_100": (0.797005, 0.749882, 0.812536, 0.512366),
    "t4_yolov9t_50": (0.749910, 0.711167, 0.773553, 0.469511),
    "t4_yolov9t_100": (0.799071, 0.737427, 0.808770, 0.505313),
    "t6_yolo11n_100": (0.845419, 0.801229, 0.859837, 0.558822),
    "t6_yolov10n_100": (0.835622, 0.763534, 0.837255, 0.548967),
    "t6_yolov9t_100": (0.817222, 0.759236, 0.828777, 0.528051),
    "t7_smoke_yolo11n_300": (0.870, 0.802, 0.866, 0.594),
    "t7_smoke_yolo11x_300": (0.904, 0.816, 0.889, 0.709),
    "t7_dfire_yolo11n_100": (0.772, 0.718, 0.780, 0.458),
    "t7_dfire_yolo11x_100": (0.795, 0.743, 0.780, 0.470),
}
TABLE5 = {"optimizer": "SGD", "batch": 64, "lr0": 0.01,
          "warmup_epochs": 3.0, "warmup_bias_lr": 0.1,
          "momentum": 0.8, "weight_decay": 0.0005, "amp": True}


def build_experiments(table7_dfire_overrides=None):
    """13 konfigurasi penulis; nilai asumsi dan rujukan selalu disertakan."""
    jobs = []
    for model in SMALL_MODELS:
        for epochs in (50, 100):
            jobs.append({"id": f"t4_{model}_{epochs}", "table": "4", "dataset": "smoke_fire",
                         "model": model, "epochs": epochs,
                         "train_args": {"imgsz": 640, "batch": 16, "optimizer": "Adam"},
                         "assumptions": ["Penempatan Tabel 4 pada Smoke and Fire diasumsikan dari konteks paper.",
                                         "imgsz=640 berdasarkan deskripsi model; Tabel 3 tidak merincinya.",
                                         "50 dan 100 epoch dijalankan terpisah; paper tidak memastikan metode ini.",
                                         "Argumen lain mengikuti default Ultralytics terpasang."]})
    for model in SMALL_MODELS:
        jobs.append({"id": f"t6_{model}_100", "table": "6", "dataset": "smoke_fire",
                     "model": model, "epochs": 100, "train_args": {**TABLE5, "imgsz": 640},
                     "assumptions": ["Penempatan Tabel 6 pada Smoke and Fire mengikuti kesimpulan paper.",
                                     "Argumen yang tidak dilaporkan memakai default Ultralytics."]})
    for model in ("yolo11n", "yolo11x"):
        jobs.append({"id": f"t7_smoke_{model}_300", "table": "7", "dataset": "smoke_fire",
                     "model": model, "epochs": 300, "train_args": {**TABLE5, "imgsz": 416},
                     "assumptions": ["Run 300 epoch memakai Tabel 5 dan imgsz=416 seperti teks paper."]})
    # Paper tidak menguraikan seluruh pengaturan run D-Fire 100 epoch.
    assumed = {**TABLE5, "imgsz": 640}
    assumed.update(table7_dfire_overrides or {})
    for model in ("yolo11n", "yolo11x"):
        jobs.append({"id": f"t7_dfire_{model}_100", "table": "7", "dataset": "dfire",
                     "model": model, "epochs": 100, "train_args": dict(assumed),
                     "assumptions": ["Setting D-Fire 100 epoch tidak sepenuhnya dijelaskan;"
                                     " memakai Tabel 5 dan imgsz=640 sebagai asumsi yang dapat diubah."]})
    for job in jobs:
        job["paper_metrics"] = dict(zip(("precision", "recall", "map50", "map50_95"),
                                        PAPER_SCORES[job["id"]]))
    return jobs


def environment_info(device=0):
    """Periksa environment tanpa instalasi, tanpa CUDA allocation dan tanpa unduhan."""
    info = {"python": sys.version, "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": {}}
    for pkg in ("torch", "ultralytics", "PyYAML", "numpy", "matplotlib"):
        try:
            info["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][pkg] = None
    if info["packages"]["torch"]:
        torch = importlib.import_module("torch")
        info["torch_cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            if int(device) >= torch.cuda.device_count():
                raise ValueError(f"Device {device} tidak tersedia; jumlah GPU terlihat: {torch.cuda.device_count()}")
            free, total = torch.cuda.mem_get_info(int(device))
            info["gpu"] = {"index_visible": int(device), "name": torch.cuda.get_device_name(int(device)),
                           "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}
    else:
        info["cuda_available"] = False
    return info


def load_ultralytics_without_install():
    """Disable Ultralytics auto-install even if it was imported earlier in this kernel."""
    os.environ["YOLO_AUTOINSTALL"] = "False"
    ultra = importlib.import_module("ultralytics")
    importlib.import_module("ultralytics.utils").AUTOINSTALL = False
    importlib.import_module("ultralytics.utils.checks").AUTOINSTALL = False
    return ultra


def require_training_environment(device=0):
    if int(device) != 0:
        raise ValueError("Gunakan GPU pertama yang diberikan scheduler: DEVICE=0 dan CUDA_VISIBLE_DEVICES dari HPC.")
    info = environment_info(device)
    missing = [name for name in ("torch", "ultralytics", "PyYAML") if not info["packages"][name]]
    if missing:
        raise RuntimeError("Paket belum tersedia pada kernel aktif: " + ", ".join(missing) +
                           ". Gunakan environment HPC yang sudah memiliki paket tersebut.")
    if not info["cuda_available"]:
        raise RuntimeError("PyTorch pada kernel aktif tidak mendeteksi CUDA; periksa GPU allocation dan kernel HPC.")
    return info


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path, content):
    """Jangan menimpa manifest/split yang pernah dipakai training."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Isi {path} berbeda. Gunakan OUTPUT_ROOT baru; data lama tidak ditimpa.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def image_paths(images_dir):
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {images_dir}")
    result = sorted((p.resolve() for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: str(p))
    if not result:
        raise ValueError(f"Tidak ada gambar dalam {images_dir}")
    # Ultralytics maps images/... to labels/... when a TXT file list is used.
    for image in result:
        if "images" not in image.parts:
            raise ValueError(f"Folder gambar harus bernama images agar label dipetakan dengan benar: {image}")
    return result


def label_for_image(image):
    image = Path(image)
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if not indices:
        raise ValueError(f"Folder images tidak ditemukan dalam path: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def audit_split(images):
    """Baca semua label; gambar negatif adalah label kosong dan tidak dihapus."""
    stats = Counter(images=len(images))
    class_boxes = Counter()
    examples = []
    fingerprint = hashlib.sha256()
    for image in images:
        label = label_for_image(image)
        if not label.is_file():
            stats["missing_labels"] += 1
            if len(examples) < 12:
                examples.append({"file": str(label), "issue": "missing_label"})
            continue
        try:
            raw_label = label.read_bytes()
            fingerprint.update(str(image).encode("utf-8"))
            fingerprint.update(str(image.stat().st_size).encode("ascii"))
            fingerprint.update(raw_label)
            lines = [s.strip() for s in raw_label.decode("utf-8-sig").splitlines() if s.strip()]
        except (UnicodeError, OSError) as exc:
            stats["unreadable_labels"] += 1
            if len(examples) < 12:
                examples.append({"file": str(label), "issue": str(exc)})
            continue
        if not lines:
            stats["negative_images"] += 1
        for lineno, line in enumerate(lines, 1):
            parts = line.split()
            issue = None
            try:
                if len(parts) != 5:
                    raise ValueError("harus 5 kolom")
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if cls not in (0, 1) or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("kelas/koordinat tidak valid")
                class_boxes[cls] += 1
                if not (0 <= x <= 1 and 0 <= y <= 1) or w < 0 or h < 0:
                    stats["invalid_values"] += 1
                    issue = "nilai koordinat negatif/di luar 0..1"
                if w == 0 or h == 0:
                    stats["zero_area_boxes"] += 1
                    issue = "bbox luas nol"
                if w > 1 or h > 1:
                    stats["oversized_boxes"] += 1
                    issue = "lebar/tinggi > 1"
                if x - w/2 < -1e-6 or y - h/2 < -1e-6 or x + w/2 > 1 + 1e-6 or y + h/2 > 1 + 1e-6:
                    stats["outside_image_boxes"] += 1
                    issue = issue or "tepi bbox di luar gambar"
            except (ValueError, OverflowError) as exc:
                stats["malformed_rows"] += 1
                issue = str(exc)
            if issue and len(examples) < 12:
                examples.append({"file": str(label), "line": lineno, "issue": issue, "raw": line})
    stats["class_0_boxes"] = class_boxes[0]
    stats["class_1_boxes"] = class_boxes[1]
    for name in ("missing_labels", "unreadable_labels", "negative_images", "invalid_values",
                 "zero_area_boxes", "oversized_boxes", "outside_image_boxes", "malformed_rows"):
        stats.setdefault(name, 0)
    return {**dict(stats), "label_and_image_list_sha256": fingerprint.hexdigest(),
            "examples": examples}


def _validate_classes(names, dataset):
    if not isinstance(names, (list, tuple)) or len(names) != 2:
        raise ValueError(f"{dataset}: isi urutan kelas sebagai list dua nama, misalnya ['smoke','fire'].")
    names = [str(x).strip().lower() for x in names]
    if set(names) != {"fire", "smoke"}:
        raise ValueError(f"{dataset}: nama harus 'fire' dan 'smoke', sesuai urutan label ID 0 dan 1.")
    return names


def prepare_datasets(smoke_root, dfire_root, output_root, smoke_classes, dfire_classes=("smoke", "fire"),
                     dfire_val_count=1723, split_seed=42, expected_counts=True):
    """Audit dan buat manifest TXT tanpa mengubah gambar atau label asli.

    D-Fire resmi: pecah train menjadi train/val, pertahankan seluruh test.
    Layout lama data/{train,val,test} tetap didukung dengan menggabungkan train+val.
    Seed dan pembagian ini adalah pilihan reproduksi, bukan split asli penulis.
    """
    smoke_root, dfire_root, output_root = [Path(x).expanduser().resolve() for x in
                                         (smoke_root, dfire_root, output_root)]
    classes = {"smoke_fire": _validate_classes(smoke_classes, "Smoke and Fire"),
               "dfire": _validate_classes(dfire_classes, "D-Fire")}
    roots = {"smoke_fire": smoke_root, "dfire": dfire_root}
    if any(output_root == root or root in output_root.parents for root in roots.values()):
        raise ValueError("OUTPUT_ROOT harus berada di luar folder dataset sumber.")
    official_dfire = (dfire_root / "train/images").is_dir()
    rel = {"smoke_fire": {"train": "train/images", "val": "valid/images", "test": "test/images"},
           "dfire": ({"train": "train/images", "test": "test/images"} if official_dfire else
                     {"train": "data/train/images", "val": "data/val/images", "test": "data/test/images"})}
    original = {ds: {split: image_paths(roots[ds] / folder) for split, folder in paths.items()}
                for ds, paths in rel.items()}
    audit = {}
    for dataset, group in original.items():
        for split, images in group.items():
            audit[f"{dataset}/{split}"] = audit_split(images)
        for split1, split2 in combinations(group, 2):
            # Basename is useful for finding collisions even across different source directories.
            a, b = {p.name for p in group[split1]}, {p.name for p in group[split2]}
            if a & b:
                raise ValueError(f"{dataset}: {len(a & b)} nama gambar sama antara {split1} dan {split2}.")
    failures = {name: {k: info[k] for k in ("missing_labels", "unreadable_labels", "malformed_rows", "invalid_values")
                       if info[k]}
                for name, info in audit.items()}
    failures = {name: issue for name, issue in failures.items() if issue}
    if failures:
        raise ValueError("Masalah label yang perlu diperiksa: " + json.dumps(failures, ensure_ascii=False))
    expected = {"smoke_fire": {"train": 8494, "val": 2114, "test": 1059},
                "dfire": ({"train": 17221, "test": 4306} if official_dfire else
                          {"train": 14122, "val": 3099, "test": 4306})}
    if expected_counts:
        for ds, group in original.items():
            for split, images in group.items():
                if len(images) != expected[ds][split]:
                    raise ValueError(f"{ds}/{split}: ditemukan {len(images)}, diharapkan {expected[ds][split]}. "
                                     "Periksa versi/layout; ubah expected_counts hanya jika yakin.")
    pool = sorted(original["dfire"]["train"] + original["dfire"].get("val", []), key=str)
    if not 0 < dfire_val_count < len(pool):
        raise ValueError("dfire_val_count harus di antara 1 dan jumlah train+val dikurangi 1.")
    random.Random(split_seed).shuffle(pool)
    selected = {"smoke_fire": original["smoke_fire"],
                "dfire": {"train": sorted(pool[dfire_val_count:], key=str),
                          "val": sorted(pool[:dfire_val_count], key=str), "test": original["dfire"]["test"]}}
    manifests = {}
    for dataset, groups in selected.items():
        destination = output_root / "data_manifests" / dataset
        lists = {}
        for split, paths in groups.items():
            text = "".join(str(p) + "\n" for p in paths)
            target = destination / f"{split}.txt"
            _write_once(target, text)
            lists[split] = str(target)
        data_yaml = destination / "data.yaml"
        # JSON is valid YAML 1.2; absolute paths make the manifests portable within the HPC.
        payload = {"path": str(destination), **lists, "names": classes[dataset], "nc": 2}
        _write_once(data_yaml, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        manifests[dataset] = {"yaml": str(data_yaml), "names": classes[dataset],
                              "split_sizes": {k: len(v) for k, v in groups.items()},
                              "list_sha256": {k: sha256_file(v) for k, v in lists.items()}}
    provenance = {"paper": PAPER, "roots": {k: str(v) for k, v in roots.items()},
                  "dfire_split_seed": split_seed, "dfire_val_count": dfire_val_count,
                  "dfire_original_split_unknown": True, "datasets": manifests, "audit_original_splits": audit}
    _write_once(output_root / "data_manifests" / "provenance.json",
                json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    return provenance


def show_label_samples(smoke_root, samples_per_class=4, seed=42):
    """Bantu menentukan kelas arsip Smoke and Fire melalui contoh anotasi ID 0 dan 1."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    candidates = {0: [], 1: []}
    for img in image_paths(Path(smoke_root) / "train/images"):
        lab = label_for_image(img)
        if not lab.is_file():
            continue
        lines = [line.split() for line in lab.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        labels = {int(row[0]) for row in lines if len(row) == 5 and row[0] in {"0", "1"}}
        for c in labels:
            if len(candidates[c]) < 80:
                candidates[c].append((img, lines))
        if all(len(v) >= 80 for v in candidates.values()):
            break
    rng = random.Random(seed)
    samples = []
    for cls in (0, 1):
        if not candidates[cls]:
            raise ValueError(f"Tidak ditemukan contoh untuk label ID {cls}.")
        samples.extend((cls, x) for x in rng.sample(candidates[cls], min(samples_per_class, len(candidates[cls]))))
    fig, axes = plt.subplots(2, samples_per_class, figsize=(4 * samples_per_class, 7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    positions = {0: 0, 1: 0}
    for cls, (path, lines) in samples:
        ax = axes[cls, positions[cls]]
        positions[cls] += 1
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            ax.imshow(im)
            for row in lines:
                if len(row) != 5:
                    continue
                c = int(row[0]); x, y, w, h = map(float, row[1:])
                if c != cls:
                    continue
                ax.add_patch(patches.Rectangle(((x-w/2)*width, (y-h/2)*height), w*width, h*height,
                                               fill=False, edgecolor="lime", linewidth=2))
            ax.set_title(f"ID {cls}: {path.name[:27]}", fontsize=9)
    fig.suptitle("Periksa objek pada bbox hijau sebelum menentukan urutan kelas")
    fig.tight_layout()
    plt.show()


def validate_weight_files(weights_dir, jobs):
    weights_dir = Path(weights_dir).expanduser().resolve()
    result = {}
    for model in sorted({j["model"] for j in jobs}):
        path = weights_dir / f"{model}.pt"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Bobot awal lokal tidak ditemukan: {path}. "
                                    "Sediakan bobot sebelumnya; kode ini tidak mengunduhnya.")
        result[model] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _metric_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def evaluate_checkpoint(model, job, dataset, output_root, device, batch):
    """Evaluasi best.pt pada val dan test; simpan setiap split terpisah."""
    results = {}
    for split in ("val", "test"):
        results_path = output_root / "evaluation" / job["id"] / split / "metrics.json"
        if results_path.is_file():
            results[split] = json.loads(results_path.read_text(encoding="utf-8"))
            continue
        # The API creates project/name; do not point a failed retry at an existing folder.
        eval_dir = output_root / "evaluation" / job["id"] / split
        if eval_dir.exists():
            raise FileExistsError(f"Evaluasi parsial sudah ada: {eval_dir}. Pindahkan folder tersebut sebelum mengulang.")
        metrics = model.val(data=dataset["yaml"], split=split, imgsz=job["train_args"]["imgsz"],
                            batch=batch, device="", project=str(eval_dir.parent), name=split,
                            exist_ok=False, plots=True, verbose=False)
        values = metrics.results_dict
        precision = _metric_float(values.get("metrics/precision(B)"))
        recall = _metric_float(values.get("metrics/recall(B)"))
        f1 = (2*precision*recall/(precision+recall)) if precision is not None and recall is not None and precision+recall else None
        box = metrics.box
        class_results = {}
        # ap_class_index maps each row of the metric arrays to its actual class ID.
        for row, class_id in enumerate(getattr(box, "ap_class_index", [])):
            idx = int(class_id)
            name = dataset["names"][idx]
            class_results[name] = {"class_id": idx, "ap50": _metric_float(box.ap50[row]),
                                   "ap50_95": _metric_float(box.ap[row]),
                                   "precision": _metric_float(box.p[row]), "recall": _metric_float(box.r[row])}
        result = {"split": split, "checkpoint": "best.pt", "precision": precision, "recall": recall,
                  "f1_from_aggregate_precision_recall": f1,
                  "map50": _metric_float(values.get("metrics/mAP50(B)")),
                  "map50_95": _metric_float(values.get("metrics/mAP50-95(B)")),
                  "per_class": class_results,
                  "note": "F1 dihitung dari P dan R agregat; bukan F1 pada threshold tetap yang terverifikasi."}
        _write_json(results_path, result)
        results[split] = result
    return results


def run_queue(jobs, provenance, weights_dir, output_root, device=0, workers=4, seed=42,
              eval_batch=16, continue_incomplete=True):
    """Jalankan run satu per satu; restart memakai last.pt jika tersedia.

    Tidak mengubah batch saat OOM. Folder tanpa checkpoint dihentikan untuk ditinjau.
    """
    output_root = Path(output_root).expanduser().resolve()
    info = require_training_environment(device)
    weights = validate_weight_files(weights_dir, jobs)
    YOLO = load_ultralytics_without_install().YOLO
    # Fail before a long queue if one of the local weights cannot be opened or is not a detector.
    for model_name, weight_info in weights.items():
        try:
            probe = YOLO(weight_info["path"])
            if getattr(probe, "task", None) != "detect":
                raise ValueError(f"task={getattr(probe, 'task', None)!r}; diharapkan detect")
            parameter_count = sum(p.numel() for p in probe.model.parameters())
            print(f"Bobot {model_name}: {parameter_count:,} parameter; task=detect", flush=True)
            del probe
        except Exception as exc:
            raise RuntimeError(f"Bobot {model_name} tidak kompatibel dengan environment aktif: {exc}") from exc
    selected = {j["id"] for j in jobs}
    if len(selected) != len(jobs):
        raise ValueError("ID eksperimen duplikat.")
    results = []
    for number, job in enumerate(jobs, 1):
        run_dir = output_root / "training" / job["id"]
        meta_path = output_root / "metadata" / f"{job['id']}.json"
        completed_path = output_root / "metadata" / f"{job['id']}.complete.json"
        current_config = {"job": job, "dataset": provenance["datasets"][job["dataset"]],
                          "weights": weights[job["model"]], "seed": seed, "device": device,
                          "workers": workers, "eval_batch": eval_batch, "ultralytics_device_arg": "",
                          "assumptions": [f"Seed={seed} dan deterministic=True adalah pilihan reproduksi, bukan informasi dari paper.",
                                          "Bobot awal lokal diasumsikan pretrained; jenis dan versi bobot paper tidak dipastikan.",
                                          "patience=0 memastikan jadwal epoch dijalankan penuh.",
                                          "Best checkpoint dievaluasi pada validation dan test secara terpisah."]}
        config_text = json.dumps(current_config, indent=2, ensure_ascii=False) + "\n"
        _write_once(meta_path, config_text)
        if completed_path.is_file():
            print(f"[{number}/{len(jobs)}] {job['id']}: sudah selesai, dilewati.")
            results.append(json.loads(completed_path.read_text(encoding="utf-8")))
            continue
        environment_log = output_root / "metadata" / f"{job['id']}.environment.jsonl"
        environment_log.parent.mkdir(parents=True, exist_ok=True)
        with environment_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                     "environment": info}, ensure_ascii=False, default=str) + "\n")
        print(f"[{number}/{len(jobs)}] {job['id']} | {job['dataset']} | {job['epochs']} epoch", flush=True)
        last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
        started = time.time()
        try:
            if run_dir.exists():
                if not continue_incomplete or not last.is_file():
                    raise FileExistsError(f"Run parsial ada tanpa checkpoint yang dapat dipakai: {run_dir}. "
                                          "Pindahkan folder setelah diperiksa atau gunakan OUTPUT_ROOT baru.")
                # If results.csv shows all epochs, do not resume an already-finished training.
                results_csv = run_dir / "results.csv"
                finished = False
                if results_csv.is_file():
                    with results_csv.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    if rows:
                        epoch_value = next((v for k, v in rows[-1].items() if k and k.strip() == "epoch"), None)
                        try:
                            # Ultralytics writes one-based epoch numbers in results.csv.
                            finished = int(float(epoch_value)) >= job["epochs"]
                        except (ValueError, TypeError):
                            finished = False
                if not finished:
                    print(f"Melanjutkan dari {last}", flush=True)
                    model = YOLO(str(last))
                    model.train(resume=True, device="", workers=workers)
            else:
                model = YOLO(weights[job["model"]]["path"])
                args = dict(job["train_args"])
                model.train(data=provenance["datasets"][job["dataset"]]["yaml"], epochs=job["epochs"],
                            seed=seed, deterministic=True, patience=0, device="", workers=workers,
                            project=str(output_root / "training"), name=job["id"], exist_ok=False,
                            plots=True, save=True, **args)
            if not best.is_file():
                raise FileNotFoundError(f"Training belum menghasilkan {best}.")
            model = YOLO(str(best))
            metrics = evaluate_checkpoint(model, job, provenance["datasets"][job["dataset"]],
                                          output_root, device, eval_batch)
            finished = {"id": job["id"], "status": "complete", "best_checkpoint": str(best),
                        "best_sha256": sha256_file(best), "metrics": metrics,
                        "elapsed_seconds_this_session": round(time.time() - started, 2),
                        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            _write_json(completed_path, finished)
            results.append(finished)
        except Exception as exc:
            error_path = output_root / "metadata" / f"{job['id']}.error.json"
            _write_json(error_path, {"id": job["id"], "status": "error", "error": str(exc),
                                     "traceback": traceback.format_exc(),
                                     "note": "Batch dan konfigurasi tidak diubah otomatis."})
            raise
    return results


def summarize_results(jobs, output_root):
    """Tulis tabel hasil paper dan hasil validasi/test; target paper tidak dipasangkan diam-diam."""
    output_root = Path(output_root).expanduser().resolve()
    rows = []
    for job in jobs:
        done = output_root / "metadata" / f"{job['id']}.complete.json"
        result = json.loads(done.read_text(encoding="utf-8")) if done.is_file() else {}
        row = {"run_id": job["id"], "table": job["table"], "dataset": job["dataset"],
               "model": job["model"], "epochs": job["epochs"], "status": result.get("status", "not_finished")}
        for key, value in job["paper_metrics"].items():
            row[f"paper_{key}"] = value
        for split in ("val", "test"):
            d = result.get("metrics", {}).get(split, {})
            for key in ("precision", "recall", "map50", "map50_95"):
                row[f"{split}_{key}"] = d.get(key)
            row[f"{split}_smoke_ap50"] = d.get("per_class", {}).get("smoke", {}).get("ap50")
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "comparison.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return target, rows
