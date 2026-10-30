"""Jalankan antrean reproduksi dari terminal, tanpa menginstal paket."""
import argparse
import os
from pathlib import Path
from pprint import pprint

os.environ["YOLO_AUTOINSTALL"] = "False"
import fire2025_reproduction as repro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--dfire-root", type=Path, required=True,
                        help="Root D-Fire resmi berisi train/{images,labels} dan test/{images,labels}")
    parser.add_argument("--weights-dir", type=Path, help="Folder bobot lokal; wajib untuk --mode train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-classes", nargs=2, choices=("smoke", "fire"), required=True,
                        help="Urutan nama kelas menurut ID 0 dan 1; verifikasi dataset terlebih dahulu")
    parser.add_argument("--mode", choices=("audit", "train"), default="audit",
                        help="audit: hanya periksa label dan tulis manifest, tanpa GPU atau bobot")
    parser.add_argument("--run-ids", nargs="+", help="ID eksperimen; default semua 13")
    parser.add_argument("--device", type=int, default=0,
                        help="Indeks GPU nonnegatif yang terlihat dalam CUDA_VISIBLE_DEVICES")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed split D-Fire dan training; pilihan reproduksi, bukan seed terverifikasi dari paper")
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device harus berupa indeks GPU nonnegatif")
    if args.mode == "train" and args.weights_dir is None:
        parser.error("--weights-dir wajib untuk --mode train")
    if set(args.smoke_classes) != {"smoke", "fire"}:
        parser.error("--smoke-classes harus berisi smoke dan fire masing-masing sekali")
    jobs = repro.build_experiments()
    selected = set(args.run_ids or [])
    unknown = selected - {job["id"] for job in jobs}
    if unknown:
        parser.error(f"ID eksperimen tidak dikenal: {sorted(unknown)}")
    chosen = [job for job in jobs if not selected or job["id"] in selected]
    if args.mode == "train":
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError("tqdm belum tersedia di environment HPC; gunakan environment yang menyediakannya.") from exc
        pprint(repro.require_training_environment(args.device))
        pprint(repro.validate_weight_files(args.weights_dir, chosen))
    provenance = repro.prepare_datasets(args.smoke_root, args.dfire_root, args.output_root,
                                       args.smoke_classes, ["smoke", "fire"],
                                       dfire_val_count=1723, split_seed=args.seed)
    pprint(provenance["datasets"])
    print("Audit label:", flush=True)
    pprint(provenance["audit_original_splits"])
    if args.mode == "audit":
        print("Audit selesai. Training tidak dijalankan.", flush=True)
        return
    try:
        with tqdm(total=len(chosen), desc="Eksperimen", unit="run", disable=False, mininterval=30) as progress:
            for job in chosen:
                progress.set_postfix_str(job["id"], refresh=True)
                repro.run_queue([job], provenance, args.weights_dir, args.output_root,
                                device=args.device, workers=args.workers, seed=args.seed,
                                eval_batch=args.eval_batch, continue_incomplete=True)
                progress.update(1)
    finally:
        path, _ = repro.summarize_results(jobs, args.output_root)
        print(f"Tabel perbandingan: {path}", flush=True)


if __name__ == "__main__":
    main()
