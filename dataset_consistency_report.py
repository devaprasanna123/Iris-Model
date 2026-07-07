import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dataset_exclusion import load_excluded_samples, is_excluded_sample



def list_files_by_ext(dir_path: Path, exts: set) -> List[Path]:
    if not dir_path.exists():
        return []
    out: List[Path] = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def safe_read_json(path: Path) -> Tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, ""
    except Exception as e:
        return False, str(e)


def stem(p: Path) -> str:
    return p.stem


def main():
    dataset_root = Path("MedicalAI") / "dataset"
    excluded = load_excluded_samples(Path(__file__).resolve().parent)

    images_dir = dataset_root / "images"
    anns_dir = dataset_root / "annotations"
    masks_dir = dataset_root / "masks"

    image_paths = list_files_by_ext(images_dir, {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    ann_paths = list_files_by_ext(anns_dir, {".json"})
    mask_paths = list_files_by_ext(masks_dir, {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"})

    image_paths = [p for p in image_paths if not is_excluded_sample(p.stem, excluded)]
    ann_paths = [p for p in ann_paths if not is_excluded_sample(p.stem, excluded)]
    mask_paths = [p for p in mask_paths if not is_excluded_sample(p.stem, excluded)]

    img_by_stem: Dict[str, Path] = {stem(p): p for p in image_paths}
    ann_by_stem: Dict[str, Path] = {stem(p): p for p in ann_paths}
    mask_by_stem: Dict[str, Path] = {stem(p): p for p in mask_paths}


    # Detect missing/corrupt json
    corrupted_json: List[Dict[str, Any]] = []
    for p in sorted(ann_paths, key=lambda x: x.name):
        ok, err = safe_read_json(p)
        if not ok:
            corrupted_json.append(
                {
                    "filename": p.name,
                    "full_path": str(p.resolve()),
                    "error": err,
                }
            )

    # Missing masks are determined relative to the image/annotation stems present.
    # (Masks are expected to exist for every image stem.)
    missing_mask_stems = sorted([s for s in img_by_stem.keys() if s not in mask_by_stem])
    missing_masks = [img_by_stem[s] for s in missing_mask_stems if s in img_by_stem]


    # Cross-check for presence issues
    images_without_annotations = sorted([s for s in img_by_stem.keys() if s not in ann_by_stem])
    annotations_without_images = sorted([s for s in ann_by_stem.keys() if s not in img_by_stem])
    images_without_masks = sorted([s for s in img_by_stem.keys() if s not in mask_by_stem])
    masks_without_images = sorted([s for s in mask_by_stem.keys() if s not in img_by_stem])

    # JSON-report requires filenames + full paths
    def pack_stems(stems_list: List[str], mapping: Dict[str, Path]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in stems_list:
            if s in mapping:
                p = mapping[s]
                out.append({"stem": s, "filename": p.name, "full_path": str(p.resolve())})
            else:
                out.append({"stem": s, "filename": None, "full_path": None})
        return out

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "counts": {
            "images": len(image_paths),
            "annotations": len(ann_paths),
            "masks": len(mask_paths),
            "corrupted_json_files": len(corrupted_json),
            "missing_masks": len(missing_masks),
        },
        "offenders": {
            "missing_mask": [
                {
                    "stem": m.stem,
                    "filename": m.name,
                    "full_path": str(m.resolve()),
                }
                for m in missing_masks
            ],
            "corrupted_json": corrupted_json,
        },
        "comparisons_by_filename_stems": {
            "images_without_annotations": pack_stems(images_without_annotations, img_by_stem),
            "annotations_without_images": pack_stems(annotations_without_images, ann_by_stem),
            "images_without_masks": pack_stems(images_without_masks, img_by_stem),
            "masks_without_images": pack_stems(masks_without_images, mask_by_stem),
        },
    }

    out_path = dataset_root / "dataset_consistency_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print the exact offending filenames and full paths
    if report["offenders"]["missing_mask"]:
        print("Missing mask offenders (using annotation stems as reference):")
        for x in report["offenders"]["missing_mask"]:
            print(f"- filename: {x['filename']}")
            print(f"  full_path: {x['full_path']}")
    else:
        print("Missing mask offenders: None")

    if report["offenders"]["corrupted_json"]:
        print("Corrupted JSON offenders:")
        for x in report["offenders"]["corrupted_json"]:
            print(f"- filename: {x['filename']}")
            print(f"  full_path: {x['full_path']}")
            print(f"  error: {x['error']}")
    else:
        print("Corrupted JSON offenders: None")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

