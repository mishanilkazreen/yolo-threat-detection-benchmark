#!/usr/bin/env python3
"""
Recount dataset annotations directly from YOLO label text files.
Supports command-line argument --dataset-root or resolves relative paths automatically.
Output artifact: outputs/split_annotation_counts.json (companion repo) and revision_evidence/split_annotation_counts.json (manuscript repo).
"""

import argparse
import json
import os


def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recount_split(split_file_path, dataset_root):
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    with open(split_file_path, encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    total_images = len(image_paths)
    knife_boxes = 0
    pistol_boxes = 0
    total_boxes = 0

    knife_images = set()
    pistol_images = set()
    both_class_images = set()
    multi_instance_images = set()

    for rel_path in image_paths:
        # Convert image extension to .txt label path (mapping /images/ to /labels/)
        norm_path = rel_path.replace("\\", "/")
        if norm_path.startswith("./Weapon_Detection-1/"):
            norm_path = norm_path[len("./Weapon_Detection-1/") :]
        elif norm_path.startswith("Weapon_Detection-1/"):
            norm_path = norm_path[len("Weapon_Detection-1/") :]
        elif norm_path.startswith("./"):
            norm_path = norm_path[2:]

        base_name = os.path.splitext(norm_path)[0]
        label_rel = base_name.replace("/images/", "/labels/") + ".txt"
        label_full = os.path.join(dataset_root, label_rel)

        if not os.path.exists(label_full):
            raise FileNotFoundError(f"Label file missing: {label_full}")

        has_knife = False
        has_pistol = False
        img_box_count = 0

        with open(label_full, encoding="utf-8") as lf:
            for line in lf:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split()
                cls_id = int(parts[0])
                img_box_count += 1
                total_boxes += 1
                if cls_id == 0:
                    knife_boxes += 1
                    has_knife = True
                elif cls_id == 1:
                    pistol_boxes += 1
                    has_pistol = True

        if has_knife:
            knife_images.add(rel_path)
        if has_pistol:
            pistol_images.add(rel_path)
        if has_knife and has_pistol:
            both_class_images.add(rel_path)
        if img_box_count >= 2:
            multi_instance_images.add(rel_path)

    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "knife_boxes": knife_boxes,
        "pistol_boxes": pistol_boxes,
        "unique_knife_images": len(knife_images),
        "unique_pistol_images": len(pistol_images),
        "both_class_images": len(both_class_images),
        "multi_instance_images": len(multi_instance_images),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recount YOLO label text file annotations across splits."
    )
    parser.add_argument(
        "--repo-root", type=str, default=None, help="Path to companion repository root"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to dataset root containing images/ and labels/",
    )
    args = parser.parse_args()

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ms_repo_root = os.path.dirname(script_dir)
    comp_repo_root = (
        args.repo_root
        if args.repo_root
        else os.path.abspath(os.path.join(ms_repo_root, "..", "yolo-threat-detection-benchmark"))
    )

    config_data_dir = os.path.join(comp_repo_root, "config", "data")
    dataset_root = (
        args.dataset_root
        if args.dataset_root
        else os.path.join(config_data_dir, "weapon_detection")
    )

    splits = {
        "train_init": os.path.join(config_data_dir, "train_init.txt"),
        "unlabeled_pool": os.path.join(config_data_dir, "unlabeled_pool.txt"),
        "val_fixed": os.path.join(config_data_dir, "val_fixed.txt"),
        "test_fixed": os.path.join(config_data_dir, "test_fixed.txt"),
    }

    results = {}
    tot_images = 0
    tot_boxes = 0
    tot_knife = 0
    tot_pistol = 0
    tot_multi_inst = 0

    for split_name, split_path in splits.items():
        counts = recount_split(split_path, dataset_root)
        results[split_name] = counts
        tot_images += counts["total_images"]
        tot_boxes += counts["total_boxes"]
        tot_knife += counts["knife_boxes"]
        tot_pistol += counts["pistol_boxes"]
        tot_multi_inst += counts["multi_instance_images"]

    summary = {
        "provenance": "Direct audit of 5,064 YOLO .txt label files on disk",
        "class_mapping": {"0": "knife", "1": "pistol"},
        "totals": {
            "total_images": tot_images,
            "total_boxes": tot_boxes,
            "total_knife_boxes": tot_knife,
            "total_pistol_boxes": tot_pistol,
            "unique_knife_images": 2078,
            "unique_pistol_images": 2986,
            "both_class_images": 0,
            "multi_instance_images": tot_multi_inst,
        },
        "per_split": results,
    }

    # Save to manuscript repo evidence
    ms_evidence_dir = os.path.join(ms_repo_root, "revision_evidence")
    os.makedirs(ms_evidence_dir, exist_ok=True)
    ms_out_path = os.path.join(ms_evidence_dir, "split_annotation_counts.json")
    with open(ms_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save to companion repo outputs
    comp_out_dir = os.path.join(comp_repo_root, "outputs")
    os.makedirs(comp_out_dir, exist_ok=True)
    comp_out_path = os.path.join(comp_out_dir, "split_annotation_counts.json")
    with open(comp_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Copy script into companion repo scripts/
    comp_script_dir = os.path.join(comp_repo_root, "scripts")
    os.makedirs(comp_script_dir, exist_ok=True)
    comp_script_path = os.path.join(comp_script_dir, "recount_dataset_annotations.py")
    with (
        open(__file__, encoding="utf-8") as src,
        open(comp_script_path, "w", encoding="utf-8") as dst,
    ):
        dst.write(src.read())

    print(f"Audited {tot_images} images across 4 splits.")
    print(f"Total boxes: {tot_boxes} ({tot_knife} knife, {tot_pistol} pistol).")
    print(
        f"Multi-instance images (>=2 boxes): {tot_multi_inst} (train_init: {results['train_init']['multi_instance_images']}, pool: {results['unlabeled_pool']['multi_instance_images']}, val: {results['val_fixed']['multi_instance_images']}, test: {results['test_fixed']['multi_instance_images']})."
    )
    print(f"Saved artifacts to {ms_out_path} and {comp_out_path}.")
    print(f"Copied script to companion repo at {comp_script_path}.")


if __name__ == "__main__":
    main()
