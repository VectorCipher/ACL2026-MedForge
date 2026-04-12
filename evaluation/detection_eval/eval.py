#!/usr/bin/env python3
"""
Visualize bounding box inference results and calculate IoU metrics with multi-processing acceleration.

Note: This script is used for processing inference results from the model.
"""

import os
import json
import re
import argparse
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
DEFAULT_LINE_WIDTH = 4
FONT_SIZE_LABEL = 24
FONT_SIZE_IOU = 32
FONT_SIZE_STATUS = 20
COORD_SCALE = 1000  # Qwen-VL uses 0-1000 normalization
DATASET_MAPPING_PATH = "dataset_mapping.json"

# Global variable for dataset mapping (used by worker processes)
_DATASET_MAPPING = None


class MeanMetric:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def get_font(size: int = 24):
    """Load a font or fallback to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue
    return ImageFont.load_default()


def draw_text_with_outline(draw, pos, text, font, fill="white", outline="black", outline_width=2):
    """Draw text with an outline."""
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(pos, text, font=font, fill=fill)


def init_worker(mapping_path: str):
    """Initialize worker process with dataset mapping (called once per worker)."""
    global _DATASET_MAPPING
    if _DATASET_MAPPING is None:
        print(f"[Worker] Loading dataset mapping...")
        if not os.path.exists(mapping_path):
            print(f"[WARNING] Dataset mapping file not found: {mapping_path}")
            _DATASET_MAPPING = {}
            return
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a mapping from image_path to item info
        _DATASET_MAPPING = {}
        for item in data.get('items', []):
            img_path = item.get('image_path', '')
            if img_path:
                _DATASET_MAPPING[img_path] = item
        
        print(f"[Worker] Loaded {len(_DATASET_MAPPING)} entries")


def get_image_metadata(img_path: str, dataset_mapping: Optional[Dict[str, Dict]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract dataset, type, and model from image path using dataset_mapping.
    
    Args:
        img_path: Path to the image
        dataset_mapping: Optional mapping dict. If None, uses global _DATASET_MAPPING
    
    Returns:
        Tuple of (dataset, type, model) where:
        - dataset: 'chest-xray', 'fundus', 'brain-mri', or None
        - type: 'real', 'edit', 'remove', or None
        - model: model name or None
    """
    if not img_path:
        return None, None, None
    
    # Use global mapping if not provided
    if dataset_mapping is None:
        global _DATASET_MAPPING
        dataset_mapping = _DATASET_MAPPING or {}
    
    # Normalize path for matching
    normalized_path = img_path
    
    # Try to find in dataset_mapping
    if normalized_path in dataset_mapping:
        item = dataset_mapping[normalized_path]
        return item.get('dataset'), item.get('type'), item.get('model')
    
    # Fallback: parse from path
    parts = img_path.split(os.sep)
    
    # Determine dataset
    dataset = None
    if 'chest-xray' in img_path:
        dataset = 'chest-xray'
    elif 'fundus' in img_path:
        dataset = 'fundus'
    elif 'brain-mri' in img_path:
        dataset = 'brain-mri'
    
    # Determine type
    img_type = None
    if '/real/' in img_path:
        img_type = 'real'
    elif '-edit.' in img_path:
        img_type = 'edit'
    elif '-remove.' in img_path:
        img_type = 'remove'
    
    # Determine model
    model = None
    try:
        deepfake_idx = parts.index('deepfake')
        if deepfake_idx + 1 < len(parts):
            model = parts[deepfake_idx + 1]
    except ValueError:
        pass
    
    return dataset, img_type, model


def process_single_item(args):
    """Worker function to process a single JSONL line."""
    line_num, line_content, output_dir, should_visualize = args
    try:
        data = json.loads(line_content)
    except json.JSONDecodeError:
        return None, "skipped"

    response = data.get("response", "")
    labels = data.get("labels", "")
    images = data.get("images", [])
    
    if not images:
        return None, "skipped"
            
    img_path = images[0].get("path") if isinstance(images[0], dict) else images[0]
    if not img_path or not os.path.exists(img_path):
        return None, "skipped"
            
    pred_class = classify_image(response)
    gt_class = classify_image(labels)
    pred_bbox = extract_bbox(response)
    gt_bbox = extract_bbox(labels)
    model_name = extract_model_name(img_path)
    
    # Get image metadata (uses global _DATASET_MAPPING)
    dataset, img_type, _ = get_image_metadata(img_path)
    
    stats_update = {
        'tp': 1 if pred_class == 'deepfake' and gt_class == 'deepfake' else 0,
        'tn': 1 if pred_class == 'real' and gt_class == 'real' else 0,
        'fp': 1 if pred_class == 'deepfake' and gt_class == 'real' else 0,
        'fn': 1 if pred_class == 'real' and gt_class == 'deepfake' else 0,
        'iou': None,
        'perfect': 0,
        'good': 0,
        'missed': 0,
        'false_alarm': 0
    }
    
    iou = None
    # Only compute IoU for deepfake predictions (real images don't need IoU)
    if pred_class == 'deepfake' and gt_class == 'deepfake':
        if pred_bbox and gt_bbox:
            iou = compute_iou(pred_bbox, gt_bbox)
            stats_update['iou'] = iou
            if iou > 0.95: stats_update['perfect'] = 1
            if iou > 0.5: stats_update['good'] = 1
        elif gt_bbox and not pred_bbox:
            stats_update['missed'] = 1
            iou = 0.0
            stats_update['iou'] = 0.0
    elif pred_class == 'real' and gt_class == 'deepfake':
        stats_update['missed'] = 1
        # No IoU calculation for real predictions
    elif pred_class == 'deepfake' and gt_class == 'real':
        stats_update['false_alarm'] = 1
        # No IoU calculation for false alarms
    
    img_basename = os.path.basename(img_path)
    model_prefix = f"{model_name}/" if model_name else ""
    
    # Only visualize deepfake predictions, skip real images
    vis_ok = True
    if not should_visualize:
        # Skip visualization if limit reached
        vis_ok = False
    elif pred_class == 'real':
        # Skip visualization for real images (no need to visualize)
        vis_ok = False
    else:
        # Only visualize deepfake predictions
        output_subdir = "deepfake"
        output_filename = f"vis_bbox/{output_subdir}/{model_prefix}{os.path.splitext(img_basename)[0]}_bbox" + \
                          (f"_iou_{iou:.4f}.png" if iou is not None else ".png")
        output_path = os.path.join(output_dir, output_filename)
        vis_ok = visualize_result(img_path, output_path, pred_bbox, gt_bbox, iou)
    
    result_info = {
        "line_num": line_num, "image_path": img_path, "image_name": img_basename,
        "model_name": model_name, "pred_class": pred_class, "gt_class": gt_class,
        "pred_bbox": pred_bbox, "gt_bbox": gt_bbox, "iou": iou,
        "dataset": dataset, "type": img_type,
        "response": response, "labels": labels
    }
    
    return {
        "result_info": result_info,
        "stats_update": stats_update,
        "vis_ok": vis_ok
    }, "ok"


def process_infer_results(input_jsonl: str, output_dir: str, num_workers: int = 8, max_vis_per_model: int = 100) -> None:
    """Process all results in the JSONL file using multi-processing."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'processed': 0, 'skipped': 0, 'total_iou': 0.0, 'iou_count': 0,
        'perfect_matches': 0, 'good_matches': 0, 'missed_detections': 0, 'false_alarms': 0,
        'true_positives': 0, 'true_negatives': 0, 'false_positives': 0, 'false_negatives': 0,
    }
    
    # Statistics by type and dataset
    type_stats = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
    dataset_type_stats = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}))
    
    all_results = []
    results_by_image = {}
    
    print(f"Reading from: {input_jsonl}")
    print(f"Output directory: {output_dir}")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Found {total_lines} lines. Using {num_workers} workers.")
    print(f"Will visualize max {max_vis_per_model} images per model.")

    # First pass: process all items without visualization to collect data
    tasks = [(i + 1, line, output_dir, False) for i, line in enumerate(lines)]
    all_processed_results = []
    
    # Use initializer to load dataset_mapping once per worker
    with ProcessPoolExecutor(max_workers=num_workers, 
                            initializer=init_worker, 
                            initargs=(DATASET_MAPPING_PATH,)) as executor:
        futures = {executor.submit(process_single_item, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res, status = future.result()
            if status == "skipped":
                stats['skipped'] += 1
                continue
            
            # Aggregating statistics
            su = res['stats_update']
            stats['true_positives'] += su['tp']
            stats['true_negatives'] += su['tn']
            stats['false_positives'] += su['fp']
            stats['false_negatives'] += su['fn']
            if su['iou'] is not None:
                stats['total_iou'] += su['iou']
                stats['iou_count'] += 1
            stats['perfect_matches'] += su['perfect']
            stats['good_matches'] += su['good']
            stats['missed_detections'] += su['missed']
            stats['false_alarms'] += su['false_alarm']
            
            all_results.append(res['result_info'])
            
            info = res['result_info']
            results_by_image[info['image_path']] = {
                "image_path": info['image_path'], "pred_class": info['pred_class'],
                "gt_class": info['gt_class'], "iou": info['iou']
            }
            
            # Collect type-based and dataset-type-based statistics
            img_type = info.get('type')
            dataset = info.get('dataset')
            pred_class = info['pred_class']
            gt_class = info['gt_class']
            
            if img_type:
                # For type-based statistics
                # We need to collect stats for ALL images to compute metrics for each type
                # For 'real' type metrics: positive=real, negative=fake (edit+remove)
                # For 'edit'/'remove' type metrics: positive=fake, negative=real
                
                # For 'real' type classification
                if img_type == 'real':
                    if pred_class == 'real' and gt_class == 'real':
                        type_stats['real']['tp'] += 1  # TP: correctly identified real
                    elif pred_class == 'deepfake' and gt_class == 'real':
                        type_stats['real']['fn'] += 1  # FN: real but predicted as fake
                elif img_type in ['edit', 'remove']:
                    # These are negative samples for 'real' classification
                    if pred_class == 'real' and gt_class == 'deepfake':
                        type_stats['real']['fp'] += 1  # FP: fake but predicted as real
                    elif pred_class == 'deepfake' and gt_class == 'deepfake':
                        type_stats['real']['tn'] += 1  # TN: correctly identified as fake
                
                # For 'edit' type classification
                if img_type == 'edit':
                    if pred_class == 'deepfake' and gt_class == 'deepfake':
                        type_stats['edit']['tp'] += 1  # TP: correctly identified edit as fake
                    elif pred_class == 'real' and gt_class == 'deepfake':
                        type_stats['edit']['fn'] += 1  # FN: edit but predicted as real
                elif img_type == 'real':
                    # Real images are negative samples for 'edit' classification
                    if pred_class == 'deepfake' and gt_class == 'real':
                        type_stats['edit']['fp'] += 1  # FP: real but predicted as fake
                    elif pred_class == 'real' and gt_class == 'real':
                        type_stats['edit']['tn'] += 1  # TN: correctly identified as real
                
                # For 'remove' type classification
                if img_type == 'remove':
                    if pred_class == 'deepfake' and gt_class == 'deepfake':
                        type_stats['remove']['tp'] += 1  # TP: correctly identified remove as fake
                    elif pred_class == 'real' and gt_class == 'deepfake':
                        type_stats['remove']['fn'] += 1  # FN: remove but predicted as real
                elif img_type == 'real':
                    # Real images are negative samples for 'remove' classification
                    if pred_class == 'deepfake' and gt_class == 'real':
                        type_stats['remove']['fp'] += 1  # FP: real but predicted as fake
                    elif pred_class == 'real' and gt_class == 'real':
                        type_stats['remove']['tn'] += 1  # TN: correctly identified as real
                
                # For dataset-type-based statistics (same logic)
                if dataset:
                    # For 'real' type in this dataset
                    if img_type == 'real':
                        if pred_class == 'real' and gt_class == 'real':
                            dataset_type_stats[dataset]['real']['tp'] += 1
                        elif pred_class == 'deepfake' and gt_class == 'real':
                            dataset_type_stats[dataset]['real']['fn'] += 1
                    elif img_type in ['edit', 'remove']:
                        if pred_class == 'real' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['real']['fp'] += 1
                        elif pred_class == 'deepfake' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['real']['tn'] += 1
                    
                    # For 'edit' type in this dataset
                    if img_type == 'edit':
                        if pred_class == 'deepfake' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['edit']['tp'] += 1
                        elif pred_class == 'real' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['edit']['fn'] += 1
                    elif img_type == 'real':
                        if pred_class == 'deepfake' and gt_class == 'real':
                            dataset_type_stats[dataset]['edit']['fp'] += 1
                        elif pred_class == 'real' and gt_class == 'real':
                            dataset_type_stats[dataset]['edit']['tn'] += 1
                    
                    # For 'remove' type in this dataset
                    if img_type == 'remove':
                        if pred_class == 'deepfake' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['remove']['tp'] += 1
                        elif pred_class == 'real' and gt_class == 'deepfake':
                            dataset_type_stats[dataset]['remove']['fn'] += 1
                    elif img_type == 'real':
                        if pred_class == 'deepfake' and gt_class == 'real':
                            dataset_type_stats[dataset]['remove']['fp'] += 1
                        elif pred_class == 'real' and gt_class == 'real':
                            dataset_type_stats[dataset]['remove']['tn'] += 1
            
            all_processed_results.append(res)

            if (i + 1) % 50 == 0 or (i + 1) == total_lines:
                print(f"Progress: [{i+1}/{total_lines}] Processed...")

    # Second pass: select top N per model for visualization
    print("\nSelecting images for visualization (max {} per model)...".format(max_vis_per_model))
    model_to_items = {}
    for res in all_processed_results:
        info = res['result_info']
        model_name = info.get('model_name') or 'unknown'
        pred_class = info.get('pred_class')
        
        # Only visualize deepfake predictions
        if pred_class == 'deepfake':
            if model_name not in model_to_items:
                model_to_items[model_name] = []
            model_to_items[model_name].append((res, info))
    
    # Select top N per model (keep original order, which is line order)
    items_to_visualize = set()
    for model_name, items in model_to_items.items():
        selected = items[:max_vis_per_model]
        for res, info in selected:
            items_to_visualize.add(info['image_path'])
        print(f"Model '{model_name}': selected {len(selected)}/{len(items)} images for visualization")
    
    # Third pass: visualize selected images
    print(f"\nVisualizing {len(items_to_visualize)} selected images...")
    vis_tasks = []
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            images = data.get("images", [])
            if not images:
                continue
            img_path = images[0].get("path") if isinstance(images[0], dict) else images[0]
            if img_path and img_path in items_to_visualize:
                vis_tasks.append((i + 1, line, output_dir, True))
        except:
            continue
    
    # Use initializer for visualization pass as well
    with ProcessPoolExecutor(max_workers=num_workers,
                            initializer=init_worker,
                            initargs=(DATASET_MAPPING_PATH,)) as executor:
        futures = {executor.submit(process_single_item, task): task for task in vis_tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res, status = future.result()
            if status == "ok" and res and res.get('vis_ok'):
                stats['processed'] += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(vis_tasks):
                print(f"Visualization progress: [{i+1}/{len(vis_tasks)}]")

    save_summary(output_dir, all_results, stats, results_by_image, type_stats, dataset_type_stats)


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total
    }


def save_summary(output_dir: str, results: List[Dict], stats: Dict, results_by_image: Dict, 
                 type_stats: Dict, dataset_type_stats: Dict):
    """Save statistics and result details to JSON files."""
    mean_iou = stats['total_iou'] / stats['iou_count'] if stats['iou_count'] > 0 else 0
    total_classified = stats['true_positives'] + stats['true_negatives'] + stats['false_positives'] + stats['false_negatives']
    accuracy = (stats['true_positives'] + stats['true_negatives']) / total_classified if total_classified > 0 else 0.0
    tp, fp, fn = stats['true_positives'], stats['false_positives'], stats['false_negatives']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute type-based metrics (Category 1: Overall by type)
    type_metrics = {}
    for img_type in ['real', 'edit', 'remove']:
        if img_type in type_stats:
            s = type_stats[img_type]
            type_metrics[img_type] = compute_metrics(s['tp'], s['tn'], s['fp'], s['fn'])
        else:
            type_metrics[img_type] = compute_metrics(0, 0, 0, 0)
    
    # Compute dataset-type-based metrics (Category 2: By dataset and type)
    dataset_metrics = {}
    for dataset in ['chest-xray', 'fundus', 'brain-mri']:
        dataset_metrics[dataset] = {}
        for img_type in ['real', 'edit', 'remove']:
            if dataset in dataset_type_stats and img_type in dataset_type_stats[dataset]:
                s = dataset_type_stats[dataset][img_type]
                dataset_metrics[dataset][img_type] = compute_metrics(s['tp'], s['tn'], s['fp'], s['fn'])
            else:
                dataset_metrics[dataset][img_type] = compute_metrics(0, 0, 0, 0)
    
    summary = {
        "total_lines": len(results) + stats['skipped'],
        "processed": stats['processed'], "skipped": stats['skipped'],
        "classification_metrics": {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,
            "true_positives": stats['true_positives'], "true_negatives": stats['true_negatives'],
            "false_positives": stats['false_positives'], "false_negatives": stats['false_negatives']
        },
        "localization_metrics": {
            "mean_iou": mean_iou, "iou_count": stats['iou_count'],
            "perfect_matches_gt_0.95": stats['perfect_matches'], "good_matches_gt_0.5": stats['good_matches'],
            "missed_detections": stats['missed_detections'], "false_alarms": stats['false_alarms']
        },
        "type_based_metrics": {
            "description": "Metrics by image type (real/edit/remove). For 'real': positive=real, negative=fake. For 'edit'/'remove': positive=fake, negative=real.",
            "real": type_metrics.get('real', {}),
            "edit": type_metrics.get('edit', {}),
            "remove": type_metrics.get('remove', {})
        },
        "dataset_type_based_metrics": {
            "description": "Metrics by dataset and image type. For 'real': positive=real, negative=fake. For 'edit'/'remove': positive=fake, negative=real.",
            "chest-xray": dataset_metrics.get('chest-xray', {}),
            "fundus": dataset_metrics.get('fundus', {}),
            "brain-mri": dataset_metrics.get('brain-mri', {})
        }
    }
    results.sort(key=lambda x: x['iou'] if x['iou'] is not None else -1)
    
    with open(os.path.join(output_dir, "summary_bbox.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "details_bbox.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "results_by_image.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_image, f, indent=2, ensure_ascii=False)
        
    print(f"\nProcessing Complete. Processed: {stats['processed']}, Skipped: {stats['skipped']}")
    print(f"Overall - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}, Mean IoU: {mean_iou:.4f}")
    
    # Print type-based metrics
    print("\n=== Type-based Metrics ===")
    for img_type in ['real', 'edit', 'remove']:
        m = type_metrics.get(img_type, {})
        if m.get('total', 0) > 0:
            print(f"{img_type:8s}: Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f} (n={m['total']})")
    
    # Print dataset-type-based metrics
    print("\n=== Dataset-Type-based Metrics ===")
    for dataset in ['chest-xray', 'fundus', 'brain-mri']:
        print(f"\n{dataset}:")
        for img_type in ['real', 'edit', 'remove']:
            m = dataset_metrics.get(dataset, {}).get(img_type, {})
            if m.get('total', 0) > 0:
                print(f"  {img_type:8s}: Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f} (n={m['total']})")


def generate_output_path(input_path: str) -> str:
    """Generate output path based on input path pattern."""
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    match = re.search(r'checkpoint-(\d+)\.json', input_filename)
    if match:
        return os.path.join(input_dir, f"infer_results-{match.group(1)}")
    return os.path.join(input_dir, f"infer_results_{os.path.splitext(input_filename)[0]}")


def main():
    parser = argparse.ArgumentParser(description="Deepfake Bbox Visualization & Metrics (Multi-threaded)")
    parser.add_argument("--input", required=True, help="Path to infer_results.jsonl")
    parser.add_argument("--output", required=False, default=None, help="Output directory")
    parser.add_argument("--workers", type=int, default=64, help="Number of processes for parallel processing")
    parser.add_argument("--max-vis-per-model", type=int, default=100, help="Maximum number of images to visualize per model (default: 100)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
    if args.output is None:
        args.output = generate_output_path(args.input)
    
    process_infer_results(args.input, args.output, num_workers=args.workers, max_vis_per_model=args.max_vis_per_model)


if __name__ == "__main__":
    main()