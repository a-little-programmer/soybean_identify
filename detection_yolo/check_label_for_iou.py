import os
import json
from common import *  # expects label_src_dir defined here

def compute_iou_xyxy(a, b):
    """IoU for boxes (xmin, ymin, xmax, ymax) without external deps."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def scan_json_for_issues(folder, iou_threshold=0.5, edge_tol=0.0):
    """
    Checks:
      1) Overlapping boxes with IoU > iou_threshold
    """
    issues = {"overlaps": []}

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(folder, fname)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes = data.get("shapes", [])
            img_h = data.get("imageHeight")
            img_w = data.get("imageWidth")

            boxes, labels = [], []
            for s in shapes:
                if s.get("shape_type") == "rectangle" and "points" in s:
                    (x1, y1), (x2, y2) = s["points"]
                    xmin, ymin = min(float(x1), float(x2)), min(float(y1), float(y2))
                    xmax, ymax = max(float(x1), float(x2)), max(float(y1), float(y2))
                    boxes.append((xmin, ymin, xmax, ymax))
                    labels.append(s.get("label", ""))

            # --- 1) IoU overlaps ---
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = compute_iou_xyxy(boxes[i], boxes[j])
                    if iou > iou_threshold:
                        issues["overlaps"].append({
                            "file": fname,
                            "pair_idx": (i, j),
                            "iou": round(iou, 4),
                            "boxes": (boxes[i], boxes[j])
                        })

        except Exception as e:
            print(f"Error reading {fname}: {e}")

    return issues

if __name__ == "__main__":
    # Set edge_tol>0 (e.g., 2–5) if you want to allow slight annotation drift.
    issues = scan_json_for_issues(label_src_dir, iou_threshold=0.5, edge_tol=3.0)

    if issues["overlaps"]:
        print("Files with overlapping boxes (IoU > 0.5):")
        for item in issues["overlaps"]:
            print(f"{item['file']}  pair={item['pair_idx']}  IoU={item['iou']}  boxes={item['boxes']}")
    else:
        print("No overlapping bounding boxes found.")