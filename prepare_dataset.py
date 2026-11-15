# prepare_dataset.py - Converts labelme JSON to YOLO format
# WITH auto-crop to tool region (removes white background)
# WITH 4x oversample of defect images

import os
import shutil
import random
import json
from PIL import Image

random.seed(42)

SRC_DEFECT = os.path.join("DataAI", "defect")
SRC_GOOD   = os.path.join("DataAI", "good")
YOLO_DIR   = os.path.join("DataAI", "yolo_dataset")
BG_THRESH  = 210
VAL_COUNT  = 3      # how many defect images go to val
OVERSAMPLE = 4      # repeat each defect image N times in train


def find_tool_bbox(img: Image.Image):
    """Find bounding box of dark tool on white background."""
    import numpy as np
    gray = np.array(img.convert("L"))
    H, W = gray.shape
    mask = gray < BG_THRESH
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, W, H
    pad_x = max(10, int((cols[-1]-cols[0])*0.05))
    pad_y = max(10, int((rows[-1]-rows[0])*0.05))
    return (max(0,cols[0]-pad_x), max(0,rows[0]-pad_y),
            min(W,cols[-1]+pad_x), min(H,rows[-1]+pad_y))


def json_to_yolo(json_path, img_w, img_h, cx1, cy1, cx2, cy2):
    """Convert labelme JSON to YOLO format, adjusting for crop."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    lines  = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts  = shape["points"]
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        xmin = max(cx1, min(xs))
        xmax = min(cx2, max(xs))
        ymin = max(cy1, min(ys))
        ymax = min(cy2, max(ys))
        if xmax <= cx1 or xmin >= cx2 or ymax <= cy1 or ymin >= cy2:
            continue
        cx = ((xmin+xmax)/2 - cx1) / crop_w
        cy = ((ymin+ymax)/2 - cy1) / crop_h
        bw = (xmax-xmin) / crop_w
        bh = (ymax-ymin) / crop_h
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        bw = max(0.002, min(0.999, bw))
        bh = max(0.002, min(0.999, bh))
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def save_item(src_img, src_ann, dst_img, dst_lbl):
    """Crop tool region, save image and YOLO label."""
    img  = Image.open(src_img).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = find_tool_bbox(img)
    img.crop((x1,y1,x2,y2)).save(dst_img, quality=95)
    if src_ann:
        lines = json_to_yolo(src_ann, W, H, x1, y1, x2, y2)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))
        return len(lines)
    else:
        open(dst_lbl, "w").close()
        return 0


def main():
    exts = {".jpg",".jpeg",".png",".bmp"}

    # Collect defect images with JSON
    defect_items = []
    if os.path.isdir(SRC_DEFECT):
        for fname in os.listdir(SRC_DEFECT):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            base = os.path.splitext(fname)[0]
            ann  = os.path.join(SRC_DEFECT, base+".json")
            if os.path.exists(ann):
                defect_items.append((os.path.join(SRC_DEFECT,fname), ann))
            else:
                print(f"  WARNING: no JSON for {fname} — label it in labelme")

    good_items = []
    if os.path.isdir(SRC_GOOD):
        for fname in os.listdir(SRC_GOOD):
            if os.path.splitext(fname)[1].lower() in exts:
                good_items.append(os.path.join(SRC_GOOD, fname))

    print(f"Defect photos with JSON : {len(defect_items)}")
    print(f"Good photos (negatives) : {len(good_items)}")

    if len(defect_items) == 0:
        print("\nERROR: No annotated defect photos found.")
        print("  1. Open labelme")
        print("  2. Open DataAI/defect/")
        print("  3. Draw rectangles around each chip, save")
        print("  4. Run this script again")
        return

    # Clear old dataset
    if os.path.isdir(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    for split in ("train","val"):
        os.makedirs(os.path.join(YOLO_DIR,"images",split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR,"labels",split), exist_ok=True)

    # Split: keep VAL_COUNT defect images for val
    random.shuffle(defect_items)
    val_d  = defect_items[:VAL_COUNT]
    trn_d  = defect_items[VAL_COUNT:]

    random.shuffle(good_items)
    n_gval = max(4, int(len(good_items)*0.2))
    val_g  = good_items[:n_gval]
    trn_g  = good_items[n_gval:]

    total_chips = 0

    # Defect train — oversample x OVERSAMPLE
    print(f"\nProcessing defect images (crop + YOLO labels, {OVERSAMPLE}x oversample)...")
    trn_d_over = [(img,ann,k)
                  for img,ann in trn_d
                  for k in range(OVERSAMPLE)]
    random.shuffle(trn_d_over)

    for src_img, src_ann, copy_idx in trn_d_over:
        base    = os.path.splitext(os.path.basename(src_img))[0]
        fname   = f"{base}_{copy_idx}.jpg" if copy_idx > 0 else f"{base}.jpg"
        dst_img = os.path.join(YOLO_DIR,"images","train",fname)
        dst_lbl = os.path.join(YOLO_DIR,"labels","train",
                               os.path.splitext(fname)[0]+".txt")
        n = save_item(src_img, src_ann, dst_img, dst_lbl)
        total_chips += n
        if copy_idx == 0:
            print(f"  [train] {os.path.basename(src_img)}  ->  {n} chip(s) (x{OVERSAMPLE})")

    # Defect val
    print(f"\nProcessing defect val images...")
    for src_img, src_ann in val_d:
        base    = os.path.splitext(os.path.basename(src_img))[0]
        fname   = f"{base}.jpg"
        dst_img = os.path.join(YOLO_DIR,"images","val",fname)
        dst_lbl = os.path.join(YOLO_DIR,"labels","val",
                               os.path.splitext(fname)[0]+".txt")
        n = save_item(src_img, src_ann, dst_img, dst_lbl)
        total_chips += n
        print(f"  [val]   {os.path.basename(src_img)}  ->  {n} chip(s)")

    # Good images
    print(f"\nProcessing good images (crop only)...")
    for src_img in trn_g + val_g:
        split   = "train" if src_img in trn_g else "val"
        base    = os.path.splitext(os.path.basename(src_img))[0]
        dst_img = os.path.join(YOLO_DIR,"images",split,f"{base}.jpg")
        dst_lbl = os.path.join(YOLO_DIR,"labels",split,f"{base}.txt")
        save_item(src_img, None, dst_img, dst_lbl)

    # dataset.yaml
    abs_dir   = os.path.abspath(YOLO_DIR)
    yaml_path = os.path.join(YOLO_DIR,"dataset.yaml")
    with open(yaml_path,"w") as f:
        f.write(f"path: {abs_dir}\n"
                f"train: images/train\n"
                f"val:   images/val\n"
                f"nc: 1\n"
                f"names: ['chip']\n")

    print(f"\nDone!")
    print(f"  Defect: train={len(trn_d)} x{OVERSAMPLE}={len(trn_d)*OVERSAMPLE}  val={len(val_d)}")
    print(f"  Good:   train={len(trn_g)}  val={len(val_g)}")
    print(f"  Total chips annotated: {total_chips}")
    print(f"  White background cropped — chips are ~2.5x larger")
    print(f"\nNext: press Переобучить in the app")


if __name__ == "__main__":
    main()
