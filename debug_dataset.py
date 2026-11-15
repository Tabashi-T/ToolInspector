# debug_dataset.py - checks what YOLO actually sees in your dataset
# Run: python debug_dataset.py

import os
import json
from PIL import Image, ImageDraw

DATA_DIR  = "DataAI"
YOLO_DIR  = os.path.join(DATA_DIR, "yolo_dataset")
DEBUG_DIR = os.path.join(DATA_DIR, "debug_preview")
os.makedirs(DEBUG_DIR, exist_ok=True)

print("="*56)
print("DATASET DIAGNOSTIC")
print("="*56)

# Check labels
for split in ("train", "val"):
    lbl_dir = os.path.join(YOLO_DIR, "labels", split)
    img_dir = os.path.join(YOLO_DIR, "images", split)

    if not os.path.isdir(lbl_dir):
        print(f"[{split}] labels folder NOT FOUND: {lbl_dir}")
        continue

    labels  = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    with_boxes  = []
    empty       = []

    for lf in labels:
        path = os.path.join(lbl_dir, lf)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            with_boxes.append((lf, lines))
        else:
            empty.append(lf)

    print(f"\n[{split}]  total={len(labels)}  "
          f"with_chips={len(with_boxes)}  empty(good)={len(empty)}")

    if with_boxes:
        print(f"  Files with chips:")
        for lf, lines in with_boxes:
            print(f"    {lf}  ->  {len(lines)} chip(s)")
            for l in lines:
                parts = l.split()
                if len(parts) == 5:
                    cls, cx, cy, bw, bh = parts
                    print(f"      class={cls}  cx={cx}  cy={cy}  w={bw}  h={bh}")
                else:
                    print(f"      BAD LINE: '{l}'")
    else:
        print(f"  WARNING: no chips in {split} set!")

# Draw preview of first 3 annotated images
print("\n" + "="*56)
print("DRAWING PREVIEW IMAGES -> DataAI/debug_preview/")
print("="*56)

drawn = 0
for split in ("train", "val"):
    lbl_dir = os.path.join(YOLO_DIR, "labels", split)
    img_dir = os.path.join(YOLO_DIR, "images", split)
    if not os.path.isdir(lbl_dir):
        continue

    for lf in os.listdir(lbl_dir):
        if drawn >= 6:
            break
        path = os.path.join(lbl_dir, lf)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue

        base    = os.path.splitext(lf)[0]
        img_path= None
        for ext in (".bmp",".jpg",".jpeg",".png"):
            p = os.path.join(img_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue

        img  = Image.open(img_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            draw.rectangle([x1,y1,x2,y2], outline="red", width=4)
            draw.rectangle([x1,y1,x1+60,y1+22], fill="red")
            draw.text((x1+4, y1+4), "chip", fill="white")

        out = os.path.join(DEBUG_DIR, f"{split}_{base}.jpg")
        img.thumbnail((800, 800))
        img.save(out, quality=90)
        print(f"  Saved: {out}  ({len(lines)} chips)")
        drawn += 1

print(f"\nOpen DataAI/debug_preview/ to check if boxes look correct")
print("If boxes are wrong size/position -> re-label in labelme")
