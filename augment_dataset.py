# augment_dataset.py - Dataset augmentation for EfficientNet
# ONLY: 180 degree rotation and/or mirror (horizontal flip)
# No random angles - they caused model to focus on center instead of edges

import os
import random
from PIL import Image

random.seed(42)

SRC_GOOD   = os.path.join("DataAI", "good")
SRC_DEFECT = os.path.join("DataAI", "defect")
DST_GOOD   = os.path.join("DataAI", "augmented", "good")
DST_DEFECT = os.path.join("DataAI", "augmented", "defect")

# Augmentation variants - ONLY 180 rotation and mirror
# Each original produces 4 versions: original, mirror, 180, mirror+180
VARIANTS = [
    ("orig",       False, 0),
    ("mirror",     True,  0),
    ("rot180",     False, 180),
    ("mirror_rot", True,  180),
]


def augment(img: Image.Image, mirror: bool, angle: int) -> Image.Image:
    result = img.copy()
    if mirror:
        result = result.transpose(Image.FLIP_LEFT_RIGHT)
    if angle == 180:
        result = result.rotate(180)
    return result


def process_folder(src: str, dst: str, label: str):
    os.makedirs(dst, exist_ok=True)
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [f for f in os.listdir(src)
             if os.path.splitext(f)[1].lower() in exts]

    if not files:
        print(f"  {label}: no images found in {src}")
        return 0

    total = 0
    for fname in files:
        img  = Image.open(os.path.join(src, fname)).convert("RGB")
        base = os.path.splitext(fname)[0]

        for suffix, mirror, angle in VARIANTS:
            aug     = augment(img, mirror, angle)
            outname = f"{base}_{suffix}.jpg"
            aug.save(os.path.join(dst, outname), quality=95)
            total += 1

    print(f"  {label}: {len(files)} originals -> {total} total "
          f"(x{len(VARIANTS)}: orig + mirror + 180 + mirror+180)")
    return total


print("Augmenting dataset (180° rotation and mirror only)...")
print()
n_good   = process_folder(SRC_GOOD,   DST_GOOD,   "GOOD")
n_defect = process_folder(SRC_DEFECT, DST_DEFECT, "DEFECT")
print()
print(f"Done!  good={n_good}  defect={n_defect}")
print(f"Saved to DataAI/augmented/")
print()
print("Next steps:")
print("  1. Label defect photos in labelme")
print("  2. python prepare_dataset.py")
print("  3. Press Retrain in the app")
