# ai_analyzer.py - Dual pipeline: EfficientNet -> YOLOv8
#
# Step 1: EfficientNet-B0 classifies the photo (fast, ~0.1s)
#   - confidence >= 75% -> return result immediately
#   - confidence <  75% -> pass to YOLOv8 for precise analysis
#
# Step 2 (only if needed): YOLOv8 finds exact chip locations

import os
import json
import shutil
import numpy as np
from PIL import Image

DATA_DIR    = "DataAI"
AUG_DIR     = os.path.join(DATA_DIR, "augmented")
GOOD_DIR    = os.path.join(DATA_DIR, "good")
DEFECT_DIR  = os.path.join(DATA_DIR, "defect")
MODEL_DIR   = os.path.join(DATA_DIR, "model_weights")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
EFFNET_PATH = os.path.join(MODEL_DIR, "classifier.pth")
YOLO_PATH   = os.path.join(MODEL_DIR, "best.pt")
INFO_PATH   = os.path.join(MODEL_DIR, "info.json")
YAML_PATH   = os.path.join(DATA_DIR,  "yolo_dataset", "dataset.yaml")

IMG_SIZE     = 224
EFFNET_CONF  = 0.50   # EfficientNet decision threshold
YOLO_TRIGGER = 0.60   # if EfficientNet confidence < 60% -> run YOLO
YOLO_CONF    = 0.35
YOLO_IOU     = 0.45

_effnet_cache = None
_yolo_cache   = None


# ── Public ────────────────────────────────────────────────────────────────────

def analyze_photo(image_path: str) -> dict:
    global _effnet_cache, _yolo_cache

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: EfficientNet ──────────────────────────────────────────────────
    if _effnet_cache is None:
        r = _load_effnet()
        if r.get("error"):
            return r
        _effnet_cache = r["model"]

    try:
        eff_score, eff_conf, eff_is_fit = _predict_effnet(_effnet_cache, image_path)
    except Exception as e:
        return {"error": f"EfficientNet error: {e}"}

    used_yolo   = False
    yolo_detections = []
    result_path = None

    # ── Step 2: YOLO only if EfficientNet is not confident enough ─────────────
    if eff_conf < YOLO_TRIGGER:
        # Try to load YOLO
        if _yolo_cache is None and os.path.exists(YOLO_PATH):
            r2 = _load_yolo()
            if not r2.get("error"):
                _yolo_cache = r2["model"]

        if _yolo_cache is not None:
            try:
                yolo_detections, result_path = _predict_yolo(
                    _yolo_cache, image_path)
                used_yolo = True
            except Exception as e:
                pass   # YOLO failed — fall back to EfficientNet result

    # ── Combine results ───────────────────────────────────────────────────────
    if used_yolo:
        n      = len(yolo_detections)
        is_fit = (n == 0)

        if n == 0:
            # YOLO found nothing -> trust it: tool is good
            verdict = f"ПРИГОДЕН  (EfficientNet {eff_conf:.0%} → уточнено YOLO)"
            conf_pct = round((1 - eff_score) * 100, 1) if not eff_is_fit else round(eff_score * 100 if not eff_is_fit else (1-eff_score)*100, 1)
        else:
            avg_c   = sum(d["conf"] for d in yolo_detections) / n
            verdict = f"БРАК — {n} скол(а/ов)  (YOLO {avg_c:.0f}%)"

        explanation = _explanation_dual(
            eff_score, eff_conf, eff_is_fit,
            yolo_detections, used_yolo=True)
    else:
        is_fit  = eff_is_fit
        conf_str= f"{eff_conf:.0%}"
        verdict = (f"ПРИГОДЕН  (уверенность {conf_str})"
                   if is_fit else f"БРАК  (уверенность {conf_str})")

        result_path = _draw_effnet(image_path, eff_score, is_fit)
        explanation = _explanation_dual(
            eff_score, eff_conf, eff_is_fit,
            yolo_detections, used_yolo=False)

    return {
        "verdict":      verdict,
        "is_fit":       is_fit,
        "explanation":  explanation,
        "heatmap_path": result_path,
        "score":        round(eff_score, 4),
        "defect_count": len(yolo_detections) if used_yolo else (0 if is_fit else 1),
        "used_yolo":    used_yolo,
        "error":        None,
    }


def retrain() -> dict:
    global _effnet_cache, _yolo_cache
    _effnet_cache = None
    _yolo_cache   = None
    results = {}

    # Train EfficientNet
    r1 = _train_effnet()
    results["effnet"] = r1

    # Train YOLO if dataset exists
    if os.path.exists(YAML_PATH):
        r2 = _train_yolo()
        results["yolo"] = r2
    else:
        results["yolo"] = {"info": "YOLO dataset not prepared. Run prepare_dataset.py first."}

    if results["effnet"].get("error"):
        return results["effnet"]
    return {"ok": True, "results": results}


def get_model_info() -> str:
    eff_ok  = os.path.exists(EFFNET_PATH)
    yolo_ok = os.path.exists(YOLO_PATH)
    parts   = []
    if eff_ok:
        parts.append("EfficientNet ✓")
    else:
        parts.append("EfficientNet — не обучен")
    if yolo_ok:
        parts.append("YOLOv8 ✓")
    else:
        parts.append("YOLOv8 — не обучен")
    parts.append(f"порог переключения {YOLO_TRIGGER:.0%}")
    return "  |  ".join(parts)


# ── EfficientNet ──────────────────────────────────────────────────────────────

def _load_effnet() -> dict:
    good_src   = os.path.join(AUG_DIR,"good")   if os.path.isdir(os.path.join(AUG_DIR,"good"))   else GOOD_DIR
    defect_src = os.path.join(AUG_DIR,"defect") if os.path.isdir(os.path.join(AUG_DIR,"defect")) else DEFECT_DIR

    if _count(good_src) == 0:
        return {"error": "DataAI/good/ пуста.\nЗапустите: python augment_dataset.py"}
    if _count(defect_src) == 0:
        return {"error": "DataAI/defect/ пуста.\nДобавьте дефектные фото."}

    if os.path.exists(EFFNET_PATH):
        try:
            import torch
            model = _build_effnet()
            model.load_state_dict(torch.load(EFFNET_PATH, map_location="cpu"))
            model.eval()
            return {"model": model}
        except Exception:
            pass

    return _train_effnet()


def _train_effnet() -> dict:
    try:
        import torch, torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
        import torchvision.transforms as T
    except ImportError:
        return {"error": "pip install torch torchvision"}

    good_src   = os.path.join(AUG_DIR,"good")   if os.path.isdir(os.path.join(AUG_DIR,"good"))   else GOOD_DIR
    defect_src = os.path.join(AUG_DIR,"defect") if os.path.isdir(os.path.join(AUG_DIR,"defect")) else DEFECT_DIR
    n_good, n_def = _count(good_src), _count(defect_src)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tfm_tr = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(),
                         T.RandomRotation(10), T.ColorJitter(0.2,0.15),
                         T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
    tfm_vl = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])

    all_imgs = ([(p,0) for p in _list_images(good_src)] +
                [(p,1) for p in _list_images(defect_src)])
    import random as rnd; rnd.seed(42); rnd.shuffle(all_imgs)
    split  = int(len(all_imgs)*0.8)
    train, val = all_imgs[:split], all_imgs[split:]

    class DS(torch.utils.data.Dataset):
        def __init__(self,items,tfm): self.items,self.tfm=items,tfm
        def __len__(self): return len(self.items)
        def __getitem__(self,i):
            path,label=self.items[i]
            return self.tfm(Image.open(path).convert("RGB")), label

    labels  = [l for _,l in train]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1./counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    tr_dl = DataLoader(DS(train,tfm_tr), batch_size=16, sampler=sampler, num_workers=0)
    vl_dl = DataLoader(DS(val,  tfm_vl), batch_size=16, shuffle=False,  num_workers=0)

    model  = _build_effnet()
    crit   = nn.BCEWithLogitsLoss()
    optim  = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=25)
    EPOCHS = 25
    best_acc = 0.0
    cb = getattr(_load_effnet, "_progress_cb", None)

    print(f"\n{'='*50}\n  EfficientNet обучение  |  норма:{n_good}  дефект:{n_def}\n{'='*50}")
    print(f"  {'Эпоха':>6}  {'Loss':>8}  {'Точность':>9}")

    for ep in range(EPOCHS):
        model.train()
        tot_loss, nb = 0.0, 0
        for imgs, lbs in tr_dl:
            optim.zero_grad()
            loss = crit(model(imgs).squeeze(1), lbs.float())
            loss.backward(); optim.step()
            tot_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbs in vl_dl:
                preds   = (torch.sigmoid(model(imgs).squeeze(1)) > EFFNET_CONF).long()
                correct += (preds==lbs).sum().item(); total += len(lbs)
        acc = round(correct/max(total,1)*100,1)
        mark = " ★" if acc > best_acc else ""
        print(f"  {ep+1:>6}/{EPOCHS}  {tot_loss/nb:>8.4f}  {acc:>8.1f}%{mark}", flush=True)
        if cb: cb(ep+1, EPOCHS, acc, tot_loss/nb)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), EFFNET_PATH)

    print(f"\n  Готово! Точность EfficientNet: {best_acc}%\n{'='*50}\n")
    model.load_state_dict(torch.load(EFFNET_PATH, map_location="cpu"))
    model.eval()

    with open(INFO_PATH,"w") as f:
        json.dump({"effnet_acc": best_acc, "good": n_good, "defect": n_def}, f)

    global _effnet_cache
    _effnet_cache = model
    return {"model": model, "accuracy": best_acc}


def _build_effnet():
    import torch.nn as nn
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(pretrained=True)
    m.classifier[1] = __import__("torch").nn.Linear(m.classifier[1].in_features, 1)
    return m


def _predict_effnet(model, image_path: str):
    import torch
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
                     T.Normalize([.485,.456,.406],[.229,.224,.225])])
    img   = Image.open(image_path).convert("RGB")
    inp   = tfm(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        score = float(torch.sigmoid(model(inp).squeeze()))
    is_fit = score < EFFNET_CONF
    conf   = abs(score - 0.5) * 2   # 0..1, how far from decision boundary
    return score, conf, is_fit


def _draw_effnet(image_path: str, score: float, is_fit: bool) -> str:
    import cv2
    orig   = np.array(Image.open(image_path).convert("RGB"))
    H, W   = orig.shape[:2]
    result = cv2.cvtColor(orig.copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(result, (0,0), (W,52), (0,0,0), -1)
    col = (0,200,80) if is_fit else (0,50,255)
    txt = f"ПРИГОДЕН  score={score:.3f}" if is_fit else f"БРАК  score={score:.3f}"
    cv2.putText(result, txt, (14,34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    out = os.path.join(RESULTS_DIR, "eff_" + os.path.basename(image_path))
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(out)
    return out


# ── YOLOv8 ───────────────────────────────────────────────────────────────────

def _load_yolo() -> dict:
    if not os.path.exists(YOLO_PATH):
        return {"error": "YOLO model not found"}
    try:
        from ultralytics import YOLO
        return {"model": YOLO(YOLO_PATH)}
    except ImportError:
        return {"error": "pip install ultralytics"}
    except Exception as e:
        return {"error": str(e)}


def _train_yolo() -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"error": "pip install ultralytics"}

    if not os.path.exists(YAML_PATH):
        return {"error": "Run prepare_dataset.py first"}

    print(f"\n{'='*50}\n  YOLOv8n обучение\n{'='*50}")
    model   = YOLO("yolov8n.pt")
    results = model.train(
        data=YAML_PATH, epochs=100, imgsz=1280, batch=2,
        patience=30, device="cpu", project=MODEL_DIR, name="train",
        exist_ok=True, verbose=True,
        conf=YOLO_CONF, iou=YOLO_IOU, augment=True,
        degrees=15, flipud=0.5, fliplr=0.5,
        hsv_v=0.5, hsv_s=0.5, scale=0.5,
        mosaic=0.0, cls=0.3, box=9.0, lr0=0.001, lrf=0.001,
    )

    best_src = os.path.join(MODEL_DIR,"train","weights","best.pt")
    if not os.path.exists(best_src):
        import glob
        candidates = glob.glob(os.path.join("runs","detect","**","best.pt"), recursive=True)
        if candidates: best_src = sorted(candidates)[-1]
    if os.path.exists(best_src):
        shutil.copy(best_src, YOLO_PATH)

    try:
        map50 = round(float(results.results_dict.get("metrics/mAP50(B)",0))*100,1)
    except Exception:
        map50 = "?"
    print(f"\n  Готово! YOLOv8 mAP50={map50}%\n{'='*50}\n")

    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f: info = json.load(f)
    else:
        info = {}
    info["yolo_map50"] = map50
    with open(INFO_PATH,"w") as f: json.dump(info, f)

    global _yolo_cache
    _yolo_cache = YOLO(YOLO_PATH)
    return {"model": _yolo_cache, "map50": map50}


def _predict_yolo(model, image_path: str):
    import cv2
    results = model.predict(source=image_path, conf=YOLO_CONF,
                            iou=YOLO_IOU, verbose=False, imgsz=1280)
    orig    = np.array(Image.open(image_path).convert("RGB"))
    H, W    = orig.shape[:2]
    result  = cv2.cvtColor(orig.copy(), cv2.COLOR_RGB2BGR)
    detections = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf         = round(float(box.conf[0])*100, 1)
            detections.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})

    for i, d in enumerate(detections):
        x1,y1,x2,y2 = d["x1"],d["y1"],d["x2"],d["y2"]
        ov = result.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,220),-1)
        cv2.addWeighted(ov,0.28,result,0.72,0,result)
        cv2.rectangle(result,(x1,y1),(x2,y2),(0,0,255),3)
        s = min(16,(x2-x1)//3,(y2-y1)//3)
        for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            sx = s if px==x1 else -s
            sy = s if py==y1 else -s
            cv2.line(result,(px,py),(px+sx,py),(0,0,255),3)
            cv2.line(result,(px,py),(px,py+sy),(0,0,255),3)
        lbl = f"Скол {i+1}  {d['conf']:.0f}%"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        lx = x1; ly = y1-10 if y1>th+16 else y2+th+10
        cv2.rectangle(result,(lx-2,ly-th-4),(lx+tw+4,ly+4),(0,0,0),-1)
        cv2.putText(result,lbl,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,80,255),2,cv2.LINE_AA)

    cv2.rectangle(result,(0,0),(W,52),(0,0,0),-1)
    n = len(detections)
    if n == 0:
        txt,col = "YOLO: сколов не найдено",(0,200,80)
    else:
        avg = sum(d["conf"] for d in detections)/n
        txt,col = f"YOLO БРАК: {n} скол(а/ов)  ср.уверенность {avg:.0f}%",(0,50,255)
    cv2.putText(result,txt,(14,34),cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2,cv2.LINE_AA)

    out = os.path.join(RESULTS_DIR,"dual_"+os.path.basename(image_path))
    Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)).save(out)
    return detections, out


# ── Explanation ───────────────────────────────────────────────────────────────

def _explanation_dual(eff_score, eff_conf, eff_is_fit,
                       yolo_dets, used_yolo: bool) -> str:
    L = ["СДВОЕННЫЙ АНАЛИЗ  EfficientNet + YOLOv8", ""]
    L.append(f"Шаг 1 — EfficientNet-B0:")
    L.append(f"  Оценка дефектности: {eff_score:.4f}")
    verdict1 = "ПРИГОДЕН" if eff_is_fit else "ДЕФЕКТ"
    L.append(f"  Вердикт:            {verdict1}")
    L.append(f"  Уверенность:        {eff_conf:.0%}")

    if used_yolo:
        L.append(f"  → Уверенность ниже {YOLO_TRIGGER:.0%} — передано в YOLOv8")
        L.append("")
        L.append(f"Шаг 2 — YOLOv8n (точная локализация):")
        if not yolo_dets:
            L.append("  Сколов не обнаружено.")
            L.append("  Итоговый вердикт: ПРИГОДЕН")
        else:
            L.append(f"  Найдено сколов: {len(yolo_dets)}")
            for i,d in enumerate(yolo_dets,1):
                L.append(f"    Скол {i}: ({d['x1']},{d['y1']})–({d['x2']},{d['y2']})"
                         f"  {d['conf']:.0f}%")
            L.append("  Итоговый вердикт: БРАК")
    else:
        L.append(f"  → Уверенность ≥ {YOLO_TRIGGER:.0%} — YOLOv8 не требуется")
        L.append("")
        L.append("Шаг 2 — пропущен (достаточно уверенности EfficientNet)")

    L += ["",
          f"Порог переключения на YOLO: {YOLO_TRIGGER:.0%}",
          "Логика: быстрый EfficientNet + точный YOLOv8 при сомнении"]
    return "\n".join(L)


# ── helpers ───────────────────────────────────────────────────────────────────

def _count(folder):
    if not os.path.isdir(folder): return 0
    exts={".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in exts)

def _list_images(folder):
    exts={".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
    return [os.path.join(folder,f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts]
