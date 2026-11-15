# ui.py - Interface for photo-based tool inspection (classifier version)

import os
import threading
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from ai_analyzer import analyze_photo, retrain, get_model_info, YOLO_TRIGGER

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG     = "#0f1117"
C_PANEL  = "#1a1d27"
C_CARD   = "#12151f"
C_ACCENT = "#3b82f6"
C_GREEN  = "#10b981"
C_RED    = "#ef4444"
C_YELLOW = "#f59e0b"
C_TEXT   = "#e2e8f0"
C_MUTED  = "#64748b"

FONT_MONO = "Courier New"
IMG_PREVIEW = 300
IMG_RESULT  = 420


class ToolInspectorApp:

    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("ToolInspector — Классификатор дефектов")
        self.window.geometry("1100x760")
        self.window.configure(fg_color=C_BG)
        self.window.resizable(True, True)

        self.photo_path   = None
        self._tk_preview  = None
        self._tk_result   = None
        self._result_path = None

        self._build_ui()
        self._refresh_status()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_model_bar()
        self._build_body()
        self._build_footer()

    def _build_header(self):
        hdr = ctk.CTkFrame(self.window, fg_color=C_PANEL,
                           corner_radius=0, height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="ToolInspector",
                     font=ctk.CTkFont(FONT_MONO, 22, "bold"),
                     text_color=C_ACCENT).pack(side="left", padx=24, pady=14)

        ctk.CTkLabel(hdr, text="EfficientNet  |  YOLOv8  |  без интернета",
                     font=ctk.CTkFont(size=12),
                     text_color=C_MUTED).pack(side="left", pady=14)

        self.status_lbl = ctk.CTkLabel(
            hdr, text="● Ожидание",
            font=ctk.CTkFont(size=12), text_color=C_MUTED)
        self.status_lbl.pack(side="right", padx=24)

    def _build_model_bar(self):
        bar = ctk.CTkFrame(self.window, fg_color="#0d1018",
                           corner_radius=0, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="МОДЕЛЬ:",
                     font=ctk.CTkFont(FONT_MONO, 10, "bold"),
                     text_color=C_MUTED).pack(side="left", padx=16, pady=12)

        self.model_lbl = ctk.CTkLabel(
            bar, text="Проверка...",
            font=ctk.CTkFont(FONT_MONO, 10), text_color=C_TEXT)
        self.model_lbl.pack(side="left", pady=12)

        # Retrain button
        self.btn_retrain = ctk.CTkButton(
            bar, text="↺  Переобучить",
            command=self._on_retrain,
            fg_color="#1e2235", hover_color=C_ACCENT,
            font=ctk.CTkFont(size=11, weight="bold"),
            height=28, width=140, corner_radius=6)
        self.btn_retrain.pack(side="right", padx=16, pady=8)

        ctk.CTkLabel(bar, text=f"порог YOLO < {int(YOLO_TRIGGER*100)}%",
                     font=ctk.CTkFont(FONT_MONO, 10),
                     text_color=C_MUTED).pack(side="right", padx=4)

    def _build_body(self):
        body = ctk.CTkFrame(self.window, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        self._build_left_panel(body)
        self._build_right_panel(body)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, fg_color=C_PANEL,
                            corner_radius=12, width=340)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Section: photo
        self._section(left, "ФОТО ИНСТРУМЕНТА")

        self.preview_frame = ctk.CTkFrame(
            left, fg_color=C_CARD, corner_radius=10,
            width=IMG_PREVIEW, height=IMG_PREVIEW)
        self.preview_frame.pack(padx=14, pady=(0, 8))
        self.preview_frame.pack_propagate(False)

        self.preview_lbl = ctk.CTkLabel(
            self.preview_frame,
            text="JPG · PNG · BMP",
            font=ctk.CTkFont(FONT_MONO, 12), text_color=C_MUTED)
        self.preview_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_load = ctk.CTkButton(
            left, text="Выбрать фото",
            command=self._on_load,
            fg_color=C_ACCENT, hover_color="#2563eb",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40, corner_radius=8)
        self.btn_load.pack(fill="x", padx=14, pady=(0, 4))

        self.fname_lbl = ctk.CTkLabel(
            left, text="", font=ctk.CTkFont(FONT_MONO, 10),
            text_color=C_MUTED)
        self.fname_lbl.pack(padx=14)

        # Section: info
        self._section(left, "ПАРАМЕТРЫ ФОТО", pady_top=12)

        self.info_box = ctk.CTkTextbox(
            left, fg_color=C_CARD, text_color=C_TEXT,
            font=ctk.CTkFont(FONT_MONO, 11),
            corner_radius=8, height=100, state="disabled")
        self.info_box.pack(fill="x", padx=14, pady=(0, 10))

        # Analyze button
        self.btn_analyze = ctk.CTkButton(
            left, text="Запустить анализ",
            command=self._on_analyze,
            fg_color=C_GREEN, hover_color="#059669",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=48, corner_radius=8, state="disabled")
        self.btn_analyze.pack(fill="x", padx=14, pady=(0, 16))

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color=C_PANEL, corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        # Verdict bar
        self._section(right, "РЕЗУЛЬТАТ")

        self.verdict_frame = ctk.CTkFrame(
            right, fg_color=C_CARD, corner_radius=10, height=72)
        self.verdict_frame.pack(fill="x", padx=14, pady=(0, 10))
        self.verdict_frame.pack_propagate(False)

        self.verdict_lbl = ctk.CTkLabel(
            self.verdict_frame,
            text="Загрузите фото и нажмите Анализ",
            font=ctk.CTkFont(FONT_MONO, 14, "bold"),
            text_color=C_MUTED)
        self.verdict_lbl.pack(expand=True)

        # Result image + details side by side
        mid = ctk.CTkFrame(right, fg_color="transparent")
        mid.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Result image column
        img_col = ctk.CTkFrame(mid, fg_color="transparent", width=IMG_RESULT+10)
        img_col.pack(side="left", fill="y", padx=(0, 10))
        img_col.pack_propagate(False)

        self._section(img_col, "ОБНАРУЖЕННЫЕ ДЕФЕКТЫ", pady_top=0)

        self.result_frame = ctk.CTkFrame(
            img_col, fg_color=C_CARD, corner_radius=10,
            width=IMG_RESULT, height=IMG_RESULT)
        self.result_frame.pack()
        self.result_frame.pack_propagate(False)

        self.result_img_lbl = ctk.CTkLabel(
            self.result_frame,
            text="Появится после анализа",
            font=ctk.CTkFont(FONT_MONO, 11), text_color=C_MUTED)
        self.result_img_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # Details column
        det_col = ctk.CTkFrame(mid, fg_color="transparent")
        det_col.pack(side="left", fill="both", expand=True)

        self._section(det_col, "ДЕТАЛИ", pady_top=0)

        self.detail_box = ctk.CTkTextbox(
            det_col, fg_color=C_CARD, text_color=C_TEXT,
            font=ctk.CTkFont(FONT_MONO, 12),
            corner_radius=8, wrap="word")
        self.detail_box.pack(fill="both", expand=True)
        self.detail_box.configure(state="disabled")

    def _build_footer(self):
        ftr = ctk.CTkFrame(self.window, fg_color=C_PANEL,
                           corner_radius=0, height=26)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)
        ctk.CTkLabel(
            ftr,
            text="ToolInspector  |  EfficientNet-B0 + YOLOv8n  |  без интернета",
            font=ctk.CTkFont(size=10), text_color=C_MUTED
        ).pack(side="left", padx=16)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_load(self):
        path = filedialog.askopenfilename(
            title="Выберите фото инструмента",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Все файлы", "*.*")
            ])
        if not path:
            return

        self.photo_path = path
        self._show_preview(path)
        self._show_info(_img_info(path))
        self._clear_result()
        self.btn_analyze.configure(state="normal")
        self._set_status("Фото загружено", C_GREEN)

        fname = os.path.basename(path)
        self.fname_lbl.configure(
            text=fname[:40] + "..." if len(fname) > 40 else fname)

    def _on_analyze(self):
        if not self.photo_path:
            return
        self._set_status("Анализ...", C_ACCENT)
        self.btn_analyze.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        threading.Thread(target=self._run_analyze, daemon=True).start()

    def _run_analyze(self):
        result = analyze_photo(self.photo_path)
        self.window.after(0, lambda: self._on_analyze_done(result))

    def _on_analyze_done(self, result):
        self.btn_analyze.configure(state="normal")
        self.btn_load.configure(state="normal")
        self._refresh_status()

        if result.get("error"):
            self._set_status("Ошибка", C_RED)
            self._set_verdict("Ошибка", None)
            self._show_detail(result["error"])
            return

        is_fit    = result["is_fit"]
        used_yolo = result.get("used_yolo", False)
        self._set_status(
            "Готово  [EfficientNet + YOLO]" if used_yolo else "Готово  [EfficientNet]",
            C_GREEN)
        self._set_verdict(result["verdict"], is_fit)
        self._show_detail(result.get("explanation", ""))

        rpath = result.get("heatmap_path")
        if rpath and os.path.exists(rpath):
            self._show_result_image(rpath)

    def _on_retrain(self):
        self.btn_retrain.configure(state="disabled", text="Обучение...")
        self.model_lbl.configure(text="Обучение модели...", text_color=C_YELLOW)
        threading.Thread(target=self._run_retrain, daemon=True).start()

    def _run_retrain(self):
        import ai_analyzer

        def on_progress(epoch, total, acc, loss):
            txt = (f"EfficientNet  эпоха {epoch}/{total}  "
                   f"точность {acc:.1f}%  потери {loss:.4f}")
            self.window.after(0, lambda t=txt: self.model_lbl.configure(
                text=t, text_color=C_YELLOW))

        # Attach progress callback to _load_effnet
        ai_analyzer._load_effnet._progress_cb = on_progress
        result = retrain()
        ai_analyzer._load_effnet._progress_cb = None
        self.window.after(0, lambda: self._on_retrain_done(result))

    def _on_retrain_done(self, result):
        self.btn_retrain.configure(state="normal", text="↺  Переобучить")
        if result.get("error"):
            self.model_lbl.configure(
                text=f"Ошибка: {result['error'][:70]}", text_color=C_RED)
        else:
            self._refresh_status()
            r        = result.get("results", {})
            eff_acc  = r.get("effnet", {}).get("accuracy", None)
            yolo_map = r.get("yolo",   {}).get("map50",    None)
            parts    = []
            if eff_acc  is not None: parts.append(f"EfficientNet {eff_acc}%")
            if yolo_map is not None: parts.append(f"YOLO mAP50={yolo_map}%")
            if parts:
                self.model_lbl.configure(
                    text="Обучено: " + "  |  ".join(parts),
                    text_color=C_GREEN)

    # ── UI helpers ─────────────────────────────────────────────────────────────

    def _section(self, parent, text, pady_top=16):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(FONT_MONO, 10, "bold"),
            text_color=C_MUTED
        ).pack(anchor="w", padx=18, pady=(pady_top, 4))

    def _set_status(self, text, color):
        self.status_lbl.configure(text=f"● {text}", text_color=color)

    def _set_verdict(self, text, is_fit):
        if is_fit is True:
            color, icon = C_GREEN, "✓"
        elif is_fit is False:
            color, icon = C_RED, "✗"
        else:
            color, icon = C_YELLOW, "?"
        self.verdict_lbl.configure(
            text=f"{icon}  {text}", text_color=color)

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((IMG_PREVIEW, IMG_PREVIEW), Image.LANCZOS)
            self._tk_preview = ctk.CTkImage(
                light_image=img, dark_image=img,
                size=(img.width, img.height))
            self.preview_lbl.configure(image=self._tk_preview, text="")
        except Exception as e:
            self.preview_lbl.configure(text=f"Ошибка: {e}", image=None)

    def _show_result_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((IMG_RESULT, IMG_RESULT), Image.LANCZOS)
            self._tk_result = ctk.CTkImage(
                light_image=img, dark_image=img,
                size=(img.width, img.height))
            self.result_img_lbl.configure(image=self._tk_result, text="")
        except Exception as e:
            self.result_img_lbl.configure(text=f"Ошибка: {e}", image=None)

    def _show_info(self, text):
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("end", text)
        self.info_box.configure(state="disabled")

    def _show_detail(self, text):
        self.detail_box.configure(state="normal")
        self.detail_box.delete("1.0", "end")
        self.detail_box.insert("end", text)
        self.detail_box.configure(state="disabled")

    def _clear_result(self):
        self.verdict_lbl.configure(
            text="Анализируем...", text_color=C_MUTED)
        self._show_detail("")
        self.result_img_lbl.configure(image=None, text="Появится после анализа")
        self._tk_result = None

    def _refresh_status(self):
        info  = get_model_info()
        color = C_GREEN if "✓" in info else C_MUTED
        self.model_lbl.configure(text=info, text_color=color)

    def run(self):
        self.window.mainloop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_info(path: str) -> str:
    try:
        img  = Image.open(path)
        size = os.path.getsize(path)
        return (f"Файл:    {os.path.basename(path)}\n"
                f"Размер:  {img.width} × {img.height} px\n"
                f"Режим:   {img.mode}\n"
                f"Размер:  {size // 1024} КБ")
    except Exception as e:
        return f"Ошибка: {e}"
