# ToolInspector

Десктопное приложение для поиска сколов на фотографиях режущего инструмента.
Работает локально, без интернета и внешних API.

## Как это работает

Двухступенчатый анализ:

1. **EfficientNet-B0** быстро классифицирует фото: *пригоден* / *брак*.
2. Если уверенность классификатора ниже порога (`YOLO_TRIGGER = 0.60` в `ai_analyzer.py`),
   фото передаётся в **YOLOv8n**, который находит точное положение сколов и рисует рамки.

Интерфейс: [customtkinter](https://github.com/TomSchimansky/CustomTkinter).

## Структура проекта

```
ToolInspector/
├── main.py               # точка входа
├── ui.py                 # интерфейс
├── ai_analyzer.py        # EfficientNet + YOLOv8: обучение и анализ
├── augment_dataset.py    # аугментация (зеркало и поворот на 180°)
├── prepare_dataset.py    # labelme JSON -> формат YOLO, обрезка фона
├── debug_dataset.py      # проверка разметки (рисует рамки на примерах)
├── requirements.txt
├── LICENSE
├── weights/
│   └── best.pt           # обученные веса YOLOv8n (детектор сколов)
└── training_results/     # метрики и графики одного из запусков обучения YOLOv8
```

Датасет в репозиторий **не входит**, он создаётся локально
в папке `DataAI/` (она в `.gitignore`):

```
DataAI/
├── good/      # фото годных инструментов
└── defect/    # фото с дефектами + разметка labelme (.json рядом с каждым фото)
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Для CPU-версии PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Подготовка данных и обучение

1. Положите фото годных инструментов в `DataAI/good/`, фото с дефектами в `DataAI/defect/`.
2. Разметьте сколы прямоугольниками в [labelme](https://github.com/wkentaro/labelme)
   (`pip install labelme`), JSON сохраняйте рядом с фото.
3. Аугментация для EfficientNet (по желанию):
   ```bash
   python augment_dataset.py
   ```
4. Датасет для YOLO:
   ```bash
   python prepare_dataset.py
   python debug_dataset.py      # проверить, что рамки легли правильно
   ```
5. Запустите приложение и нажмите **«Переобучить»**:
   ```bash
   python main.py
   ```
   Будут обучены обе модели, веса сохранятся в `DataAI/model_weights/`.

## Использование

1. Нажмите **«Выбрать фото»**.
2. Нажмите **«Запустить анализ»**.
3. Получите вердикт, подробное описание шагов и изображение с отмеченными сколами.

## Советы по съёмке

- Одинаковое расстояние до инструмента, зафиксированная камера.
- Один ракурс для всех фото.
- Равномерное освещение без бликов.
- Однородный светлый фон (обрезка фона в `prepare_dataset.py` рассчитана на тёмный инструмент на белом).

## Готовые веса YOLOv8

В `weights/best.pt` лежат веса детектора сколов, обученные на авторском датасете
(один класс: `chip`). Использовать их отдельно от приложения можно так:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict("photo.jpg", conf=0.35, iou=0.45, imgsz=1280)
results[0].save("result.jpg")
```

Чтобы использовать их внутри приложения, скопируйте файл в `DataAI/model_weights/best.pt`.
Приложению также нужен классификатор EfficientNet (`DataAI/model_weights/classifier.pth`),
он создаётся при обучении на вашем датасете (кнопка «Переобучить»), поэтому для полного
двухступенчатого анализа датасет понадобится.

## Лицензия

[MIT](LICENSE)

## Примечание

`yolov8n.pt` (предобученные веса YOLOv8) скачивается библиотекой `ultralytics`
автоматически при первом обучении.
