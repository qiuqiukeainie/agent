# C 组多模态文本与融合标签接口规范

本文档面向 C（多模态融合工程师），用于接入 OCR、字幕 OCR、Whisper ASR、人工文本等文本信号，并触发统一融合标签生成。

## 1. 文本类型约定

系统必须区分不同文本来源，避免音频转写和画面文字混在一起。

| text_type | 含义 | 典型来源 |
| --- | --- | --- |
| `ocr` | 图片或视频画面文字 | EasyOCR、PaddleOCR、Tesseract |
| `subtitle_ocr` | 视频字幕 OCR | 视频抽帧 + OCR |
| `asr` | 音频语音转写 | Whisper、faster-whisper |
| `manual_text` | 人工录入文本 | 人工备注、外部标注 |

推荐规则：
- 识别视频里的说话内容，写 `asr`。
- 识别画面上的字幕、牌匾、PPT、菜单、标题，写 `ocr` 或 `subtitle_ocr`。
- 不要把 ASR 结果伪装成 OCR，否则详情页和评测会混乱。

## 2. 单条写入接口

```http
POST /api/multimodal-text
Content-Type: application/json
```

```json
{
  "asset_id": "素材 ID",
  "text_type": "asr",
  "text": "识别出的文本内容",
  "engine": "whisper:faster_whisper",
  "confidence": 0.92,
  "auto_fuse": true
}
```

字段说明：
- `asset_id`：素材 ID。
- `text_type`：`ocr`、`subtitle_ocr`、`asr`、`manual_text`。
- `text`：识别出的文本。
- `engine`：模块来源，例如 `easyocr`、`paddleocr`、`whisper:faster_whisper`。
- `confidence`：0-1，可为空。
- `auto_fuse`：默认 `true`，写入后自动刷新融合标签。

## 3. 详情读取接口

```http
GET /api/asset?id=<asset_id>
```

关键返回字段：

```json
{
  "asset": {
    "id": "commons_text_Kara_Fern_-_Interview_The",
    "filename": "commons_text_Kara_Fern_-_Interview_The.webm",
    "asr_text": "音频转写合并文本",
    "visual_text": "画面 OCR / 字幕 OCR 合并文本",
    "ocr_text": "全部文本合并文本，兼容旧字段",
    "text_summary": {
      "asr": {
        "label": "音频转写",
        "count": 1,
        "summary": "摘要文本",
        "engines": ["whisper:faster_whisper"]
      },
      "visual": {
        "label": "画面/字幕 OCR",
        "count": 0,
        "summary": "",
        "engines": []
      },
      "manual": {
        "label": "人工文本",
        "count": 0,
        "summary": "",
        "engines": []
      }
    },
    "text_signals": [
      {
        "text_type": "asr",
        "engine": "whisper:faster_whisper",
        "confidence": null,
        "summary": "单条信号摘要",
        "text": "完整文本",
        "created_at": "2026-07-04T..."
      }
    ]
  }
}
```

E 组页面展示时优先使用：
- `text_summary.asr` 展示音频摘要。
- `text_summary.visual` 展示画面/字幕 OCR 摘要。
- `text_signals` 展示来源、引擎、置信度和单条摘要。
- `ocr_text` 只作为旧版本兼容字段。

## 4. 批量 CSV 导入

CSV 表头：

```csv
asset_id,text_type,text,engine,confidence
```

导入命令：

```powershell
.\.venv\Scripts\python.exe tools\import_multimodal_text.py data\my_text_signals.csv
```

## 5. OCR / ASR 抽取脚本

能力检测：

```powershell
.\.venv\Scripts\python.exe tools\extract_text_signals.py --capabilities
```

图片 OCR：

```powershell
.\.venv\Scripts\python.exe tools\extract_text_signals.py --library personal --kind image --ocr-engine easyocr --output data\personal_ocr.csv --import-signals
```

视频字幕 OCR，CPU 环境建议小批量：

```powershell
.\.venv\Scripts\python.exe tools\extract_text_signals.py --library public --kind video --offset 8 --limit 6 --ocr-engine easyocr --asr-engine none --video-frames 2 --min-confidence 0.45 --output data\easyocr_video_sample.csv --import-signals
```

视频音频 ASR：

```powershell
.\.venv\Scripts\python.exe tools\extract_text_signals.py --library public --kind video --filename-contains commons_text_ --limit 2 --ocr-engine none --asr-engine whisper --asr-model tiny --output data\asr_sample.csv --import-signals
```

低质量 OCR 清理：

```powershell
.\.venv\Scripts\python.exe tools\cleanup_text_signals.py --engine subtitle_ocr:easyocr --below 0.45
```

## 6. 后台任务方式

提交文本抽取任务：

```http
POST /api/process-assets
Content-Type: application/json
```

```json
{
  "actions": ["text_signals"],
  "library": "public",
  "kind": "video",
  "offset": 8,
  "limit": 6,
  "ocr_engine": "easyocr",
  "asr_engine": "none",
  "video_frames": 2,
  "min_confidence": 0.45
}
```

查询任务：

```http
GET /api/tasks
```

## 7. 与 A / D / E 的约定

- C 不直接操作 SQLite。
- C 通过 `/api/multimodal-text`、CSV 导入工具或后台任务提交文本信号。
- A 的检索重排读取融合标签和文本信号，ASR/OCR 都可以参与排序。
- D 的处理流水线负责批量调度缩略图、文本抽取、融合标签等任务。
- E 展示时必须区分 `asr_text` 与 `visual_text`，不要只展示合并后的 `ocr_text`。
