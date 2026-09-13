# Proglint Shelfwatch

A local retail shelf-monitoring application built with Python, FastAPI, OpenCV, ONNX Runtime,
MobileNetV3 and FAISS. It detects visible products, matches them against a reference catalog,
compares shelf positions with a planogram, and records persistent missing/misplaced alerts.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Open **http://127.0.0.1:8765**. Python 3.12 is recommended. The project uses an isolated `.venv`.
VS Code also includes a **Shelfwatch dashboard** launch configuration. Ctrl+C stops a foreground server.

## Try your supplied models

1. Open **Models & setup** and click **Validate model bundle**.
2. In **Shelf monitor**, click **Supplied shelf photo** to run both models on the supplied scene.
3. Click **Run supplied video** to process the provided video and compare it with its supplied planogram.
4. Review positions and open **Alert history** for persistent changes. The imported shelf is named `sample`.

The supplied scene is synthetic, assembled from product images. These buttons use your real model exports.
The separate **Synthetic demo** mode uses scripted detections to demonstrate the interface without models.

## Your model files

The current configuration loads these files directly from `artifacts/colab/`:

```text
yolov12m_sku110k.onnx             Your trained YOLO detector
mobilenetv3_embedding.onnx        Your MobileNetV3 embedding graph
mobilenetv3_embedding.onnx.data   External embedding weights; keep beside the graph
embedding_preprocessing.json     Input normalization settings
sku_index.faiss                  Normalized cosine-similarity reference index
sku_mapping.json                 Index-position to SKU mapping
thresholds.json                  Matching rules and unknown-product rejection threshold
best.pt                         Original training checkpoint, retained unchanged
```

Model weights are not retrained or downloaded during application startup. The runtime reproduces the supplied
notebook's MobileNetV3 preprocessing: Pillow bilinear resize of the shorter side to 232, center crop to 224,
RGB conversion, ImageNet normalization and L2-normalized embeddings. Recognition retrieves five neighbors,
selects the SKU by majority vote then mean similarity, and applies the supplied similarity threshold.

The runtime retains support for the original generic ONNX/L2 bundle format through configuration.
[The original export guide](docs/COLAB_EXPORT.md) describes that alternative format; your current bundle
already uses the adapter above and does not need to be re-exported.

## Reference products and new SKUs

Your existing catalog is loaded from:

`artifacts/colab/data/references/<SKU ID>/`

For example, `sku_001/000.jpg`. Names with letters, digits, underscores and hyphens are supported.
Stop monitoring, open **Products**, add a product and reference images, then choose **Rebuild reference index**.
The entire local reference catalog is included. The previous index/mapping are backed up before replacement.
No YOLO retraining is needed to register a product. Review unknown rejection after changing the catalog.

## Your own shelf or camera

1. Analyze a correctly stocked shelf photo under a new shelf name.
2. In **Planogram**, generate positions from the last frame, review expected SKUs and save.
3. Choose a video file, upload a video, or enter a camera index such as `0` or an RTSP URL.
4. Start monitoring with the same shelf name.

Unknown products remain unknown. Missing/misplaced states require repeated observations over time before
creating alerts. Acknowledging an alert records review; a stable corrected shelf resolves it. Single-image
analysis displays results but does not create persistence-based alerts.

## Features

- Product detection and SKU recognition with replaceable local model exports.
- Product registration and full-catalog FAISS index rebuilding.
- Named planograms, editable positions and one-to-one detection assignment.
- Kalman/Hungarian tracking, latest-frame capture and periodic/keyframe inference.
- Optional ORB/RANSAC alignment with alert suppression when alignment fails.
- SQLite alert history, acknowledgment, resolution, JSON results and CSV exports.
- Local browser dashboard, VS Code launch settings and automated tests.

## Project layout

```text
shelfwatch/                  API, inference, recognition adapters, tracking and storage
web/                         Dashboard HTML, CSS and JavaScript
artifacts/colab/              Model bundle and original evaluation files
artifacts/colab/data/         Supplied references, validation images and sample media
data/planograms/             Imported and edited shelf plans/reference images
data/media/                  Videos uploaded through the dashboard
data/runtime/                SQLite database
outputs/                     Runtime logs and verification outputs
scripts/                     Windows setup/start and optional export helpers
tests/                       Automated behavior and integration tests
```

Paths and timing settings live in `config.yaml`; restart the server after changing them.

```powershell
.\.venv\Scripts\python.exe -m shelfwatch.cli doctor
.\.venv\Scripts\python.exe -m pytest
```

The app listens only on localhost and supports one active monitoring source per server. ONNX Runtime uses
CPU inference by default; optional OpenVINO requires its package and compatible exports.

## Reference and scope

Architecture reference: [Alijanloo/Retail-Shelf-Monitoring](https://github.com/Alijanloo/Retail-Shelf-Monitoring).
The local application is independently written, using a browser dashboard and SQLite for straightforward setup.
Your own supplied YOLO and MobileNet weights are used; no unpublished author checkpoint is included.

Alerts describe visible shelf positions, not hidden stock or warehouse inventory. Inspect results with your
actual camera, product assortment and lighting before relying on them. Original evaluation files are preserved.
SKU-110K's source terms specify academic/noncommercial use; check data/model licenses for your intended use.
