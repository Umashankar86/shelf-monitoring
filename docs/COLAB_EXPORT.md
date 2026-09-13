# Colab → local application contract

Train in Colab while running the local UI here. The application defaults to ONNX Runtime CPU inference;
you do not need PyTorch, a GPU, PostgreSQL, or Redis to run the dashboard locally.

## Output location

Copy your final bundle to:

`C:\Users\Umashankar\Proglint\artifacts\colab\`

Copy the **entire reference image catalog** to `data/references/` if you want to add products locally later.
The application rebuilds from all local reference folders, not from the original Colab file paths.

## Export cells after YOLO training

Upload `scripts/colab_export.py` into your Colab working directory. In the same runtime where your trained
YOLO checkpoint loads correctly, install the export dependencies:

```python
%pip install onnx onnxruntime faiss-cpu opencv-python-headless
```

Keep the YOLOv12 author's installed implementation; do not replace it with another Ultralytics package while
loading that checkpoint. Its current and earlier architectures can differ.

```python
from colab_export import export_models, build_reference_index, calibrate_threshold

BUNDLE = '/content/drive/MyDrive/shelfwatch_bundle'
export_models('/content/path/to/weights/best.pt', BUNDLE)
```

This exports your detector and an ImageNet-pretrained MobileNetV3-Large pooled feature model. It does not
fine-tune MobileNet. If you already fine-tuned an embedding model, pass that model as `embedding_model=`;
it must output `[batch, features]` and use the documented preprocessing. Do not pass a classification head
or reuse an index built using a different embedding model.

```python
build_reference_index('/content/references', BUNDLE)
```

Reference structure: `/content/references/101/*.jpg`, `/content/references/102/*.jpg`, etc. Include multiple
views per SKU. All embedding extraction uses the **exported ONNX model**, ensuring the local app agrees.

```python
report = calibrate_threshold(
    known_dir='/content/validation_known',
    unknown_dir='/content/validation_unknown',
    output_dir=BUNDLE,
    max_unknown_acceptance=0.05,
)
print(report)
```

Known validation folders use registered numeric SKU IDs. Unknown folders contain product crops absent
from the catalog. The function requires at least five images of each kind as a basic check; realistic
calibration needs substantially more varied examples. Keep these images separate from reference/training
images. Inspect the accepted-correct rate: a low rate means the embedding model/reference catalog needs work.
The reported calibration performance is not an unbiased test result; evaluate again on independent images.

```python
import shutil
shutil.make_archive('/content/shelfwatch_bundle', 'zip', BUNDLE)
from google.colab import files
files.download('/content/shelfwatch_bundle.zip')
```

Extract the contents directly into `artifacts/colab/` (avoid an extra nested directory). Put `best.pt` there
as an optional backup. Then select **Models & setup → Validate model bundle** in the local app.

## Exact contracts

Detector: float32 RGB NCHW, square letterbox with padding 114, scale pixels by 1/255. Export batch 1,
`nms=False`, `half=False`, opset 17. Output raw YOLO `[1,4+classes,N]`, product-only class preferred.
The local decoder performs NMS and reverses letterbox coordinates. NMS-export `[1,N,6]` is also supported.

Embedding: float32 RGB NCHW; OpenCV linear resize directly to 224×224; divide by 255; normalize with
mean `[.485,.456,.406]`, std `[.229,.224,.225]`. Output `[1,D]`, followed by L2 normalization outside the model.
This deliberately uses a single specified transform rather than torchvision's default resize/center-crop.

Index: `faiss.IndexFlatL2`, containing normalized float32 vectors. Its distances are **squared L2**, not
probabilities. The threshold must use the same metric. SKU labels use Unicode or integer NumPy arrays
without pickled objects. The code does not load reference path files to perform inference.

Metadata: SHA-256 fingerprints of the embedding model, preprocessing JSON, index, and labels prevent
mixing unrelated exports. Do not hand-edit these hashes to bypass validation; regenerate the bundle.

## New references later

Add reference images in the local app, rebuild the full reference index, then restart monitoring. No YOLO
retraining is required merely to register a new SKU. Review recognition and rejection quality after catalog
changes. If you change the embedding model or preprocessing, regenerate every reference embedding and
recalibrate the threshold. For materially different shelf imagery, consider additional detector fine-tuning.

