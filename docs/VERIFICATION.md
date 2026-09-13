# Local verification

Checked on Windows with Python 3.12.10. `requirements-lock.txt` records the installed dependency versions.

- 22 automated tests passed.
- Ruff lint and Python formatting checks passed.
- Dashboard JavaScript parsed successfully.
- The running server returned HTTP 200 for `/`, `/health`, and `/api/status`.
- An empty `artifacts/colab/` is correctly reported as not ready for real monitoring.

Behavior covered: YOLO coordinate reversal/NMS, unknown vs empty positions, one-to-one slot assignment,
product swaps, tracking without ghost stock, time/count filtering, persistent alert deduplication and resolution,
invalid alignment rejection, SKU upload/path validation, real ONNX inference and FAISS matching with small
test graphs, adding a SKU without altering model weights, mismatched artifact rejection, agreement between
Colab helper and local preprocessing, single-image analysis without persistent alerts, and synthetic demo alerts.

Tests create their own tiny deterministic ONNX models in temporary folders. Those models are not product
detectors and are not included in the application artifact folder.

Outstanding validation: the browser connection was unavailable, so the dashboard has not been visually
inspected through browser automation. Your trained YOLO/MobileNet exports, actual webcam/RTSP hardware,
real shelf footage, and optional OpenVINO inference remain to be checked when available. Colab's Torch-to-ONNX
export itself was not executed locally; the ONNX/index contract and shared preprocessing were tested.

The test runner emits two dependency deprecation warnings from Starlette/httpx/AnyIO; they do not fail the tests.

