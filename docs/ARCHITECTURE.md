# Implementation map

This is original application code informed by the reference repository's published pipeline.

| Reference capability | Local implementation |
|---|---|
| Product detector | YOLOv12-compatible ONNX/OpenVINO adapter and NMS |
| SKU embeddings + FAISS | Shared preprocessing, model fingerprints, normalized IndexFlatL2, unknown rejection |
| Reference catalog growth | Complete local reindex, product upload UI, restart/reload on source start |
| Shelf alignment | ORB matches + RANSAC homography + coverage/inlier checks |
| Grid/planogram | Row grouping, editable normalized boxes, named plans |
| Tracking | Kalman box/velocity state with Hungarian IoU assignment |
| Video pipeline | Capture worker and analysis worker with latest-frame buffer |
| Temporal consensus | State AND observed identity streak + elapsed time; unknown resets evidence |
| Alerts | SQLite active uniqueness per shelf/slot, acknowledgment, resolution and CSV export |
| Management interface | Local HTML/CSS/JS dashboard and FastAPI |
| Hardware export | ONNX CPU default; optional OpenVINO; TensorRT is a future deployment choice |

Index operations and model changes require monitoring to stop. API mutations share an operation lock.
Only the analysis worker owns the active model instances. Read-only frame/status endpoints return snapshots.
Each source start creates fresh tracking and temporal state. Capture sequence IDs prevent re-counting an
identical buffered frame as fresh evidence. Periodic inference still processes newly captured static frames
so a persistent empty position can be confirmed. A single uploaded image never updates alert history.

Model validation uses manifest fingerprints and dimensions rather than mere file existence. Inference failures
stop real monitoring. Alignment failures clear temporal evidence and suppress alerts. Unrecognized objects
occupy their position as UNKNOWN, rather than being treated as missing or forced into an existing SKU.

Reference detection generates one slot per visible product facing. These slots must be reviewed. No module
infers invisible stock behind products. Source selection is operator-controlled; use one saved plan per camera
view and avoid reusing a plan for a different view. Alignment requires a matching saved reference image.

The default deployment is a trusted user's localhost process. There is no public hosting, user authentication,
multi-tenant isolation, cloud upload, or background service installation. Files and camera sources are explicitly
selected by the user. Do not expose the server to a network without adding appropriate access controls.

