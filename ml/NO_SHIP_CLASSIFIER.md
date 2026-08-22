# License: do not ship a classifier trained on REHAB24-6

REHAB24-6 ([Zenodo 13305826](https://zenodo.org/records/13305826)) is **CC BY-NC 4.0**.

Allowed in this repo:

- Run a **frozen** Google Pose Landmarker on the RGB videos.
- Score rep counting and pose error against `Segmentation.csv` and mocap `.npy`.
- Calibrate **rule thresholds** (hysteresis, angle cutoffs).

Not allowed for a commercial/shipped APK:

- Training or fine-tuning pose weights on this set.
- Shipping a GRU, 1D-CNN, DTW prototype, or any other **learned head** whose parameters were fit on REHAB24-6 labels or features.

DTW/kNN **without fitting a shipped model file** can stay as offline research in a notebook; it is not part of this bench and must not be bundled as product ML.

Collect a separate, license-clean labeled set before any on-device quality classifier.
