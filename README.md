Whole-Body 86-Keypoint Extraction Pipeline (wb86)

Overview
- Extracts 86 OpenPose-style keypoints without training: BODY_25 + 21 left hand + 21 right hand + 19 face.
- Uses pretrained detectors (YOLOv8/RTMDet) and MMPose whole-body and region refiners.
- Designed for ~256×256 inputs; applies optional upscaling, CLAHE, ROI refinement, and One-Euro smoothing.

Install (base)
- Python 3.9–3.12
- pip install -e .

Optional extras
- Detection: pip install 'wb86-pose-pipeline[detect]'
- WholeBody/Refine: pip install 'wb86-pose-pipeline[wholebody]'

Quickstart
- Single image: wb86 --input path/to/image.jpg --out out/ --visualize
- Directory: wb86 --input path/to/dir --out out/ --visualize
- Video: wb86 --video path/to/video.mp4 --out out/frames

Outputs
- JSON per frame with 86×(x,y,conf), schema, boxes, and meta.
- Optional visualization overlays saved alongside JSON.

Notes
- Model weights resolved by MMPose/Ultralytics; see configs/default.yaml to adjust models, thresholds, and ROI paddings.
- Tests skip at runtime if optional deps are missing.

