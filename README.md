vigia-argus

Argus-V8X — a practical YOLOv8 plugin for robust road-hazard detection on edge devices.

Argus adds two export-friendly upgrades to YOLOv8—SimAM (parameter-free attention) and a tiny Swin block at the deep stage—without forking Ultralytics. Result: better resilience in rain / fog / glare / occlusion, while staying mobile-ready (ONNX / TFLite INT8).

⸻

Why

Dashcams and phones see messy roads: glare, rain streaks, low light, partial occlusions. Vanilla detectors wobble. vigia-argus fortifies YOLOv8 with lightweight attention and global context at the right place (P5), delivering higher recall in adverse conditions with minimal latency overhead.

⸻

What you get
	•	✅ Drop-in plugin (no fork): auto-registers custom layers into Ultralytics at import time.
	•	🎯 SimAM where it matters: after each C2f in backbone & neck to suppress noise and emphasize salient edges.
	•	🪟 Tiny Swin at P5: one windowed self-attention layer (lazy channel-aware) before SPPF to integrate local+global context for partially obscured hazards.
	•	📦 Export-safe ops: plain Linear / MatMul / Softmax / Reshape / Conv → ONNX → TFLite INT8 works.
	•	🧩 Two ready configs:
	•	argus_v8x.yaml — standard 3-scale head (P3/P4/P5), best for realtime mobile.
	•	argus_v8x_p2.yaml — extra P2 (stride-4) head for tiny objects (slightly slower, higher recall).
	•	📏 Scale-agnostic: works with YOLOv8 n/s/m/l/x (Swin adapts to channel dims on first forward).
	•	🕊 Lite fallback: if a device/NNAPI dislikes attention ops, drop Swin and keep SimAM only.

⸻

Installation
	1.	Install CUDA-enabled PyTorch matching your GPU (example: CUDA 12.1):

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121

	2.	Install Ultralytics and the plugin:

pip install ultralytics
pip install vigia-argus
# or:
pip install "git+https://github.com/<your-org-or-user>/vigia-argus.git"


⸻

Quick start

Python

import vigia_argus  # registers SimAM & Swin into Ultralytics
from ultralytics import YOLO

# Build the model from the packaged YAML
m = YOLO(vigia_argus.model_yaml("argus_v8x.yaml"))  # or "argus_v8x_p2.yaml"

# Train
m.train(data="data.yaml", imgsz=640, epochs=100)

# Export early to validate deployment path
m.export(format="onnx", opset=12, imgsz=640)
m.export(format="tflite", int8=True, imgsz=640)

CLI

# get model yaml path provided by the plugin
model=$(python -c "import vigia_argus; print(vigia_argus.model_yaml())")

# train with Ultralytics CLI
yolo detect train model="$model" data=data.yaml imgsz=640 epochs=100


⸻

Model scales

Ultralytics applies width/depth multipliers per scale. The Swin block here auto-adapts to the actual P5 channels, so you can train n for mobile or s/m if you can spend more compute.

Scale	Depth mult	Width mult	Typical use
n	0.33	0.25	Mobile realtime
s	0.33	0.50	Mobile/edge (more recall)
m	0.67	0.75	Desktop/edge server
l/x	1.00	1.00/1.25	Research/high-accuracy

Tip: rename the YAML to …-s.yaml to select a scale by filename, or set depth_multiple / width_multiple directly.

⸻

Performance guidance
	•	Start with argus_v8x.yaml @ 640 INT8 for on-device demos.
	•	If tiny debris / far potholes are missed, try argus_v8x_p2.yaml (expect ~10–25% more latency).
	•	If PTQ loses > ~1.5 mAP, run QAT for 10–20 epochs.
	•	Targets (mid-tier Android, NNAPI/GPU): p50 < 120 ms, p95 < 250 ms.

⸻

Design choices
	•	SimAM: parameter-free, cheap; placed widely (after C2f) to denoise features.
	•	Swin @ P5: small, single block where features are compact; adds global context with limited cost.
	•	Head unchanged: preserves exportability and tooling compatibility.

⸻

Compatibility
	•	Ultralytics: ≥ 8.2.x
	•	PyTorch: ≥ 2.1
	•	ONNX: opset 12–13 recommended
	•	TFLite: INT8 (PTQ/QAT) with representative dataset

⸻

Roadmap
	•	Optional shifted windows (export-tested).
	•	Argus-V8X-Lite preset (SimAM-only YAML).
	•	Pretrained checkpoints on public road datasets.
	•	Extra robustness augmentations pack (rain/fog/night suite).

⸻

License & notes
	•	This plugin’s code is under your chosen license (e.g., Apache-2.0 / MIT).
	•	It depends on Ultralytics, which is licensed separately (AGPL-3.0 / Enterprise). Ensure your usage complies with Ultralytics’ terms when training/serving models over a network.

⸻

Keywords

yolo · ultralytics · object-detection · simam · swin-transformer · attention · edge-ai · tflite · onnx · mobile

⸻

made with love by team Vigia
