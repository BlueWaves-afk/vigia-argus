vigia-argus

Argus-V8X: a practical YOLOv8 plugin for robust road-hazard detection on edge devices.
It adds two export-friendly upgrades to YOLOv8—SimAM (parameter-free attention) and a tiny Swin block at the deep stage—without forking Ultralytics. Result: better resilience in rain/fog/glare/occlusion, while staying mobile-ready (ONNX/TFLite INT8).

Why

Dashcams/phones see messy roads: glare, rain streaks, low light, partial occlusions. Vanilla detectors often miss or wobble. vigia-argus fortifies YOLOv8 with lightweight attention and global context at the right place (P5), so you get higher recall under adverse conditions with minimal latency overhead.

What you get
	•	Drop-in plugin (no fork): auto-registers custom layers into ultralytics at import time.
	•	SimAM everywhere it helps: applied after each C2f block in backbone & neck to suppress noise and emphasize salient edges.
	•	Tiny Swin at P5: one windowed self-attention layer (lazy channel-aware) before SPPF to integrate local+global context for partially obscured hazards.
	•	Export-safe ops: implemented with plain Linear/MatMul/Softmax/Reshape/Conv so ONNX → TFLite INT8 works.
	•	Two ready configs:
	•	argus_v8x.yaml – standard 3-scale (P3/P4/P5) head, best for realtime mobile.
	•	argus_v8x_p2.yaml – extra P2 (stride-4) head for tiny objects (slower, higher recall).
	•	Scale-agnostic: works with YOLOv8 n/s/m/l/x (Swin block adapts to channel dims on first forward).
	•	Lite fallback: if a device/NNAPI dislikes attention ops, drop Swin and keep SimAM only.

Key features
	•	Robustness to rain/fog/glare/night, partial occlusion.
	•	Mobile-first: INT8 quantization; small memory footprint.
	•	Ablation-friendly: train and compare baseline vs +SimAM vs +Swin easily.
	•	Zero runtime overhead beyond the model’s own layers (plugin patching runs once at import).

Installation

# Install CUDA-enabled PyTorch that matches your GPU (example: CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121

pip install ultralytics
pip install vigia-argus            # (or: pip install git+https://github.com/<you>/vigia-argus.git)

Quick start

import vigia_argus                 # registers SimAM & Swin into Ultralytics
from ultralytics import YOLO

# Build the model from the packaged YAML
m = YOLO(vigia_argus.model_yaml("argus_v8x.yaml"))  # or "argus_v8x_p2.yaml"
m.train(data="data.yaml", imgsz=640, epochs=100)

# Export (test early)
m.export(format="onnx", opset=12, imgsz=640)
m.export(format="tflite", int8=True, imgsz=640)

CLI

yolo detect train model=$(python -c "import vigia_argus; print(vigia_argus.model_yaml())") \
  data=data.yaml imgsz=640 epochs=100

Model scales (n/s/m/l/x)

Ultralytics applies width/depth multipliers per scale. The Swin block here auto-adapts to the actual channel count at P5, so you can train n for mobile or s/m if you can spend more compute. Rename the YAML to …-s.yaml if you want to select a scale by filename, or set depth_multiple/width_multiple directly.

Performance guidance
	•	Start with argus_v8x.yaml at 640 INT8 for on-device demos.
	•	If tiny debris/far potholes are missed, try argus_v8x_p2.yaml (expect ~10–25% more latency).
	•	If PTQ loses >1.5 mAP, run QAT for 10–20 epochs.
	•	Targets (mid-tier Android, NNAPI/GPU): p50 < 120 ms, p95 < 250 ms.

Design choices
	•	SimAM: parameter-free, cheap; placed widely (after C2f) to denoise features.
	•	Swin@P5: small, single block where features are compact; gives global context with limited cost.
	•	Head unchanged: preserves exportability and tooling compatibility.

Compatibility
	•	Ultralytics: ≥ 8.2.x
	•	PyTorch: ≥ 2.1
	•	ONNX: opset 12–13 recommended
	•	TFLite: INT8 (PTQ/QAT) with representative dataset

Roadmap
	•	Optional shifted windows (export-tested).
	•	Argus-V8X-Lite preset (SimAM-only YAML).
	•	Pretrained checkpoints on public road datasets.
	•	Extra robustness augs pack (rain/fog/night suite).

License & notes
	•	This plugin’s code is under your chosen license (e.g., Apache-2.0/MIT).
	•	It depends on Ultralytics, which is licensed separately (AGPL-3.0/Enterprise). Ensure your usage complies with Ultralytics’ terms when training/serving models over a network.

Keywords

yolo · ultralytics · object-detection · simam · swin-transformer · attention · edge-ai · tflite · onnx · mobile

⸻