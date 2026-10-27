# Third-party dependencies, models, and licensing

This project integrates and evaluates external software and models. Original licenses remain controlling; the root Apache-2.0 LICENSE covers only the components described in LICENSING.md.

| Component | Role | Licensing / provenance |
|---|---|---|
| Google MediaPipe / TensorFlow Face Landmarker Android example | Original Android example and Face Landmarker integration | Apache License 2.0; retain existing TensorFlow headers and `android/FatigueMonitor/LICENSE.upstream`. https://github.com/google-ai-edge/mediapipe |
| MediaPipe Face Landmarker model bundle | Pretrained `face_landmarker.task` asset, also present at `models/face_landmarker.task` | Third-party model. Official download and model cards: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker . Its model-asset redistribution conditions should be independently verified; it is not relicensed by this repository. |
| llama.cpp | Optional local GGUF inference engine interfaced through JNI | MIT license of upstream llama.cpp: https://github.com/ggml-org/llama.cpp |
| Liquid AI LFM2.5-1.2B-Instruct | Optional local contextual generation, evaluated on device | LFM Open License v1.0 (`lfm1.0`); not Apache-2.0. https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct . Model weights are not distributed here. |
| Qwen3-1.7B | Benchmarked comparison model | Use its original upstream model license and version-specific terms. Weights are not distributed here. https://huggingface.co/Qwen/Qwen3-1.7B |
| AndroidX / CameraX and Android SDK components | Android camera and app support | Subject to their respective upstream licenses and platform terms. |

## External datasets discussed in the research methodology

UTA-RLDD, NTHU Driver Drowsiness Detection Dataset, and YawDD were considered as external reference datasets. **H1/H2 reported metrics were produced from local controlled validation**, not by training or evaluating on those external datasets. No third-party dataset distribution rights are claimed by this repository.

## Publication and privacy

The public repository intentionally excludes identifiable participant imagery and private notes. Public numerical experimental files were not modified for licensing. Their public availability does not automatically imply the right to redistribute identifiable participant materials.

This inventory is informational and is not legal advice. Users must review the version-specific licensing and attribution requirements of each dependency and model.
