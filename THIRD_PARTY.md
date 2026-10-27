# Third-Party Components

This project integrates or evaluates third-party software, models, and datasets. Their original licenses and terms remain applicable.

## Runtime / development dependencies

- **MediaPipe / Face Landmarker** — facial landmark inference.
- **Android / CameraX** — Android camera acquisition.
- **llama.cpp** — local GGUF model inference.
- **LFM2.5-1.2B-Instruct** — local contextual language-model candidate evaluated in the final prototype.
- **Qwen3-1.7B** — comparison model used in local benchmark experiments.

## External datasets reviewed in the research methodology

- UTA-RLDD.
- NTHU Driver Drowsiness Detection Dataset.
- YawDD.

The final H1/H2 metrics in the manuscript were obtained from controlled local validation recordings and not from those external datasets.

## License note

This file is a component inventory, not a replacement for the license files or terms of the individual upstream projects. Before public release, verify that redistributed binaries, model files, and other third-party assets are permitted for redistribution. Large model weights should normally be obtained from their original distribution source rather than committed to this repository.
