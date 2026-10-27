# Licensing scope and attribution

This project is a research snapshot containing code, measurements, documentation, and third-party material. **A public GitHub repository does not make every file subject to one license.**

## Source code

The project author's original source-code contributions are distributed under the **Apache License, Version 2.0** (see root `LICENSE`). Portions derived from the Google MediaPipe/TensorFlow Face Landmarker Android example are also under Apache-2.0; their original copyright notices must be preserved. The upstream example's license is retained in `android/FatigueMonitor/LICENSE.upstream`.

The root Apache-2.0 license does **not** supersede third-party copyright notices or separate license conditions.

## Pretrained models, dependencies, and binary assets

- `models/face_landmarker.task` is a third-party Google MediaPipe model bundle included in the research snapshot. It is **not an original contribution of this repository's author**. Its rights, provenance and redistribution conditions must be checked against the upstream model documentation; the root Apache-2.0 grant must not be interpreted as an independent license to this model.
- LFM2.5-1.2B-Instruct (Liquid AI) is a separately licensed third-party model under **LFM Open License v1.0**. Its GGUF weights are not supplied in this repository.
- llama.cpp, when obtained separately, is governed by its upstream MIT license.
- Other dependencies, datasets and models remain under their respective licenses and access conditions (see `THIRD_PARTY.md`).

## Research results and participant-related data

The root software license is **not** a blanket license for `docs/experiments/` CSV files, controlled-trial records, participant-related data, or identifiable images. The experimental CSVs remain available for inspection and verification of the scientific findings; their public availability does not by itself grant additional rights over participant-related information or private source recordings.

Only pseudonymous participant identifiers should be used in public files. Do not add faces, participant identities, relationship information, consent forms, or raw private recordings to this repository.

## Publication version

The release `v1.0-mdpi-paper` identifies an existing earlier publication snapshot. Later documentation-only changes on `main` do not alter that tagged release or its validated experimental files.

For license questions about third-party materials, consult their upstream owners and official terms.
