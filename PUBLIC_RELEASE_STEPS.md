# Public Release Steps

Do **not** change the visibility of the existing private research repository. Its Git history contains internal development metadata and previously committed identifiable evidence media.

Instead, export a clean snapshot from `publication-mdpi` into a new repository with a fresh Git history.

## Bash procedure

Run these commands from your Ubuntu terminal.

```bash
cd ~

git clone --branch publication-mdpi --single-branch \
  git@github.com:JoaquinRodrigoMermaRoque/fatigue-driver-llm-assistant-prototype.git \
  fatigue-driver-mdpi-public

cd fatigue-driver-mdpi-public

# Remove the private repository history.
rm -rf .git

# Start a completely new public history.
git init
git branch -M main

# Review the exact snapshot before publishing.
git status
find docs/experiments -maxdepth 2 -type f | sort

# Confirm that the identifiable evidence images are absent.
find docs/evidence -type f -maxdepth 1 -print

# Confirm that no model GGUF or obvious secret/config files are present.
find . -type f \( -name "*.gguf" -o -name "*.jks" -o -name "*.keystore" -o -name ".env" -o -name "local.properties" \) -print

git add .
git commit -m "Public research snapshot for MDPI Technologies manuscript"
```

At this point, create a **new empty repository** on GitHub, for example:

`fatigue-driver-llm-assistant-mdpi`

Do not initialize it with a README, .gitignore, or license because the clean snapshot already contains project files.

Then connect and push:

```bash
git remote add origin git@github.com:JoaquinRodrigoMermaRoque/fatigue-driver-llm-assistant-mdpi.git
git push -u origin main
```

## Before switching the new repository to Public

1. Open `README.md` and verify the manuscript title.
2. Verify that `subject_001` and `subject_002` are the only participant identifiers in the experimental tables.
3. Verify that `docs/evidence/` contains only its privacy README and no face screenshots.
4. Verify that the public H4 file is:
   `docs/experiments/h8_h4_final_offline_context_PUBLIC_20261009.md`.
5. Choose a source-code license only after confirming the preferred licensing terms.
6. After the first public release, create a GitHub Release (recommended tag: `v1.0-mdpi-paper`) and optionally archive it with Zenodo to obtain a DOI.

## Result integrity

The validated numerical CSV artifacts must remain unchanged. The publication staging branch was checked against the validated source branch and the manuscript-level H1, H2, calibration, MediaPipe, and LLM CSV files have identical Git blob SHAs.
