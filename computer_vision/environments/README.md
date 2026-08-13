# Computer Vision Runtime Environment

This directory defines the Python environment used by the canonical Computer
Vision integration runtime.

## Environment Files

| File | Purpose |
| --- | --- |
| `teep_integration.yml` | Conda environment specification exported from the working `teep-integration` environment |

## Runtime Baseline

The `teep-integration` environment has been used successfully to run the
integrated CPU-based prototype containing:

- OIV7-based person and object-cue detection;
- MediaPipe and OpenCV head-pose processing;
- L2CS-based coarse gaze estimation;
- reliability and event-eligibility logic;
- duration-based event management and logging; and
- asynchronous latest-frame YOLO inference.

The environment specification was exported on 13 August 2026 from the working
Windows development environment. It does not contain a local Conda prefix,
user directory, or absolute project path.

## Key Versions

| Component | Version |
| --- | --- |
| Python | `3.9.25` |
| NumPy | `2.0.2` |
| MediaPipe | `0.10.35` |
| OpenCV Python | `5.0.0.93` |
| OpenCV Contrib Python | `5.0.0.93` |
| PyTorch | `2.8.0` |
| TorchVision | `0.23.0` |
| Ultralytics | `8.4.86` |
| SciPy | `1.13.1` |
| SoundDevice | `0.5.5` |

## Create the Environment

From the repository root, run:

```bash
conda env create -f computer_vision/environments/teep_integration.yml
conda activate teep-integration
```

If an environment with the same name already exists, do not overwrite a
working setup without first recording its current state. A separate temporary
environment name may be used for recreation testing.

## Verify the Installation

After activation, run:

```bash
python --version
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import sounddevice; print('SoundDevice import: OK')"
```

The frozen evaluation baseline uses CPU inference. CUDA availability is not
required for the canonical local evaluation.

## External Model Files

The Conda specification installs software dependencies only. It does not
contain model weights. The following external files must be placed according to
`computer_vision/models/README.md` before running the complete integration:

- the selected YOLO checkpoint, checkpoint ID `5e`
  (`original_5e_best.pt`); and
- the L2CS gaze checkpoint (`L2CSNet_gaze360.pkl`).

Model binaries must not be committed as ordinary Git files. Their approved
distribution method will be documented separately.

## Reproducibility Status

| Check | Status |
| --- | --- |
| Canonical integration runs in the source environment | Verified |
| Environment export excludes local absolute paths | Verified |
| Key dependency versions recorded | Verified |
| Environment recreated from the YAML in a clean test environment | Pending |
| Canonical smoke test executed in the recreated environment | Pending |

The environment should be considered a working snapshot until the two pending
checks are completed.

## OpenCV Packaging Note

The working snapshot currently contains both `opencv-python` and
`opencv-contrib-python` at the same version. Both packages provide the `cv2`
namespace and can overlap. They are retained here because this file records the
environment in which the integration currently runs successfully.

Do not remove either package from the canonical specification without first
creating a clean test environment and confirming that all integration imports
and smoke tests still pass.

## Update Policy

- Keep the environment fixed during an individual evaluation session.
- Record dependency changes in a separate commit.
- Re-export and re-check the YAML after an intentional dependency change.
- Do not add `prefix:`, `file:///`, editable local packages, user directories,
  or absolute project paths to the committed specification.
- Update the reproducibility table after clean-environment verification.
