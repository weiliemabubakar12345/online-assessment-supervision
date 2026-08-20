# Computer Vision Model Registry

This directory documents the external source checkout and model assets required
by the computer-vision runtime. Model binaries are not stored as ordinary Git
files.

## Setup Summary

Complete both parts before running the integrated demo:

1. Download the approved frozen model files from the project
   [Computer Vision Models folder](https://drive.google.com/drive/folders/13x7uOq-bhqz_WP2Bo6Hd3_GXpEd4OmnG?usp=sharing).
2. Clone the official L2CS-Net source repository into
   `computer_vision/external/L2CS-Net/`.

The L2CS source code and `L2CSNet_gaze360.pkl` weight are separate
dependencies. Downloading one does not provide the other.

## Required Runtime Assets

| Module | Runtime asset | Canonical local path | Source | Required |
| --- | --- | --- | --- | --- |
| Object detection | `original_5e_best.pt` | `computer_vision/models/yolo/original_5e_best.pt` | Project shared Google Drive | Yes |
| Gaze estimation | `L2CSNet_gaze360.pkl` | `computer_vision/models/l2cs/L2CSNet_gaze360.pkl` | Project shared Google Drive | Yes |
| Eye reliability | `face_landmarker.task` | `computer_vision/models/mediapipe/face_landmarker.task` | Project shared Google Drive; upstream documentation linked below | Yes for the current pipeline |
| Gaze implementation | L2CS-Net source checkout | `computer_vision/external/L2CS-Net/` | Official upstream GitHub repository | Yes |

The required local structure is:

```text
computer_vision/
├── external/
│   └── L2CS-Net/
├── models/
│   ├── README.md
│   ├── yolo/
│   │   └── original_5e_best.pt
│   ├── l2cs/
│   │   └── L2CSNet_gaze360.pkl
│   └── mediapipe/
│       └── face_landmarker.task
└── resources/
    └── mediapipe/
        └── mediapipe_expanded_subset_14.csv
```

Alternative YOLO checkpoints are optional and may also be stored under
`computer_vision/models/yolo/` when checkpoint comparison is required.

## Download the Frozen Model Files

Open the project
[Computer Vision Models folder](https://drive.google.com/drive/folders/13x7uOq-bhqz_WP2Bo6Hd3_GXpEd4OmnG?usp=sharing)
and download the three files under `runtime_required/`:

```text
original_5e_best.pt
L2CSNet_gaze360.pkl
face_landmarker.task
```

Place them at the canonical local paths in the table above. Preserve the exact
filenames, then verify their SHA-256 values against the
[Integrity Record](#integrity-record).

The shared folder is the approved project handover channel for these frozen
files. Do not commit the binaries to Git or publish them through another channel
without confirming the applicable redistribution terms and intended access
scope.

## Download the L2CS-Net Source Code

The integration requires a local checkout of the official
[Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) repository. From the
online-assessment-supervision repository root, run:

```bat
git clone https://github.com/Ahmednull/L2CS-Net.git computer_vision\external\L2CS-Net
```

The external checkout must remain outside ordinary project Git tracking. Do not
copy it into `computer_vision/src/` and do not commit it as project-owned code.

For reproducibility, record the source revision used by the frozen runtime:

```bat
git -C computer_vision\external\L2CS-Net rev-parse HEAD
```

The exact approved commit SHA has not yet been added to this registry. Capture
it from the checkout used for the final evaluation, confirm that a clean clone
at that revision runs successfully, and then update the
[Reproducibility Status](#reproducibility-status) table. Until that is done, a
fresh clone follows the upstream default branch and is not a fully pinned
dependency.

## Verify the Installation

From the repository root, first confirm that the required paths exist:

```bat
dir computer_vision\external\L2CS-Net
dir computer_vision\models\yolo\original_5e_best.pt
dir computer_vision\models\l2cs\L2CSNet_gaze360.pkl
dir computer_vision\models\mediapipe\face_landmarker.task
dir computer_vision\resources\mediapipe\mediapipe_expanded_subset_14.csv
```

On Windows PowerShell, calculate file identity values with:

```powershell
Get-Item <model-path> | Select-Object Name, Length
Get-FileHash <model-path> -Algorithm SHA256
```

After the paths and checksums match, return to
[`../src/integration/README.md`](../src/integration/README.md) for the test and
runtime commands.

## YOLO Checkpoint Registry

The object detector was initialized from `yolov8s-oiv7.pt` and fine-tuned on the
project OIV7-anchor dataset. The pretrained base weight is a training reference;
it is not required for normal inference with a fine-tuned checkpoint.

| Checkpoint ID | File | Approximate training stage | Runtime role |
| --- | --- | ---: | --- |
| `5e` | `original_5e_best.pt` | 5 epochs | Frozen default balanced checkpoint |
| `15e` | `extended_epoch10.pt` | 15 epochs | Recall-oriented alternative |
| `30e` | `extended_epoch25.pt` | 30 epochs | Conservative low-background-false-positive alternative |
| `50e` | `extended_best.pt` | 50 epochs | Computer-device and audio-device-oriented alternative |

Checkpoint `5e` is the frozen integration baseline. It was selected using the
controlled recorded-video comparison, while local live-webcam checks remain
supplementary deployment evidence.

Alternative checkpoints are optional. If one is used, keep it fixed during the
entire evaluation session and record its ID in the session metadata and trial
log.

### Frozen YOLO Runtime Settings

| Setting | Value |
| --- | --- |
| Default checkpoint ID | `5e` |
| Base confidence threshold | `0.25` |
| Input size | `640` |
| IoU threshold | `0.45` |
| Canonical device | CPU |
| Canonical runtime mode | Asynchronous latest-frame inference |

The `audio_device` formal-event rule applies an additional `0.43` confidence
threshold together with its temporal rule. This does not replace the detector's
base threshold for other classes.

## YOLO Output Interface

The fine-tuned model exposes seven project classes:

- `person`
- `phone`
- `laptop`
- `book_notes`
- `calculator`
- `watch`
- `audio_device`

The output adapter maps the trained `laptop` label to the integration-facing
`computer_device` label. The frozen checkpoint validates laptop detection; it
must not be described as a validated general monitor detector.

## L2CS Gaze Model

| Field | Value |
| --- | --- |
| Upstream source | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) |
| Source-code license | MIT |
| Source location | `computer_vision/external/L2CS-Net/` |
| Approved source revision | Pending capture from the final frozen checkout |
| Model file | `L2CSNet_gaze360.pkl` |
| Model distribution | Project shared Google Drive |
| Runtime role | Coarse gaze-angle estimation |
| Canonical device | CPU |

The L2CS output is not used alone as a behavioural verdict. The integration also
uses head pose, eye reliability, gaze-output availability, and independent
gaze-event eligibility before emitting a formal gaze event.

The upstream repository provides the source code under the MIT License. Do not
assume that this statement alone grants permission to redistribute every
separately downloaded dataset or pretrained weight; keep the model-weight
distribution review recorded independently.

## MediaPipe Face Landmarker Asset

The current gaze-reliability pipeline loads `face_landmarker.task` to obtain
face landmarks and blendshape outputs used for eye-reliability checks. It is
separate from both the installed `mediapipe` Python package and the L2CS
checkpoint.

Google's official
[Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
describes the compatible task model and Python API. For the frozen project
runtime, use the exact file from the shared Google Drive and confirm its
checksum rather than silently replacing it with a newer asset.

## Integrity Record

The following values were measured from the frozen local files on 13 August
2026:

| File | Size (bytes) | Size (MiB) | SHA-256 | Runtime status |
| --- | ---: | ---: | --- | --- |
| `original_5e_best.pt` | 22,517,802 | 21.47 | `FE6C93A04730DD31F9F2EA9451A58D7EC0899DED4713DD1A0A8A13EA0AB3E2FF` | Required; default `5e` |
| `extended_epoch10.pt` | 89,527,975 | 85.38 | `1B66A6DB857F08E2DB214B5A8EE54D89BA90078E3277B882F94703FCD51C2E55` | Optional `15e` |
| `extended_epoch25.pt` | 89,529,959 | 85.38 | `98C6F3AAB9ADBE3FFE0EBB1A655A805FA365F81BD09F546728E428D741D0D06F` | Optional `30e` |
| `extended_best.pt` | 22,522,986 | 21.48 | `A8D2577F7F8A037A120896406E9A9F5D065F775505658BE84D5BE6588FDE2385` | Optional `50e` |
| `L2CSNet_gaze360.pkl` | 95,849,977 | 91.41 | `8A7F3480D868DD48261E1D59F915B0EF0BB33EA12EA00938FB2168F212080665` | Required |
| `face_landmarker.task` | 3,758,596 | 3.58 | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` | Required by the current gaze-reliability pipeline |

The three required runtime models total 122,126,375 bytes (approximately 116.47
MiB). All six registered model files total 323,707,295 bytes (approximately
308.71 MiB). The optional checkpoints therefore should not be distributed by
default unless checkpoint switching or reproduction of the comparison is
required.

Checksums verify file identity; they do not grant redistribution permission.

## Distribution Layout

Keep the exact canonical filenames in the shared folder:

```text
Computer Vision Models/
├── runtime_required/
│   ├── original_5e_best.pt
│   ├── L2CSNet_gaze360.pkl
│   └── face_landmarker.task
└── optional_yolo_checkpoints/
    ├── extended_epoch10.pt
    ├── extended_epoch25.pt
    └── extended_best.pt
```

The three required assets were present with matching filenames and byte sizes
when the shared folder was checked on 13 August 2026. Intended recipients should
still test their access before final handover.

## Reproducibility Status

| Item | Current status |
| --- | --- |
| Model identities and runtime roles | Frozen |
| Default YOLO checkpoint | Frozen as `5e` |
| Alternative YOLO checkpoint selection | Supported as an optional configuration |
| Ordinary Git tracking of `.pt`, `.pkl`, and `.task` files | Prohibited |
| Model distribution storage | Shared Google Drive folder selected |
| Required runtime assets in Drive | Filenames and byte sizes verified on 13 August 2026 |
| Full model SHA-256 checksums | Verified on 13 August 2026 |
| Intended-recipient Drive access | Pending confirmation |
| L2CS-Net source repository | Official upstream recorded |
| Exact L2CS-Net source commit SHA | Pending capture and clean-clone verification |
| Model-weight redistribution review | Required before wider publication |

## Update Policy

- Keep model filenames and checkpoint IDs stable after evaluation begins.
- Record any checkpoint or source-revision change in configuration and result
  metadata.
- Do not overwrite a model file while retaining its old checksum.
- Keep training-only base weights separate from runtime-required weights.
- Do not commit model binaries or the external L2CS checkout to ordinary Git.
- Update this registry whenever the approved storage method, checksum,
  source revision, or redistribution status changes.
