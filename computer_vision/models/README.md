# Computer Vision Model Registry

This directory documents the external model weights required by the Computer
Vision runtime. Model binaries are not stored as ordinary Git files.

## Required Runtime Models

| Module | Runtime role | Canonical file | Required |
| --- | --- | --- | --- |
| Object detection | Person and object-cue detection | `yolo/original_5e_best.pt` | Yes |
| Gaze estimation | L2CS coarse gaze estimation | `l2cs/L2CSNet_gaze360.pkl` | Yes |
| Eye reliability | MediaPipe face landmarks and blendshapes | `mediapipe/face_landmarker.task` | Yes for the current reliability-aware gaze pipeline |

The expected local structure is:

```text
computer_vision/models/
├── README.md
├── yolo/
│   ├── original_5e_best.pt
│   ├── extended_epoch10.pt
│   ├── extended_epoch25.pt
│   └── extended_best.pt
├── l2cs/
│   └── L2CSNet_gaze360.pkl
└── mediapipe/
    └── face_landmarker.task
```

Only `README.md` is intended to be tracked in ordinary Git. Runtime model
assets are distributed through the project's shared Google Drive folder.

## YOLO Checkpoint Registry

The object detector was initialized from `yolov8s-oiv7.pt` and fine-tuned on
the project OIV7-Anchor Dataset V3. The pretrained base model is a training
reference and is not required for normal inference with a fine-tuned
checkpoint.

| Checkpoint ID | File | Approximate training stage | Runtime role |
| --- | --- | ---: | --- |
| `5e` | `original_5e_best.pt` | 5 epochs | Default balanced checkpoint |
| `15e` | `extended_epoch10.pt` | 15 epochs | Recall-oriented alternative |
| `30e` | `extended_epoch25.pt` | 30 epochs | Conservative low-background-false-positive alternative |
| `50e` | `extended_best.pt` | 50 epochs | Computer-device and audio-device-oriented alternative |

Checkpoint `5e` is the frozen default for the final integration baseline. It
was selected using the controlled recorded-video checkpoint comparison, with
local live-webcam testing retained as supplementary deployment evidence.

Alternative checkpoints are optional. If they are retained, the selected
checkpoint must remain fixed during an individual evaluation session and its
ID must be written to the event and trial logs.

### Frozen YOLO Runtime Settings

| Setting | Value |
| --- | --- |
| Default checkpoint ID | `5e` |
| Base confidence threshold | `0.25` |
| Input size | `640` |
| IoU threshold | `0.45` |
| Canonical device | CPU |
| Canonical runtime mode | Asynchronous latest-frame inference |

The `audio_device` formal-event rule uses an additional confidence threshold of
`0.43` together with its temporal rule. This does not replace the detector's
base confidence threshold for all classes.

## YOLO Output Interface

The fine-tuned model exposes seven project classes:

- `person`;
- `phone`;
- `laptop`;
- `book_notes`;
- `calculator`;
- `watch`; and
- `audio_device`.

The trained `laptop` label is mapped to the integration-facing
`computer_device` label by the output adapter. The current checkpoint therefore
validates laptop detection; it must not be described as a validated general
monitor detector.

## L2CS Gaze Model

| Field | Value |
| --- | --- |
| File | `L2CSNet_gaze360.pkl` |
| Runtime role | Coarse gaze-angle estimation |
| Canonical device | CPU |
| Output use | Reliability-aware coarse gaze direction and context |

L2CS output is not used alone as a behavioural verdict. The integration also
uses head pose, eye reliability, gaze-output availability, and independent
gaze-event eligibility before emitting a formal gaze event.

The model's original source, license, redistribution conditions, and full file
checksum must be recorded before the binary is distributed through the shared
repository or another external channel.

## MediaPipe Face Landmarker Asset

The current gaze-reliability pipeline also loads `face_landmarker.task`. This
asset supplies the MediaPipe face-landmark and blendshape outputs used for eye
reliability checks; it is separate from the installed `mediapipe` Python
package and from the L2CS checkpoint.

This asset remains a runtime requirement unless the final implementation is
changed to an API that does not use `model_asset_path` or
`FaceLandmarker.create_from_options`. Its source, version, and redistribution
terms still need to be recorded; its local size and checksum are listed below.

## Model Placement and Configuration

After obtaining the approved model files, place them at the paths shown above.
Runtime code must resolve model locations from repository-relative
configuration or command-line arguments. Do not commit a developer-specific
absolute path such as `D:\\...` or `C:\\Users\\...`.

The final runtime should support an explicit YOLO checkpoint choice while
defaulting to `5e`. A checkpoint must not change during an active test session.

## Integrity Record

The following values were measured from the frozen local model files on
13 August 2026:

| File | Size (bytes) | Size (MiB) | SHA-256 | Runtime status |
| --- | ---: | ---: | --- | --- |
| `original_5e_best.pt` | 22,517,802 | 21.47 | `FE6C93A04730DD31F9F2EA9451A58D7EC0899DED4713DD1A0A8A13EA0AB3E2FF` | Required; default `5e` |
| `extended_epoch10.pt` | 89,527,975 | 85.38 | `1B66A6DB857F08E2DB214B5A8EE54D89BA90078E3277B882F94703FCD51C2E55` | Optional `15e` |
| `extended_epoch25.pt` | 89,529,959 | 85.38 | `98C6F3AAB9ADBE3FFE0EBB1A655A805FA365F81BD09F546728E428D741D0D06F` | Optional `30e` |
| `extended_best.pt` | 22,522,986 | 21.48 | `A8D2577F7F8A037A120896406E9A9F5D065F775505658BE84D5BE6588FDE2385` | Optional `50e` |
| `L2CSNet_gaze360.pkl` | 95,849,977 | 91.41 | `8A7F3480D868DD48261E1D59F915B0EF0BB33EA12EA00938FB2168F212080665` | Required |
| `face_landmarker.task` | 3,758,596 | 3.58 | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` | Required by the current gaze-reliability pipeline |

The three required runtime model assets total 122,126,375 bytes
(approximately 116.47 MiB). All six registered model assets total 323,707,295
bytes (approximately 308.71 MiB). The optional checkpoints therefore should
not be distributed by default unless checkpoint switching or reproduction of
the comparison is required.

On Windows PowerShell, calculate the values with:

```powershell
Get-Item <model-path> | Select-Object Name, Length
Get-FileHash <model-path> -Algorithm SHA256
```

Checksums verify file identity; they do not grant redistribution permission.

## Distribution Status

The selected distribution method is the project [Computer Vision Models
folder](https://drive.google.com/drive/folders/13x7uOq-bhqz_WP2Bo6Hd3_GXpEd4OmnG?usp=sharing)
on Google Drive. Keep the exact canonical filenames and use the following
layout:

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

The three required runtime assets were present with matching filenames and byte
sizes when the folder was checked on 13 August 2026. Recipient access settings
must still be tested with the intended project members. After downloading a
file, preserve its filename and compare its SHA-256 value with the integrity
record before running the integration.

| Item | Current status |
| --- | --- |
| Model identities and runtime roles | Frozen |
| Default YOLO checkpoint | Frozen as `5e` |
| Alternative YOLO checkpoint selection | Supported as an optional configuration |
| Ordinary Git tracking of `.pt` and `.pkl` files | Prohibited |
| Model distribution storage | Shared Google Drive folder selected |
| Required runtime assets in Drive | Filename and byte-size presence verified on 13 August 2026 |
| Intended-recipient access verification | Pending confirmation |
| License and redistribution review | Required before binary distribution |
| File sizes and full SHA-256 checksums | Verified on 13 August 2026 |

Do not publish model binaries merely because they are available locally.
Confirm the repository's distribution scope and each model's applicable terms
first.

## Update Policy

- Keep model filenames and checkpoint IDs stable after evaluation begins.
- Record any checkpoint change in configuration and result metadata.
- Do not overwrite a model file while retaining its old checksum.
- Keep training-only base weights separate from runtime-required weights.
- Update this registry whenever the approved storage method, checksum, or
  redistribution status changes.
