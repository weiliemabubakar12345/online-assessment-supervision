# Computer Vision Handover File Inventory

## 1. Purpose

This inventory tracks which Computer Vision artifacts are suitable for the
shared repository and what must happen before each artifact is uploaded. It is
an evolving handover record rather than a list of every historical file created
during the internship.

The inventory uses four classifications:

| Classification | Meaning |
| --- | --- |
| **Ready to upload** | Reviewed, shareable, and already suitable for its repository destination |
| **Modification required** | Technically relevant, but paths, filenames, configuration, outputs, or code structure must be cleaned first |
| **Documentation required** | The artifact may be retained or referenced only after its source, purpose, version, licence, configuration, or access instructions are documented |
| **Do not upload** | Private, raw, redundant, temporary, machine-specific, unclear-licence, or otherwise unsuitable for the shared repository |

An artifact is not considered ready merely because it runs on the original
development machine.

## 2. Current Repository Entry Files

| Artifact | Repository destination | Classification | Status / action |
| --- | --- | --- | --- |
| Computer Vision overview | `computer_vision/README.md` | Ready to upload | Added on the documentation branch; update Quick Start after clean-setup verification |
| Computer Vision ignore rules | `computer_vision/.gitignore` | Ready to upload | Added on the documentation branch |
| Project scope | `computer_vision/docs/project_scope.md` | Ready to upload | Added on the documentation branch |
| Module overview | `computer_vision/docs/module_overview.md` | Ready to upload | Added on the documentation branch |
| Object-detection class mapping | `computer_vision/docs/datasets/class_mapping.md` | Ready to upload | Added on the documentation branch |
| Dataset sources and V3 composition | `computer_vision/docs/datasets/dataset_sources.md` | Ready to upload | Added on the documentation branch; exact source access dates may be supplemented later |
| Handover file inventory | `computer_vision/docs/handover/file_inventory.md` | Ready to upload | This file; update throughout the handover |

## 3. Final Integration Code Candidates

The following are known integration artifacts. They must be checked together
because their interfaces and imports are interdependent.

| Current artifact | Intended repository area | Classification | Required action |
| --- | --- | --- | --- |
| `src/14_integration/01_head_gaze_adapter.py` | `computer_vision/src/` | Modification required | Confirm it matches the final head/gaze interface and replace machine-specific paths or imports |
| `src/14_integration/02_yolo_output_adapter.py` | `computer_vision/src/` | Modification required | Confirm the Dataset V3 label order and `laptop` to `computer_device` mapping |
| `src/14_integration/03_event_manager_updated.py` | `computer_vision/src/` | Modification required | Freeze the final implementation, remove `_updated` from the canonical filename, and verify temporal configuration |
| `src/14_integration/run_integrated_demo_updated.py` | `computer_vision/src/` | Modification required | Freeze the final entry point, remove `_updated`, replace local paths, and run a clean smoke test |
| `src/14_integration/run_integrated_demo_ui_improved.py` | `computer_vision/src/` | Modification required | Compare with the selected final demo and retain only if it is not superseded |

Before any integration script is marked ready:

- imports must work from the proposed repository structure;
- personal absolute paths must be removed;
- model and resource locations must come from configuration or documented
  command-line arguments;
- no secret, personal, or identifiable data may be embedded;
- the selected checkpoint and class mapping must be recorded; and
- a smoke test must confirm startup, event generation, logging, and clean exit.

The Event Logger and cross-module validation implementations are known parts of
the final runtime, but their exact canonical filenames have not yet been
recorded in this inventory. They must be added only after their actual files are
selected; placeholder filenames must not be invented.

## 4. Object-Detection Development and Evaluation Candidates

### 4.1 Known Source Areas

| Current source area | Classification | Handover decision |
| --- | --- | --- |
| `src/06_dataset_inspection/` | Modification required | Select only scripts needed to explain or reproduce the final dataset audit; exclude exploratory duplicates |
| `src/07_yolo11_baseline/` | Modification required | Retain only the baseline method or evaluation files referenced by the final comparison |
| `src/08_formal_baseline_evaluation/` | Modification required | Retain reviewed evaluation code and concise results required for traceability |
| `src/09_custom_dataset/` | Modification required | Select final Dataset V3 preparation and validation scripts; remove absolute dataset roots |
| `src/10_yolo11_finetuning/` | Modification required | Retain only training code or notebooks required to explain the final experiment history |
| `src/11_finetuned_evaluation/` | Modification required | Select canonical checkpoint-comparison and audio-device evaluation files |
| `src/14_yolo_improvement/` | Modification required | Treat as a candidate area; verify which files were actually used in the final OIV7-based model workflow |

The directory names reflect the local development history. They do not have to
be copied unchanged into the final repository. The final structure should
prioritize responsibility and reproducibility rather than week or experiment
number alone.

### 4.2 Known Individual Evaluation Artifacts

| Current artifact | Classification | Required action |
| --- | --- | --- |
| `08_live_yolo_checkpoint_comparison.py` | Modification required | Replace local checkpoint/output paths and document the unequal live-window limitation |
| `17_audio_device_recorded_video_checkpoint_evaluation_updated.py` | Modification required | Remove `_updated` after final review, preserve FPS-aware duration logic, and document inputs and output schema |
| `Week8_Day3_Audio_Device_Recorded_Video_Evaluation.ipynb` | Modification required | Remove credentials and local paths, clear unnecessary outputs, and add reproducible instructions |
| `08_compare_oiv7_v1_bookclean_recorded_inputs.py` | Modification required | Retain only if referenced by the final dataset-development narrative |
| `audit_custom_object_dataset_v2_targeted.py` | Modification required | Retain as historical audit evidence only if inputs and decisions are documented |
| `build_v1_bookclean_candidate.py` | Modification required | Retain only if needed to explain the superseded BookClean experiment |
| `OIV7_BookClean_Smoke_Test_Colab.ipynb` | Modification required | Historical experiment; review for duplication before upload |
| `OIV7_BookClean_Smoke_and_Video_Validation_Colab.ipynb` | Modification required | Historical experiment; review for duplication before upload |

Older V1, V2, and BookClean artifacts must not be presented as the final Dataset
V3 pipeline. If retained, they must be labelled as development-history or
ablation material.

## 5. Model and Runtime Resource Inventory

| Current artifact | Classification | Handover decision |
| --- | --- | --- |
| `models/yolo/live_checkpoint_selection/original_5e_best.pt` | Documentation required | Selected global integration checkpoint; decide whether to include the small project-generated file or provide a verified download record and checksum |
| `models/yolo/live_checkpoint_selection/extended_epoch10.pt` | Documentation required | Alternative 15-epoch checkpoint; normally document in the model registry rather than upload by default |
| `models/yolo/live_checkpoint_selection/extended_epoch25.pt` | Documentation required | Alternative 30-epoch checkpoint; normally document in the model registry rather than upload by default |
| `models/yolo/live_checkpoint_selection/extended_best.pt` | Documentation required | Alternative 50-epoch checkpoint; document its audio-device strength and other-class trade-offs |
| `models/experiment1/best.pt` | Do not upload | Historical checkpoint unless the final report requires an externally stored comparison artifact |
| `external_models/L2CS-Net/models/L2CSNet_gaze360.pkl` | Do not upload | Third-party checkpoint; provide official source, expected location, and integrity information instead |
| `models/mediapipe/face_landmarker.task` | Documentation required | Verify redistribution terms or provide official download instructions |
| `data/canonical_face_model/canonical14_scaled.csv` | Documentation required | Verify provenance and document the Canonical 14 convention before inclusion |

The repository-level model policy must be settled before any ignored model
binary is force-added. Do not bypass `.gitignore` merely to make a local run
work.

## 6. Environment and Configuration Inventory

The local `teep-integration` Conda environment has been validated with L2CS,
RetinaFace/`face_detection`, MediaPipe, Ultralytics `8.4.86`, PyTorch
`2.8.0+cpu`, torchvision `0.23.0+cpu`, and OpenCV `5.0.0`; `pip check` passed.

| Artifact | Classification | Required action |
| --- | --- | --- |
| Cleaned `teep-integration` environment specification | Modification required | Export a portable Conda YAML without machine-specific prefix information and verify it from a clean environment |
| Runtime configuration for model/resource paths | Modification required | Replace absolute Windows paths with relative paths, configuration fields, or command-line arguments |
| Example local configuration | Documentation required | Provide a sanitized example only; exclude real local configuration and credentials |
| Complete local Conda environment directory | Do not upload | Recreate from the cleaned environment specification instead |

The exact environment and configuration filenames will be recorded after the
portable files are created and tested.

## 7. Evaluation Protocols and Results

| Artifact group | Classification | Required action |
| --- | --- | --- |
| Final end-to-end evaluation protocol | Ready to upload after creation | Define scenarios, expected events, repetitions, duration, timestamps, pass criteria, and concurrent cues before formal trials |
| End-to-end trial log template | Ready to upload after creation | Use a stable CSV schema before recording formal trials |
| Selected module metrics and tables | Modification required | Copy only reviewed summaries into `results/`; retain method and denominator information |
| Selected figures | Modification required | Remove personal information and verify that source media may be shared |
| Raw event logs | Modification required | Anonymize and reduce to the evidence needed for reproducibility |
| Raw webcam videos or identifiable screenshots | Do not upload | Keep private and outside Git |
| Temporary output folders under `outputs/` | Do not upload | Preserve only reviewed, selected results in the final `results/` structure |

## 8. Dataset and Training-Data Inventory

| Current artifact | Classification | Handover decision |
| --- | --- | --- |
| Final Dataset V3 raw images and YOLO labels | Do not upload | Document sources, mappings, build process, and counts instead |
| `D:/TEEP_PROCESSED/custom_object_dataset_v1/` | Do not upload | Local historical dataset build |
| `D:/TEEP_PROCESSED/custom_object_dataset_v2/` | Do not upload | Local historical dataset build |
| `D:/TEEP_PROCESSED/custom_object_dataset_v1_bookclean_candidate/` | Do not upload | Local superseded candidate dataset |
| `custom_object_dataset_v1_bookclean_candidate.tar.gz` | Do not upload | Raw/exported dataset archive |
| `D:/TEEP_PROCESSED/custom_object_dataset_v1/mapping_manifest.csv` | Modification required | Contains provenance value but must be reviewed for absolute paths and redistribution-sensitive information before any derived manifest is shared |

## 9. Repository-Wide Exclusions

The following must remain outside the shared repository:

- raw or redistributed datasets;
- identifiable webcam images, recordings, or screenshots;
- original unreviewed Daily Worklogs;
- API keys, passwords, access tokens, and private configuration;
- complete Conda or virtual-environment folders;
- cache files, debug logs, temporary outputs, and interrupted runs;
- duplicate external-model repositories;
- machine-specific absolute paths in committed runtime files;
- redundant checkpoints and unreviewed large binaries; and
- files whose redistribution terms cannot be established.

## 10. Next Inventory Actions

1. Freeze and rename the final Event Manager and integrated-demo files.
2. Record the exact Event Logger and cross-module validation filenames.
3. Review the final integration dependency set as one unit.
4. Select the minimum Dataset V3 build, validation, and training artifacts
   required for reproducibility.
5. Create and clean the portable `teep-integration` environment specification.
6. Create the model registry and decide how the selected 5e checkpoint will be
   distributed.
7. Add the end-to-end protocol and trial-log template before formal testing.
8. Update this inventory whenever an artifact becomes ready, is superseded, or
   is excluded.
