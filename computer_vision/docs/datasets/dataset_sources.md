# Dataset Sources and Final V3 Composition

## 1. Purpose

This document records the source profiles that were normalised and combined to
form the final seven-class YOLO training dataset. It separates three different
questions:

1. which source profiles are present in the final build;
2. how many images each profile contributes; and
3. whether the original source page and redistribution terms have been
   independently verified.

The repository does not redistribute raw dataset images. The links below are
provided for provenance only. Anyone rebuilding the dataset must review and
follow the terms of every original source.

## 2. Final Dataset V3 Summary

The final V3 dataset contains **11,091 training images** and **1,833 validation
images**, for a total of **12,924 images**. Structural validation confirmed a
one-to-one match between image files and YOLO label files, with no empty label
files in either split.

The build combines a 10,091-image base training split with 1,000 additional
Open Images V7 laptop-anchor images. Its validation split combines 1,781 base
images with 52 laptop-anchor images.

| Split | Base dataset | OIV7 laptop anchor | Final total |
| --- | ---: | ---: | ---: |
| Train | 10,091 | 1,000 | 11,091 |
| Validation | 1,781 | 52 | 1,833 |
| Total | 11,872 | 1,052 | 12,924 |

## 3. Source Profile Registry

The profile names below are the canonical names stored in the final build
summary. They should be used when reproducing audits or reporting dataset
composition.

| Canonical source profile | Train | Validation | Total | Provenance status | Role in the final build |
| --- | ---: | ---: | ---: | --- | --- |
| `deteksi_anomali_ujian` | 5,574 | 984 | 6,558 | [Public source page](https://universe.roboflow.com/gabys-workspace-iw7r1/deteksi-anomali-ujian/dataset/8); page reports CC BY 4.0 | Main exam-scene source; normalised labels include person and exam-related object cues |
| `ginger_audio_device` | 515 | 91 | 606 | [Public source page](https://universe.roboflow.com/ginger-ojdx6/earphone-detection-75qzd); page reports CC BY 4.0 | Provides the retained audio-device subtypes |
| `human_laptop_notebook` | 102 | 18 | 120 | [Public source page](https://universe.roboflow.com/hoi-c7ly2/human-laptop-notebook); page reports CC BY 4.0 | Small retained source containing person, laptop, and book scenes |
| `laptopmodel` | 79 | 14 | 93 | [Public source page](https://universe.roboflow.com/dpow/laptopmodel); page reports CC BY 4.0 | Small retained source profile from the base dataset |
| `oiv7_laptop_anchor` | 1,000 | 52 | 1,052 | [Open Images V7](https://storage.googleapis.com/openimages/web/index.html); see its image and annotation licence notes | Adds laptop-specific visual diversity and anchors the trained `laptop` class |
| `online_cheat_detection` | 181 | 32 | 213 | [Public source page](https://universe.roboflow.com/eis-ysd6a/online-cheat-detection); page reports CC BY 4.0 | Small supplementary exam-scene source |
| `online_exam_proctoring` | 3,640 | 642 | 4,282 | [Public source page](https://universe.roboflow.com/nhn-dng-vt-th/online-exam-proctoring-wjh05); page reports CC BY 4.0 | Major source for person and exam-related objects; sole confirmed source of calculator instances in this build |
| **Total** | **11,091** | **1,833** | **12,924** |  |  |

The canonical profile names in the first column are the names stored in the
final build summary. They may differ in formatting from the public project
titles. The source URL, dataset version used, access date, and licence displayed
on the source page should be recorded before any public redistribution decision
is made.

## 4. Label Normalisation by Source

All retained annotations were converted to the final class IDs documented in
[`class_mapping.md`](class_mapping.md). Important source-specific mappings
include:

| Source | Original label or subtype | Final training class |
| --- | --- | --- |
| `deteksi_anomali_ujian` | `handphone` | `phone` |
| `deteksi_anomali_ujian` | `smartwatch` | `watch` |
| `human_laptop_notebook` | `book` | `book_notes` |
| `ginger_audio_device` | `AirPods` / `airpods` | `audio_device` |
| `ginger_audio_device` | `Earphone` | `audio_device` |
| `ginger_audio_device` | `Headphone` | `audio_device` |
| `ginger_audio_device` | `Neckband` | `audio_device` |
| `oiv7_laptop_anchor` | Open Images `Laptop` | `laptop` |
| `online_cheat_detection` | `Mobile` | `phone` |
| `online_cheat_detection` | `Earphones` / `Headphones` | `audio_device` |
| `online_exam_proctoring` | `student` / `extra_person` | `person` |
| `online_exam_proctoring` | `book` | `book_notes` |

The Ginger source's `null` category was not retained as a trained object class.
Keyboard and mouse labels from earlier exploratory work are also outside the
final seven-class schema.

The local source YAML used during preprocessing recorded six Online Exam
Proctoring labels: `book`, `calculator`, `extra_person`, `laptop`, `phone`, and
`student`. Because public dataset pages can change independently of a downloaded
snapshot, the local YAML and build summary remain the authority for reproducing
this project's class mapping.

## 5. Final Class-Instance Counts

Image totals and object-instance totals are different: one image can contain
multiple annotated objects. The final labels contain the following bounding-box
instances.

| Class ID | Training class | Train instances | Validation instances | Total instances |
| ---: | --- | ---: | ---: | ---: |
| 0 | `person` | 8,721 | 1,495 | 10,216 |
| 1 | `phone` | 3,220 | 568 | 3,788 |
| 2 | `laptop` | 2,645 | 253 | 2,898 |
| 3 | `book_notes` | 2,824 | 474 | 3,298 |
| 4 | `calculator` | 528 | 91 | 619 |
| 5 | `watch` | 2,201 | 390 | 2,591 |
| 6 | `audio_device` | 2,529 | 449 | 2,978 |
| **Total** |  | **22,668** | **3,720** | **26,388** |

Source-level instance counts were not preserved in the final build summary.
They must not be inferred from the per-source image counts.

## 6. OIV7 Laptop Anchor

The OIV7 component is a targeted `Laptop` subset, not a separate seven-class
dataset and not a `computer_monitor` source. It contributes 1,000 training
images and 52 validation images to strengthen laptop coverage while preserving
the final class schema.

The trained class remains `laptop`. At integration time, it is exposed as
`computer_device` so that the event interface can later be extended to other
validated computer-device types. A monitor must not be treated as supported
until monitor examples have been explicitly trained and evaluated.

## 7. Distinction from the OPS Exploratory Dataset

The earlier OPS/NoOPS dataset was used during exploratory analysis. It is not a
canonical source profile in the final V3 build and must not be presented as
another name for `online_exam_proctoring`.

This distinction matters because the two names refer to different project
artifacts:

- `online_exam_proctoring` is a verified component of the final V3 composition;
- OPS/NoOPS belongs to the earlier experimental history and is excluded from
  the final source-count table above.

## 8. Reproducibility and Redistribution Rules

- Do not commit raw source images or labels to this repository.
- Do not commit exported Roboflow archives.
- Keep any local dataset root outside Git, or under an ignored data directory.
- Preserve the final source-profile names when generating build summaries.
- Record the downloaded dataset version, access date, original URL, and licence
  displayed on the source page for every future source. A dated screenshot or
  saved PDF may be kept locally as supporting evidence.
- Recheck the original source terms before sharing derived archives or trained
  weights.
- Open Images states that its annotations are CC BY 4.0 and its images are
  listed as CC BY 2.0, while also advising users to verify the licence status of
  each image individually.

## 9. Known Documentation Gaps

The final V3 counts and mappings are frozen. The remaining provenance task is
to confirm and record the exact downloaded version and access date for each
Roboflow source.

Filling these metadata gaps should update only the provenance cells in Section
3; it should not change the final dataset composition reported in this file.
