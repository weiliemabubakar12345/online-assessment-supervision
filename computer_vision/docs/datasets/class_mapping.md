# Object-Detection Class Mapping

## 1. Purpose

This document defines the canonical object-detection classes used by OIV7-Anchor Dataset V3, the trained YOLO model, and the Computer Vision integration interface.

It distinguishes three naming layers:

1. **source labels** from the original datasets;
2. **model labels** used for training and checkpoint metadata; and
3. **integration labels** exposed to downstream event-processing components.

Dataset provenance, source counts, licenses, and dataset-version changes are documented separately. This file is the authoritative reference for label meaning and label conversion.

## 2. Canonical Dataset V3 Model Classes

OIV7-Anchor Dataset V3 uses seven trained classes. Their identifiers and order must remain fixed when training, evaluating, or loading a compatible checkpoint.

| Class ID | Model label | Canonical meaning |
|---:|---|---|
| 0 | `person` | An examinee or another visible person |
| 1 | `phone` | A mobile phone |
| 2 | `laptop` | A laptop-related object labelled as the trained laptop class |
| 3 | `book_notes` | A book, paper, notes, or equivalent study material |
| 4 | `calculator` | A calculator |
| 5 | `watch` | A wristwatch or similar visible watch |
| 6 | `audio_device` | Headphones, headsets, wired earphones, earbuds, or related wearable audio devices |

Changing the order would make existing checkpoint class IDs inconsistent with the dataset and integration code.

## 3. Model-to-Integration Mapping

The integration adapter exposes a stable seven-class interface:

| Class ID | Model label | Integration label | Conversion |
|---:|---|---|---|
| 0 | `person` | `person` | unchanged |
| 1 | `phone` | `phone` | unchanged |
| 2 | `laptop` | `computer_device` | renamed by the adapter |
| 3 | `book_notes` | `book_notes` | unchanged |
| 4 | `calculator` | `calculator` | unchanged |
| 5 | `watch` | `watch` | unchanged |
| 6 | `audio_device` | `audio_device` | unchanged |

`computer_device` is the integration-facing name for the current trained `laptop` class. The broader interface name allows future explicitly trained and validated computer-device subtypes to be added without changing the downstream event schema.

In the current checkpoint, however, `computer_device` must be interpreted as the adapter output mapped from `laptop`. It is not evidence that the model has been trained and validated as a general detector for all computer equipment.

## 4. Canonical Normalization Rules

The following semantic rules were used when consolidating selected source datasets into the seven-class task:

| Source concept or label | Canonical model label | Rationale |
|---|---|---|
| student, examinee, extra person, or person | `person` | Person role is handled later through person-count and context logic |
| mobile phone or phone | `phone` | Unified phone cue |
| laptop | `laptop` | Retains the current trained class name |
| book, paper, or notes | `book_notes` | Combines visually and functionally related study-material cues |
| calculator | `calculator` | Retained as a separate cue |
| watch | `watch` | Retained as a separate cue |
| headphone, headset, earphone, earbud, AirPods, or neckband | `audio_device` | Combines related visible wearable-audio subtypes |

Capitalization and source-specific spelling are normalized during dataset conversion. Canonical model labels use lowercase snake case.

## 5. Verified Source-Specific Examples

The following mappings have been explicitly verified in the project notes or dataset preparation workflow.

### 5.1 Online Exam Proctoring

The source YAML contains:

- `student` and `extra_person`, normalized to `person`;
- `phone`, normalized to `phone`;
- `laptop`, normalized to `laptop`;
- `book`, normalized to `book_notes`; and
- `calculator`, normalized to `calculator`.

This source does not contribute a separate `audio_device` class.

### 5.2 OIV7 Laptop Anchor

The OIV7 anchor addition uses the OIV7 `Laptop` category and maps it to the Dataset V3 `laptop` class.

The OIV7 pretrained label space also contains other categories, including `Computer monitor`, but Dataset V3 does not define an independent `computer_monitor` trained class. The current integration interface must therefore not be described as a validated general monitor detector.

### 5.3 Ginger Audio-Device Data

The verified Ginger source labels are normalized as follows:

| Ginger source label | Model label |
|---|---|
| `AirPods` or `airpods` | `audio_device` |
| `Earphone` | `audio_device` |
| `Headphone` | `audio_device` |
| `Neckband` | `audio_device` |

The Ginger `null` category is not treated as an object class and does not produce an `audio_device` annotation.

### 5.4 Other Included Source Profiles

Dataset V3 also contains normalized records from the following verified source profiles:

- `deteksi_anomali_ujian`;
- `human_laptop_notebook`;
- `laptopmodel`;
- `online_cheat_detection`; and
- the base records retained by the final Dataset V3 construction process.

Their records use the same seven canonical model labels after normalization. Detailed source provenance and per-source contribution counts belong in `dataset_sources.md` rather than this class-definition file.

## 6. Excluded or Non-Canonical Labels

The following concepts are not independent Dataset V3 trained classes:

| Concept | Current handling |
|---|---|
| `computer_monitor` | No independent trained class; must not be claimed as a separately validated subtype |
| `keyboard` | Not included in the final seven-class task |
| `mouse` | Not included in the final seven-class task |
| `null` | Represents no target object; excluded from positive object annotations |
| detailed audio subtypes | Consolidated into `audio_device` rather than exposed as separate classes |
| `student` and `extra_person` | Consolidated into `person`; the integration layer derives person states |
| separate book and paper classes | Consolidated into `book_notes` |

The absence of an independent class does not prove that visually similar objects never appear in source images. It means that the current model and evaluation do not expose that concept as a separately trained and validated output class.

## 7. Person-State Derivation

The model produces `person` detections rather than separate `student`, `extra_person`, `NO_PERSON`, or `MULTIPLE_PERSONS` object classes.

The integration layer derives person-state observations from the number and validity of normalized person detections:

- no valid person detection may produce a `NO_PERSON` candidate;
- one valid person detection represents the expected single-person condition; and
- two or more valid person detections may produce a `MULTIPLE_PERSONS` candidate.

Cross-module validation can suppress a `NO_PERSON` candidate when valid face evidence is still available from the head/gaze pipeline.

## 8. Historical Naming Note

Earlier Dataset V1 and V2 experiments and notes may use different object-class terminology. In particular, some Dataset V2 work discussed a broader `computer_device` training class and possible laptop/monitor consolidation.

OIV7-Anchor Dataset V3 supersedes that training-label convention:

- the current **model label** is `laptop`; and
- the current **integration label** is `computer_device`.

Historical mappings should be retained only when documenting dataset development. They must not replace the current Dataset V3 class order or checkpoint metadata.

## 9. Usage Rules

When adding code, configuration, evaluation results, or documentation:

1. Preserve the seven Dataset V3 class IDs and model-label order.
2. Report raw checkpoint output using the model label where model behaviour is being analysed.
3. Use integration labels when describing downstream events and the shared runtime interface.
4. State explicitly that current `computer_device` observations map from the trained `laptop` class.
5. Do not report an excluded or merged concept as an independently evaluated class.
6. Document any future class addition as a new dataset, checkpoint, configuration, and evaluation version.
