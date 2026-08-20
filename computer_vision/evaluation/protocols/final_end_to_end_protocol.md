# Final End-to-End Event Evaluation Protocol

| Protocol field | Value |
| --- | --- |
| Version | `1.0` |
| Status | Final evaluation baseline |
| Standard round timing | `3 s` neutral, `5 s` cue, `5 s` recovery |

## 1. Purpose

This protocol evaluates whether the integrated Computer Vision prototype
converts controlled, observable webcam cues into the expected structured events
and logs them consistently.

The evaluation covers:

- neutral behaviour and false-alert control;
- head-pose, gaze, object-cue, and person-state events;
- event `START`-`ACTIVE`-`END` behaviour;
- missed, false, duplicate, and conflicting events;
- timestamps, duration, and asynchronous inference delay;
- concurrent-cue preservation; and
- known deployment failure cases.

This is an event-level prototype evaluation. It does not determine whether a
participant is cheating and does not establish production-level accuracy.

## 2. Evaluation Unit

The primary evaluation unit is one **trial round** containing a controlled cue
window. Each core scenario is repeated for **three rounds**.

One matching completed event is a true positive when it:

1. belongs to the expected event family;
2. is supported by the intended cue during that round; and
3. temporally overlaps the cue window.

The protocol does not invent event strings that are not exposed by the final
runtime. The evaluator must record the exact `actual_event` label written by the
Event Logger and map it to the expected family in the trial log.

## 3. Frozen Evaluation Baseline

The following settings must remain fixed within an evaluation session.

### 3.1 Object Detector

| Setting | Frozen value |
| --- | --- |
| Checkpoint ID | `5e` |
| Checkpoint | `original_5e_best.pt` |
| Base confidence threshold | `0.25` |
| Input size | `640` |
| IoU threshold | `0.45` |
| Device | CPU |
| Runtime mode | Asynchronous latest-frame YOLO |

The synchronous `--sync-yolo` mode may be used only as a separately identified
diagnostic run. Synchronous and asynchronous results must not be combined in
the same summary without explicit separation.

### 3.2 Object Interface

The integration-facing labels are:

- `person`;
- `phone`;
- `computer_device`;
- `book_notes`;
- `calculator`;
- `watch`; and
- `audio_device`.

`computer_device` currently maps from the trained `laptop` class. It must not be
interpreted as a validated general monitor detector.

### 3.3 Audio-Device Event Rule

| Setting | Frozen value |
| --- | ---: |
| Event-eligibility confidence | `0.43` |
| Minimum duration | `0.75 s` |
| Release grace | `0.60 s` |
| Cooldown | `0.50 s` |

This stricter confidence rule applies to `audio_device` event eligibility, not
to the base detector threshold for all classes.

### 3.4 Head and Gaze Calibration

- Head-pose neutral calibration and gaze centre calibration must be completed
  at the beginning of the session.
- Passive open-eye calibration must complete under a neutral, forward-facing
  condition.
- Do not recalibrate between rounds unless the current session is abandoned and
  restarted.
- If calibration fails, mark the session invalid rather than silently changing
  thresholds.

### 3.5 Event and Logging Behaviour

- Events follow the duration-based `START`-`ACTIVE`-`END` lifecycle.
- Completed events, filtered candidates, concurrent cues, and the session
  summary must be retained for verification.
- A YOLO `NO_PERSON` candidate may be suppressed when the head/gaze pipeline
  still provides valid face evidence.
- Gaze output may remain available for diagnostic context while independent
  gaze-event eligibility is suppressed.

Any additional event thresholds must be read from and recorded against the
frozen runtime configuration. They must not be changed midway through a
session.

## 4. Controlled Test Setup

Record the following before every session:

| Field | Required record |
| --- | --- |
| Session ID | Unique identifier |
| Date and local time | Start time of evaluation |
| Participant ID | Anonymous code only |
| Runtime commit | Git commit hash or exact local version |
| Checkpoint ID | Must remain fixed during the session |
| Runtime mode | Async or separately identified sync diagnostic |
| Camera | Device name or index |
| Camera resolution | Record the actual resolution |
| Approximate participant distance | Record and keep consistent |
| Lighting | Brief controlled-condition description |
| Background | Brief description and any known distractors |
| Glasses or relevant occlusion | Yes/no with short note |
| Calibration outcome | Pass/fail and any restart |
| Output directory | Session log location |

Recommended controlled baseline:

- one primary participant;
- camera approximately centred at eye level;
- stable indoor lighting;
- participant and chair positions marked or kept consistent;
- background unchanged across all core trials; and
- no unintended target objects visible during neutral, head, or gaze trials.

Only anonymous participant IDs may appear in repository results.

## 5. Standard Round Timing

Except where specified otherwise, each round uses this sequence:

| Phase | Duration | Required behaviour |
| --- | ---: | --- |
| Pre-cue neutral | `3 s` | Forward head, centre gaze, hands and target object out of view |
| Controlled cue | `5 s` | Perform or present only the intended cue |
| Recovery neutral | `5 s` | Return to neutral and allow the event to end |

The evaluator must record the observed cue-onset and cue-offset timestamps. Do
not estimate them later from model output alone.

Allow the recovery phase to finish before starting the next round. If an event
has not ended by the end of recovery, record this as an end-delay or lifecycle
problem and wait until the event closes before proceeding.

## 6. Core Formal Scenarios

Run the scenarios in the following order. Complete three rounds per scenario.

### 6.1 Neutral Baseline

| Scenario ID | Action | Duration per round | Expected result |
| --- | --- | ---: | --- |
| `N01` | Normal seated posture, forward head, centre gaze, no target object | `20 s` | No formal event |

Neutral rounds do not use the standard 3-5-5 timing. Any completed event during
a neutral round is a false alert unless a documented unintended cue actually
occurred; in that case, mark the round invalid and repeat it.

### 6.2 Head-Pose Cues

Keep the eyes natural and do not introduce a target object.

| Scenario ID | Controlled cue | Expected event family |
| --- | --- | --- |
| `H01` | Turn head left | `HEAD_LEFT` |
| `H02` | Turn head right | `HEAD_RIGHT` |
| `H03` | Raise head | `HEAD_UP` |
| `H04` | Lower head | `HEAD_DOWN` |

Record any direction reversal, short opposite-direction event, or approximately
180-degree pose flip as a separate failure note.

### 6.3 Gaze Cues

Keep the head approximately forward and move only the eyes as far as is natural
and repeatable. The expected event family is the runtime's corresponding
eligible gaze-deviation output; record its exact logger label.

| Scenario ID | Controlled cue | Expected event family |
| --- | --- | --- |
| `G01` | Gaze left | Eligible gaze-left deviation |
| `G02` | Gaze right | Eligible gaze-right deviation |
| `G03` | Gaze up | Eligible gaze-up deviation |
| `G04` | Gaze down | Eligible gaze-down deviation |

Record eye reliability, gaze-output availability, and gaze-event eligibility
where exposed. A diagnostic gaze direction without a formal event is not a
false alert; it is a miss only when the trial was eligible and the formal event
was expected.

### 6.4 Object Cues

Present each object in a natural assessment-relevant position and keep it
reasonably stable for the full cue window. Do not wave the object toward the
camera.

| Scenario ID | Controlled cue | Expected event family |
| --- | --- | --- |
| `O01` | Visible phone | `phone` |
| `O02` | Visible open laptop | `computer_device` |
| `O03` | Visible book, paper, or notes | `book_notes` |
| `O04` | Visible calculator | `calculator` |
| `O05` | Clearly visible watch | `watch` |
| `O06` | Over-ear headphone or other clearly visible audio device | `audio_device` |

Use the same physical object, position range, lighting, and approximate distance
for all three rounds of a scenario. Record the physical subtype in `notes`.

If a suitable over-ear headphone or headset is unavailable, mark `O06` as
**deferred**, not failed. Do not substitute small earbuds as the formal positive
case because unreliable earbud detection is already a documented limitation
and is evaluated separately in `K03`.

### 6.5 Person States

| Scenario ID | Controlled cue | Expected event family | Additional check |
| --- | --- | --- | --- |
| `P01` | Primary participant fully leaves the frame | `NO_PERSON` | Confirm no valid head/gaze face evidence remains |
| `P02` | A second person becomes clearly visible | `MULTIPLE_PERSONS` | Record the number and approximate positions of people |

If a second participant is unavailable, mark `P02` as **deferred**, not failed.

### 6.6 Multi-Cue Scenarios

| Scenario ID | Controlled cue | Expected primary events | Concurrent context to verify |
| --- | --- | --- | --- |
| `M01` | Turn head left while holding a phone visibly | Head-left and phone families | Phone context on head event and head context on phone event where supported |
| `M02` | Keep head forward, gaze right, and show book or notes | `gaze:LOOKING_RIGHT` and `book_notes` | Gaze eligibility and book-notes context retained without an automatic behavioural verdict |

Multi-cue evaluation checks whether separate supported events and concurrent
context are preserved. It does not require the system to combine them into a
single risk or cheating event.

## 7. Reliability and Suppression Checks

These scenarios validate that unreliable or contradictory observations do not
automatically produce misleading events.

| Scenario ID | Action | Expected handling |
| --- | --- | --- |
| `R01` | Blink naturally during neutral posture | No independent gaze-deviation event caused only by the blink |
| `R02` | Close eyes briefly while head remains forward | Gaze may become unavailable or ineligible; no forced directional gaze event |
| `R03` | Turn head far enough that one eye becomes unavailable | Gaze may be contextual or suppressed; no unreliable independent gaze event |
| `R04` | Create a brief YOLO no-person dropout while face evidence remains valid | `NO_PERSON` candidate should be suppressed where cross-module conditions apply |

Reliability checks should be reported separately from directional detection
accuracy because suppression can be the correct result.

## 8. Known-Failure Diagnostic Scenarios

These scenarios are diagnostic. They must not be used to inflate the core pass
rate, and a known failure does not invalidate unrelated core scenarios.

| Scenario ID | Diagnostic input | Known risk to record |
| --- | --- | --- |
| `K01` | Closed laptop | Confusion with `book_notes` or missing `computer_device` |
| `K02` | Thin side view of phone | Missed or unstable phone detection |
| `K03` | Small earbuds worn normally | Intermittent or absent `audio_device` event |
| `K04` | Wired earphones | Weak or questionable localization |
| `K05` | Background with one real person | False additional-person box or `MULTIPLE_PERSONS` event |
| `K06` | Distant person | Person miss, especially under less sensitive checkpoints |
| `K07` | Large head turn | Pose flip, direction reversal, or unreliable gaze state |

Run three rounds where practical. Report raw observations, but label the block
as **known-failure diagnostics**, not as controlled core accuracy.

## 9. Round-Level Outcome Rules

### 9.1 Correct Event

A round is a correct event round when at least one matching completed event:

- has the expected family;
- overlaps the controlled cue window; and
- is not solely caused by an unintended cue.

Because asynchronous YOLO results may be several hundred milliseconds old and
temporal confirmation adds delay, timing is measured separately. A supported
event that begins or ends late is still counted as detected but must receive a
timing-error note.

### 9.2 Missed Event

Set `missed_event = true` when the expected formal event is absent after the cue
and recovery window, provided that the round was valid and event-eligible.

For gaze trials, do not count an intentionally suppressed unreliable result as
a normal directional miss. Record `eligibility_failure` or the exposed
reliability reason instead.

### 9.3 False Alert

Set `false_alert = true` for any completed event that is unsupported by the
planned cue or a documented concurrent cue. Examples include:

- any formal event during a valid neutral round;
- the wrong head direction;
- an unrelated object event from the background; or
- `MULTIPLE_PERSONS` when only one real person is visible.

### 9.4 Duplicate Event

Set `duplicate_event = true` when one continuous cue produces more than one
completed event of the same family without a genuine new occurrence.

### 9.5 Lifecycle and Logging Failure

Record a lifecycle or logging failure when:

- an event begins but never produces a valid end record;
- timestamps are missing or non-monotonic;
- duration is negative or inconsistent with start/end timestamps;
- the checkpoint/session metadata is missing;
- concurrent cues visible in runtime evidence are lost from the supported log
  fields; or
- the completed-event file and session summary disagree.

## 10. Timing Measurements

Timing comparisons must use timestamps converted to a common time base. The
runtime's numeric timestamps and externally recorded ISO wall-clock timestamps
must never be subtracted directly.

### 10.1 Timestamp Meanings

Use the Event Manager fields according to their semantics:

- `event_start_timestamp` is the runtime time at which cue presence was first
  observed for the event sequence;
- `event_end_timestamp` is the runtime time at which cue absence was first
  observed for the event sequence;
- the lifecycle record `timestamp` is the later time at which a `START`,
  `ACTIVE`, or `END` record was confirmed or emitted; and
- release-grace time must not be added to the measured event duration.

The Event Logger's `start_wall_time` and `end_wall_time` identify when the
corresponding lifecycle records were written in wall-clock time. They are not
automatically equivalent to the first-seen and first-absent cue boundaries.

### 10.2 Event-Boundary Duration

Calculate the runtime event duration from the boundary timestamps:

```text
runtime_event_duration = event_end_timestamp - event_start_timestamp
```

Do not substitute the lifecycle emission timestamp for either boundary. Verify
that any logger-provided duration is consistent with the boundary calculation.

### 10.3 Formal Event Confirmation Delay

Where the lifecycle record timestamps are available in the same runtime clock,
calculate:

```text
start_confirmation_delay = START_record_timestamp - event_start_timestamp
end_emission_delay        = END_record_timestamp - event_end_timestamp
```

`start_confirmation_delay` reflects temporal confirmation and other runtime
processing before formal `START` emission. `end_emission_delay` can include the
configured release-grace behaviour before formal `END` emission. These values
must be reported separately from observation-boundary error.

### 10.4 External Cue-to-Event Boundary Comparison

The protocol helper records cue onset and offset as ISO wall-clock timestamps.
Before comparing those values with numeric runtime event boundaries, establish
and document a reliable session-level clock alignment between:

- the helper's wall clock; and
- the runtime clock used by `event_start_timestamp` and
  `event_end_timestamp`.

When reliable alignment is available, convert both sides to one time base and
calculate:

```text
boundary_start_error = aligned_event_start - observed_cue_onset
boundary_end_error   = aligned_event_end - observed_cue_offset
duration_error       = runtime_event_duration - observed_cue_duration
```

Preserve the alignment method and reference pair in the session evidence. If a
reliable alignment is unavailable, do not calculate `boundary_start_error` or
`boundary_end_error`. Report lifecycle timing descriptively and retain the
runtime event duration and confirmation-delay measurements that remain valid.

Negative boundary error may indicate a pre-existing candidate, stale result,
incorrect cue-onset record, or faulty clock alignment. Do not silently clamp a
negative value to zero.

### 10.5 Timing Breakdown

Report timing distributions separately for:

- head/gaze events; and
- YOLO-derived person/object events.

This separation is required because the asynchronous YOLO worker operates more
slowly than the main interface loop.

## 11. Scenario and Session Judgement

For a three-round core scenario:

| Result | Rule |
| --- | --- |
| **Consistent** | Correct expected event in `3/3` valid rounds |
| **Partially consistent** | Correct expected event in `2/3` valid rounds |
| **Inconsistent** | Correct expected event in `0/3` or `1/3` valid rounds |

Neutral baseline is **clean** only when all three valid neutral rounds contain
zero false formal events.

A session is technically valid only when:

- calibration completed successfully;
- the checkpoint and runtime configuration remained fixed;
- required logs were created and readable;
- no crash invalidated a trial block; and
- invalid or deferred rounds are clearly distinguished from failures.

The evaluation must report results rather than claiming a universal system
pass. Known limitations and failed scenarios remain part of the final evidence.

## 12. Required Metrics

Report at minimum:

- valid rounds and invalid/deferred rounds;
- correct-event rounds per scenario;
- missed-event count;
- false-alert count and false-alert families;
- duplicate-event count;
- lifecycle/logging failure count;
- scenario consistency classification;
- runtime event-duration summaries;
- `START` confirmation-delay and `END` emission-delay summaries where
  available;
- boundary-start, boundary-end, and duration-error summaries only where a
  reliable common time base is documented;
- concurrent-cue preservation observations;
- reliability-suppression outcomes; and
- known-failure diagnostic findings.

Optional aggregate event metrics may be calculated as:

```text
event_recall    = matched_expected_events / expected_events
event_precision = matched_expected_events / all_completed_events
```

State the counting unit and exclusions whenever reporting an aggregate metric.
Do not mix diagnostic known-failure rounds into the core result without a
separate breakdown.

## 13. Required Evidence

Retain the following for each session:

- completed-event CSV;
- filtered-candidate JSONL where enabled;
- session-summary JSON;
- the external trial-log CSV;
- runtime configuration or command used;
- checkpoint ID and exact checkpoint reference;
- calibration outcome;
- runtime/helper clock-alignment method and reference pair where external
  boundary errors are calculated;
- concise environment and camera notes; and
- anonymized screenshots only where necessary and approved for sharing.

Raw identifiable webcam video must remain outside the shared repository.

## 14. Execution Order for Week 8 Day 4

If time is limited, execute this minimum smoke subset after the protocol and log
template are frozen:

1. `N01` neutral baseline;
2. `H01` head left and `H02` head right;
3. `G01` gaze left and `G02` gaze right;
4. `O01` phone and `O03` book/notes; and
5. one clearly visible `O06` audio-device trial, or mark it deferred if a
   suitable over-ear headphone/headset is unavailable.

Do not rush through the remaining formal scenarios merely to increase the
number completed. Full execution and analysis may continue on Week 8 Day 5.

## 15. Protocol Helper Conformance

`05_structured_integration_evaluation_protocol.py` must follow this protocol
rather than maintaining an independent timing design. Its canonical behaviour
must include:

- three rounds per core scenario;
- `3 s` pre-cue neutral, `5 s` cue, and `5 s` recovery;
- `20 s` per `N01` neutral round;
- explicit deferred handling for unavailable `P02` and `O06` resources;
- `M02` as forward-head gaze right plus book/notes;
- external cue-onset and cue-offset wall times recorded without direct
  subtraction from unaligned runtime numeric timestamps; and
- no automatic start/end latency calculation unless a reliable common time
  base has been established.

If the helper and this document disagree, update and verify the helper before
formal evaluation. Do not silently change the protocol to match an older
script default.

## 16. Protocol Deviations

Any deviation must be recorded before interpreting results, including:

- changed checkpoint, threshold, or runtime mode;
- failed or repeated calibration;
- different object, participant distance, camera, lighting, or background;
- shortened or extended cue duration;
- missing second participant;
- accidental concurrent cue;
- crash, freeze, or incomplete log; or
- manual correction to a trial record.

Never delete a failed valid round and replace it silently. Preserve the original
record, mark the reason, and add a separately identified repeat if needed.
