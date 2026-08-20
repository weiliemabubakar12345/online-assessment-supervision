# Computer Vision Evaluation

This directory separates evaluation design, reusable templates, and reviewed
results so that a blank protocol artifact cannot be mistaken for completed
evidence.

## Tracked Evaluation Files

| Path | Purpose |
| --- | --- |
| `protocols/final_end_to_end_protocol.md` | Frozen 30-scenario, 90-trial end-to-end procedure |
| `templates/end_to_end_trial_log.csv` | Blank trial-recording template |

The template contains planned rows and is not a result file. Preserve it as a
reusable template; place the reviewed filled master log under `results/` with a
different filename.

## Recommended Result Layout

```text
evaluation/
├── protocols/
│   └── final_end_to_end_protocol.md
├── templates/
│   └── end_to_end_trial_log.csv
└── results/
    ├── README.md
    ├── final_end_to_end_trial_log.csv
    └── final_end_to_end_summary.csv
```

The audited combined-branch archive dated 20 August 2026 does not contain the
three `results/` files shown above. Add them only from the reviewed canonical
evaluation records; do not infer or copy final metrics from the blank template.

The protocol also describes the expected conformance of
`05_structured_integration_evaluation_protocol.py`, but that helper is not in
the audited archive. If it is part of the reproducibility handover, review it,
remove machine-specific paths, and place it under `evaluation/scripts/`.

## Minimum Result Documentation

The result README should state:

- evaluation date and evaluator;
- frozen checkpoint and runtime configuration;
- participant/setup scope and privacy treatment;
- scenario and trial denominator;
- definitions for expected-event detection, miss, false alert, fragmentation,
  lifecycle correctness, and timing measures;
- treatment of familiarisation attempts, repeated trials, and diagnostic
  repeats;
- distinction between formal results and supplemental diagnostic evidence; and
- known limitations and conditions that affect interpretation.

## Sharing Boundary

Commit only reviewed, anonymized tables or figures. Raw event-log sessions,
screenshots, webcam media, private paths, and assessment content remain outside
ordinary Git tracking. Preserve the original formal denominator and results;
controlled repeats may explain sensitivity but must not silently replace weaker
formal outcomes.
