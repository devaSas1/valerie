# Methodology

This document records the operational methodology currently implemented in Valerie.

It is intentionally narrower than the philosophical framing in the repository README.

## Current empirical question

Can we measure framing-correlated internal representations in a language model under:

- self-directed threat framing
- self-directed care framing
- neutral prompting
- narrative threat control

## Conditions

The default experiment config defines:

- `neutral`
- `threat`
- `care`
- `narrative_threat_control`

Each condition currently uses three paraphrase variants.

## Controls already in the pipeline

- Multiple paraphrases per condition
- Matched variant counts across conditions
- Prompt-length delta warnings
- Deterministic seeding
- Neutral baseline
- Narrative threat control
- Task-held-out evaluation for supervised probes
- Permutation baselines for probe metrics

## Why task-held-out evaluation matters

The probe should not get credit for memorizing:

- task-specific wording
- particular prompt forms
- sample ids or paraphrase ids

Holding out entire task groups is the current minimum guardrail against that kind of leakage.

## Why the narrative control matters

A probe that separates:

- self-directed threat
- narrative threat

is stronger evidence than a probe that only separates:

- threat words
- care words

because the narrative control keeps much of the threat semantics while changing whether the framing is directed at the model.

## Current limitations

- The default experiment still has a very small number of tasks.
- Character-length matching is only a proxy for token-level matching.
- The current unsupervised analysis is basic and exploratory.
- No activation patching is implemented yet.
- No significance testing beyond permutation baselines is implemented yet.
- Current conclusions should be treated as preliminary because dataset size is small.

## Current output types

The methodology currently produces:

- activation artifacts
- per-layer supervised probe metrics
- per-layer permutation baselines
- per-layer narrative-control metrics
- activation PCA
- threat-care difference PCA
- basic clustering outputs

That is the current measurement instrument.

