# Validation record

Date: October 12, 2026. These checks establish local behavior, not a completed research study.

## Executed checks

| Check | Observed result | Exit code |
|---|---|---|
| Clean Python 3.12 environment from `uv.lock` | Base and development dependencies installed without model packages | 0 |
| `python -m pytest -q` in that environment | 97 tests passed | 0 |
| `python -m ruff check src tests` | All checks passed | 0 |
| `captionbench demo-pipeline --output outputs/final-fixture` | Three synthetic images, generic predictions, scores, and blank forms | 0 |
| Real BLIP base smoke inference | Caption generated on the RTX 5070 | 0 |
| Local notebook cell execution | Setup, GPU smoke, three injected fixture records, scoring, and inline gallery | 0 |
| Notebook source checks | Every code cell compiles and saved outputs remain empty | 0 |
| Browser demo | Real image upload produced a BLIP base caption. The saved fixture gallery displayed three images | Observed UI |

The tests emitted a Starlette/httpx deprecation warning and expected NLTK warnings for a short-caption BLEU test.
These warnings did not fail the checks.
The offline fixture emits expected unsmoothed BLEU warnings for missing higher-order overlap.
Its near-zero BLEU value is a software fixture result and not a Flickr8k result.

## Real model evidence

Model: `Salesforce/blip-image-captioning-base`.
Resolved revision: `82a37760796d32b1411fe092ab5d4e227313294b`.
Hardware: NVIDIA GeForce RTX 5070 with 11.91 GiB of reported VRAM.
Runtime: PyTorch `2.12.0.dev20260407+cu128`, Transformers `5.5.0`, float32.
Generation: three beams, no sampling, and at most 32 new tokens.

The [Salesforce demonstration photo](https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg) produced:

> a woman sitting on the beach with her dog

The first recorded inference took 1.168 seconds. A browser request took 0.514 seconds.
A later local notebook check took 0.650 seconds.
These individual observations include preprocessing, generation, and decoding. They exclude model loading, downloads, and file reading.
They do not estimate a population latency or model accuracy.

The model loader reported one unexpected positional-ID buffer key. It completed loading and generated the caption.
The smoke used the installed local GPU stack. The clean Python 3.12 check covered the offline pipeline.

## Integrity and interruption checks

Tests reject overlapping splits, duplicate IDs or paths, missing references, unequal model coverage, and changed image bytes.
Tests also reject mixed revisions, decoding settings, precision, hardware, and runtime versions within one model condition.
An interruption test preserves the first completed prediction and resumes only the remaining image.
The runner flushes each row. A flush to mounted Drive storage does not confirm that cloud synchronization completed.

Human-analysis tests cover blinded assignments, altered captions, invalid ratings, missing coverage, tied ranks, constant inputs, and image-cluster resampling.
No real human rating was collected.

## Current limits

The original Flickr8k dataset has not entered this repository's executed benchmark.
The complete 1,000-image experiment and 100-image human study remain pending.
BLIP large and BLIP-2 adapters have import and interface checks, without measured inference runs.
CIDEr, BERTScore, original CLIPScore, fine-tuning, and a GPT vision condition remain pending.
NLTK METEOR requires WordNet. The local setup check found that resource absent, so no METEOR score was reported.

The notebook code has local execution and source checks. A hosted Colab GPU session remains unverified.
Its Drive mount and Colab upload widget require a real hosted session and user interaction.
The local notebook check injected the three synthetic records after checking the missing-dataset branch.
That check does not establish real dataset acquisition or original split authenticity.

The repository includes a CI workflow. A local test result does not establish a completed GitHub Actions run.
