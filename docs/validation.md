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

## Midterm run, November 11, 2026

These checks ran on the 200 confirmation images of the Flickr8k validation split.
The 1,000 test images stayed unused.

| Check | Observed result |
|---|---|
| `python -m pytest -q` in the Python 3.12 environment | 107 passed, 3 skipped (the skipped tests need PyTorch) |
| Custom decoder tests with PyTorch on Python 3.14 | 31 passed |
| `python -m ruff check src tests scripts` | All checks passed |
| Flickr8k import | 8,000 images in the manifest: 6,000 train, 1,000 validation, 1,000 test |
| `cider_d` against `pycocoevalcap` | Largest per-image difference 6.7e-16 on 12 prediction files |
| BLIP base and BLIP large on the validation split | 1,000 captions each on a Colab Tesla T4 |
| Nine frozen decoder checkpoints | Imported from the October 22 search with matching SHA-256 values |
| BLIP base fine-tuning, seeds 17, 29, and 43 | Search loss 3.26 before training, 1.83 to 1.84 after epoch 1, 1.81 after epoch 2 |
| `scripts/grounding_gap.py` | Own-image and other-image scores for 12 prediction files |
| `notebooks/study.ipynb`, profile `midterm`, on Colab | All stages completed in 233 minutes on November 12, 2026 |

The aggregate scores are in [results/midterm](../results/midterm/).
`grounding-gap.json` holds each model. `summary.json` holds the family means that the report cites.

One test found a defect. A shift larger than the image count left the image order unchanged.
The script now reduces each shift by the image count. The 200-image run used shifts below 200, so its numbers did not change.

The first fine-tuning run used batch size 16 and filled the 12 GB of GPU memory. It was stopped.
The reported run used batch size 8.

Limits of this run:

- Each trained condition has three seeds.
- The reported numbers come from the Colab run. An earlier local run on an RTX 5070 gave decoder scores that differ by up to 0.2 BLEU-4 points.
- The intervals resample images. They exclude training-seed variation.
- The other-image analysis began after the team saw validation scores.
- BERTScore and BLIP-2 remain unmeasured. No human rating was collected.

## Current limits

The October 12 limits changed with the midterm run above.
The complete 1,000-image test experiment and the 100-image human study remain pending.
The BLIP-2 adapter has import and interface checks, without a measured inference run.
BERTScore and a GPT vision condition remain pending.
NLTK METEOR requires WordNet. The midterm run installed it for Python 3.14 and reported METEOR.

The `midterm` profile ran in a hosted Colab GPU session. The `quick` profile ran on a local CPU only.
The Drive save worked in that session. The Colab upload widget remains untested.
The local notebook check injected the three synthetic records after checking the missing-dataset branch.
That check does not establish real dataset acquisition or original split authenticity.

The repository includes a CI workflow. A local test result does not establish a completed GitHub Actions run.
