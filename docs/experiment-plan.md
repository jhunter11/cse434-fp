# Experiment plan

## Questions and scope

Compare released image captioners on the same Flickr8k test images.
Measure caption quality, inference cost, and agreement between automatic scores and human ratings.
This study compares systems. It does not isolate architecture, dataset size, or parameter count as causal factors.

The full target contains 1,000 original test images with five reference captions per image.
The human study target contains 100 shared test images with each model's caption.
Use accuracy and detail ratings from 1 to 5 as separate outcomes.
Do not merge them into a single score without a declared rule.

## Staged experiment

1. Run the offline fixture and validate failures for invalid data.
2. Run BLIP base on a small public image sample to check inference and timing.
3. Import the approved dataset and freeze the original test IDs and source hashes.
4. Run a 20-image pilot on validation images to choose feasible model settings.
5. Freeze decoding, metric implementations, and the human rating rubric.
6. Run the full fixed test split for each feasible model.
7. Select 100 test IDs with seed 42 before reading model performance.
8. Collect blinded ratings and inspect rater agreement.
9. Compute correlations, uncertainty, and category error analysis.
10. Report measured results and failed or omitted conditions.

## Model matrix

| Condition | Purpose | Gate |
|---|---|---|
| Generic fixed caption | Nonvisual negative control | Offline fixture passes |
| BLIP base | First pretrained baseline | Successful local inference and manifest validation |
| BLIP large | Larger released captioning system | Pilot fits memory and measured runtime |
| BLIP-2 OPT-2.7B | Frozen language model plus vision bridge | Pilot fits memory with recorded precision |
| Fine-tuned BLIP base | Domain adaptation comparison | Train/validation separation and saved training configuration |
| GPT vision | Optional hosted system comparison | Explicit model snapshot, user budget, approved image transfer, and cost record |

BLIP base and large captioning checkpoints have prior COCO training.
The BLIP-2 checkpoint has different training history. Describe this confound in the report.
Quantization changes the condition. Record it separately from a full precision run.
Do not install an API key or invoke a paid model as part of the default workflow.

## Reproducibility record

Save the dataset source, source revision, split file hashes, manifest hash, image IDs, and reference count.
Save each model ID, resolved revision, package versions, device, precision, prompt, and decoding parameters.
Keep raw predictions so metric changes do not require inference again.
Measure warm inference separately from model loading and downloads.
Record batch size and CUDA synchronization with timing.
Use the same image IDs and decoding policy across comparable conditions.

Pretraining contamination remains unknown unless the source gives auditable overlap evidence.
Call the baseline pretrained inference or transfer evaluation. Do not claim unseen-image zero-shot generalization.

## Metrics

| Metric | Role | Main limitation |
|---|---|---|
| Corpus BLEU-4 | Model-level reference overlap | Short captions and valid paraphrases can score poorly |
| Sentence BLEU-4 | Per-caption correlation | Smoothing and tokenization affect ranking |
| NLTK METEOR | Optional reference alignment | WordNet resources and implementation affect comparability |
| CIDEr | Planned consensus metric | Requires declared corpus-wide document frequency |
| BERTScore | Planned semantic reference comparison | Encoder, IDF, and rescaling change scores |
| Original CLIPScore | Planned image-caption compatibility | Vision-language biases and coarse image matching remain |

Record unimplemented or unrun metrics as unmeasured.
Do not substitute a convenient embedding cosine for a named published metric.
Compare per-metric model rankings and report disagreement rather than choosing a favorable score after evaluation.

## Human rating protocol

Show the image and one candidate caption without a model name or automatic score.
Randomize caption order and retain the private model mapping.
Use the same image sample for every model. Prefer at least two raters per image-caption pair.
Do not show reference captions during the primary image-grounded rating.

Accuracy measures whether claims match the image. A fluent invented object should reduce this score.
Detail measures useful, correct specificity. Verbosity alone should not raise this score.
Keep raw per-rater observations and missing scores explicit.
Do not fill missing ratings with neutral values.

Compute Spearman correlation between per-caption metric values and human means.
Do not compute the primary correlation across only a few model-level averages.
Captions for one image are dependent. Use image-level cluster resampling for confidence intervals when available.
Report each model's correlation as a diagnostic because pooled correlations can reflect model identity.
Constant scores have undefined correlation. Report that condition explicitly.

Tag counting, rare objects, actions, and scenes with people for error analysis.
Add explicit hallucination tags when a caption mentions unsupported content.
Treat this small team-rated study as evidence about this sample and rubric.
It does not establish accessibility quality for screen reader users.

## Substantive team ownership

| Workstream | Technical contribution | Review responsibility |
|---|---|---|
| Data and models | Dataset provenance, split validation, model adapters, timing, and optional training | Reproduce a scoring result |
| Metrics and statistics | Scorer validation, per-caption data, correlation, uncertainty, and error analysis | Audit split and timing controls |
| Human study and demo | Blinding, rating collection, gallery, API integration, and deployment instructions | Reproduce one inference run |

These are proposed workstreams. They do not assign teammates or claim completed contributions.
