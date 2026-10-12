# Research background and experiment plan

Research checked: October 12, 2026. This document proposes an experiment. It contains no measured model results.

## Project purpose

Build a reproducible comparison of released image captioning systems on the same Flickr8k images.
Measure caption quality, agreement with human ratings, inference time, and memory use.
The main question is whether better automatic scores correspond to more accurate and useful descriptions.

The project has three research questions:

1. How do BLIP base, BLIP large, and BLIP-2 OPT-2.7B compare under a fixed evaluation protocol?
2. Which automatic metrics agree most with human ratings of accuracy and detail?
3. What quality differences justify extra inference time and memory on the available hardware?

An optional vision API adds a fourth released system. It does not replace the three local model conditions.
Any API condition needs a recorded model version, prompt, image settings, response limits, date, and actual usage.
No API result or cost appears in the baseline until that run occurs.

## What other researchers have done

Image captioning and human agreement studies are established research topics.
The Flickr8k authors already studied image descriptions, caption ranking, human judgments, and automatic evaluation in 2013.
Their benchmark contains 8,000 images with five descriptions each.
Their method used a shared image and text space to rank existing captions.
This differs from the new sentences that generative models produce. [Flickr8k paper](https://www.jair.org/index.php/jair/article/view/10833)

BLIP introduced a vision and language model with multiple training objectives.
It also introduced CapFilt, which creates synthetic captions and filters noisy image and text pairs.
Its image captioning experiments already report established caption metrics.
The paper supports the choice of BLIP as a baseline, but its published scores do not predict this project's results.
Dataset, checkpoint, preprocessing, and evaluation differences affect that comparison. [BLIP paper](https://proceedings.mlr.press/v162/li22n.html)

BLIP-2 connects a frozen image encoder to a frozen language model through a trainable Querying Transformer.
The authors train this bridge in two stages to reduce the cost of vision and language pretraining.
The released OPT-2.7B condition adds a larger language model to the comparison.
Its architecture suggests a useful contrast with BLIP. It does not guarantee better captions on Flickr8k. [BLIP-2 paper](https://proceedings.mlr.press/v202/li23q.html)

Metric research also covers this project's main question.
BERTScore compares contextual token representations rather than exact words alone.
CLIPScore compares a caption with the image and requires no reference caption.
Both papers compare metrics with human judgments.
The project can test their behavior on its own model outputs and rating rubric. [BERTScore](https://github.com/Tiiiger/bert_score), [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/)

The contribution is a transparent student experiment with shared images, saved outputs, clear rating criteria, and reproducible analysis.
It is not a new captioning architecture or the first comparison of human ratings with automatic metrics.
An exact prior study with the same checkpoints, split, decoding settings, and rubric was not established by this source review.
That does not establish novelty or the absence of such a study.

## Proposed baseline matrix

| Condition | Released checkpoint | Role | Interpretation limit |
| --- | --- | --- | --- |
| BLIP base | `Salesforce/blip-image-captioning-base` | First real inference baseline | The card identifies a COCO captioning checkpoint with a ViT base backbone. |
| BLIP large | `Salesforce/blip-image-captioning-large` | Same family with a ViT large backbone | Extra size alone does not explain differences in released training recipes. |
| BLIP-2 | `Salesforce/blip2-opt-2.7b` | Frozen vision encoder, Q-Former, and OPT comparison | The card labels this checkpoint pretrained only. It is not a matched COCO adaptation. |
| Vision API, optional | Explicit version chosen before the run | External system comparison | Training data, hardware, and service changes limit direct attribution. |

Checkpoint details come from their model cards. [BLIP base](https://huggingface.co/Salesforce/blip-image-captioning-base), [BLIP large](https://huggingface.co/Salesforce/blip-image-captioning-large), [BLIP-2 OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b)

Compare these as released systems.
Do not call this a controlled test of architecture, parameter count, or pretraining data.
Use no Flickr8k training for the first baseline.
Record whether each checkpoint already received caption task training elsewhere.
Zero-shot here means no adaptation on Flickr8k. It does not prove that pretraining excluded these public images.

Use deterministic generation for the main experiment.
Fix the beam count, sampling flag, output limit, prompt, processor, dtype, and model revision before the full test run.
Start with each model's image captioning interface and record any model-specific text prefix.
Use validation images for prompt or decoding choices.
Treat a new prompt, a new precision, or quantization as a separate condition.
The same token limit does not imply the same caption length across different tokenizers.

## Data, splits, and public sharing

Use the provided Flickr8k split lists: 6,000 training images, 1,000 validation images, and 1,000 test images.
The test set must follow the actual `Flickr_8k.testImages.txt` list.
Do not replace it with a random 1,000-image sample or split captions independently of images.
Validate every test image and all five references before inference.
Stop on missing files, duplicate IDs, or split overlap.
Archive a manifest hash, original filenames, source URL, acquisition date, and local file checksums for each run.

Dataset archives and mirrors can contain extra images or different counts.
The source split lists determine the evaluation set.
Record the counts found in the acquired files before describing them as a verified local dataset.
Neither this research note nor an empty dataset folder confirms acquisition.

The original authors state that they do not own the image copyrights.
Their download page describes educational and noncommercial research use and points to Flickr terms.
Their caption page separately identifies an Attribution-ShareAlike license.
A mirror's blanket license label does not settle the rights to all photos and captions. [Original download terms](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html), [Original caption notice](https://hockenmaier.cs.illinois.edu/8k-pictures.html)

Keep dataset photos, reference captions, downloaded weights, and local caches outside public Git.
Publish code, configuration, acquisition instructions, and result metadata.
Review any example image's rights before sharing it in a report.
The repository license applies to the project's own code. It does not relicense third-party data or models.

## Automatic metrics and their limits

All five reference captions must remain available to each reference metric.
Preserve raw generated captions and record the normalization policy.
Do not lowercase semantic metric inputs merely because an overlap metric uses lowercase text.

| Metric | What it measures | Main limitation | Required report detail |
| --- | --- | --- | --- |
| BLEU-4 | Clipped overlap of word sequences up to length four, with a brevity penalty | A valid paraphrase can score poorly. Short individual captions often have no four-word match. | Corpus score, tokenizer, smoothing, reference handling, and scale |
| METEOR | Aligned word matches, precision, recall, and ordering | Stem and synonym resources change matches. It still compares text with text. | Implementation, resource versions, and aggregation |
| CIDEr | Agreement with references through weighted word sequences | It rewards reference consensus and depends on corpus document frequencies. | Exact variant, tokenizer, document-frequency corpus, and scale |
| BERTScore F1 | Contextual token similarity with references | Semantic similarity does not establish whether an object appears in the image. | Encoder revision, layer, hash, IDF, rescaling, and reference policy |
| CLIPScore | Image and caption compatibility in a shared embedding space | Image compatibility does not certify every object, count, action, or relation. | Encoder, prompt, preprocessing, formula, dtype, and scale |

Metric definitions: [BLEU](https://aclanthology.org/P02-1040/), [METEOR](https://aclanthology.org/W05-0909/), [CIDEr](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.html), [BERTScore](https://github.com/Tiiiger/bert_score), [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/).

Report corpus BLEU-4 in the main result table.
A mean of sentence BLEU scores is a different statistic.
If sentence BLEU supports correlation analysis, label it separately and specify smoothing.
Do not compare raw values from different metrics as if they share a scale.

Compute CIDEr for the full fixed corpus, then retain its per-image scores.
Do not initialize a fresh CIDEr corpus for each image.
The official implementation computes document frequencies across reference sets and uses a length penalty.
Its variant and output scaling must appear in the run metadata. [COCO caption evaluation code](https://github.com/tylin/coco-caption)

The original COCO evaluator uses a Java METEOR implementation and a Stanford tokenizer.
A Python METEOR implementation can differ.
Label the implementation actually used, and avoid claims of direct benchmark parity without a parity check. [COCO caption evaluation code](https://github.com/tylin/coco-caption)

BERTScore's original package accepts multiple references and selects matching reference scores.
Record the returned configuration hash and use one fixed encoder configuration for every caption model.
CLIPScore's original code uses ViT-B/32, the prefix `A photo depicts`, and `2.5 * max(0, cosine_similarity)`.
Raw CLIP cosine similarity is a different statistic.
Retain original image files because resizing and recompression can alter the score. [BERTScore code](https://github.com/Tiiiger/bert_score), [CLIPScore code](https://github.com/jmhessel/clipscore)

Caption hallucination remains a separate concern.
Rohrbach and colleagues show that strong sentence metric scores do not always imply fewer invented objects.
Their CHAIR metric uses verified object labels on COCO.
Do not claim standard CHAIR results on Flickr8k without suitable object annotations.
For this project, record human error tags for invented objects, wrong counts, wrong attributes, and wrong actions. [Object Hallucination in Image Captioning](https://aclanthology.org/D18-1437/)

## Human evaluation plan

Choose 100 test images with a saved random seed before inspecting model captions.
Use the same images for every model.
Keep a separate short validation pilot for rubric calibration.
Hide model names, automatic scores, and reference captions during test ratings.
Show the image and one candidate caption at a time.
Randomize presentation order and save the mapping privately until ratings close.

Each caption should receive at least two independent ratings.
All team members can rate the shared set when feasible.
With three models and two raters, this requires 600 caption ratings, each with two scale responses.
Team members who know the models are a convenience panel. Report that limit.
Save anonymous rater IDs, scores, error tags, order, and the rubric version.

Use two distinct criteria:

| Score | Accuracy: does the caption agree with visible evidence? | Detail: how much useful, correct information does it include? |
| --- | --- | --- |
| 1 | The main scene or subject is wrong. | It gives no useful correct description. |
| 2 | It has major object, action, or relation errors. | It gives little useful correct information. |
| 3 | It captures the main scene with a material error. | It identifies the main subject or action. |
| 4 | It is accurate except for a minor error. | It adds useful correct attributes or context. |
| 5 | Its visible claims are accurate. | It gives several useful correct details without needless repetition. |

These anchors are a proposed project rubric. They are not published measurement results.
Do not reward caption length alone or penalize a concise caption solely for omitted background objects.
Record the rubric, participant selection, sample method, and study procedure for reproducibility.
Report rater agreement for each criterion and inspect disagreements before interpretation. [Human evaluation reporting guidance](https://aclanthology.org/2020.inlg-1.23/)

Average raters within each image and model for the requested Spearman analysis.
The 1-5 scales are ordinal. Include rating distributions and medians beside any means.
Compute a separate association with accuracy and with detail.
Use per-caption metric scores for this analysis, not three model-level averages.
Report sample size, ties, missing ratings, uncertainty, and correlation direction.
Use image-grouped resampling because captions from different models share an image.
Treat pooled and within-model correlations separately. A pooled relationship can reflect differences between models.
Correlations describe association in this panel and sample. They do not establish metric reliability for all caption tasks.

## Compute and fine-tuning decisions

Start with a few validation images to measure memory and latency.
Use batch size one for comparable interactive latency.
Record model loading separately from steady-state inference.
Warm the model before timing, synchronize GPU work, and include processor and decoding time in the chosen boundary.
Report the exact boundary, device, dtype, batch size, caption length, median latency, tail latency, and peak memory.
Keep metric computation time separate from caption inference time.

The BLIP-2 card estimates about 7.21 GB of float16 parameter storage.
That estimate excludes extra runtime memory and does not guarantee a fit on an 8 GB GPU.
CPU execution and quantization are possible alternatives, but they change latency and numerical conditions.
Measure feasibility before promising a full run. [BLIP-2 memory table](https://huggingface.co/Salesforce/blip2-opt-2.7b)

Do not make fine-tuning part of the first MVP.
It adds optimizer memory, training choices, overfitting risk, and another experimental factor.
If the frozen baseline works, a later extension can train on Flickr8k training images and select settings on validation images.
The original test images must remain untouched until the final comparison.
Keep the unfitted checkpoint results so the extension has a valid reference point.

## Evidence needed to finish

| Stage | Evidence | What it establishes |
| --- | --- | --- |
| Offline fixture | Synthetic images and captions with pipeline checks | The file formats and analysis paths work. It establishes no real model quality. |
| First model baseline | Saved captions, manifest, timing, and exact checkpoint metadata | Real inference on the named local images |
| Full automatic comparison | Complete paired outputs and all requested metric artifacts | Measured results for the fixed test set |
| Human study | Closed blind ratings, agreement, correlations, and uncertainty | Agreement within the reported sample and panel |
| Shareable report | Reproduction commands, source citations, limits, and reviewed public files | An inspectable class project deliverable |

The first usable MVP is an inference and evaluation pipeline with honest evidence labels.
A running app or a passing fixture alone does not complete the empirical study.
The next research decision follows the first measured baseline: confirm feasibility, inspect errors, and lock the full comparison protocol.

The machine-readable bibliography in [sources.json](sources.json) records the selected primary sources and their roles.
