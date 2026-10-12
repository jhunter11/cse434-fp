# CaptionBench

Compare image captioning models and test how automatic metrics agree with human ratings.
This repository starts a CSE434 study on Flickr8k. The main workflow runs in one Colab notebook.

## Start here

Open [the Colab notebook](https://colab.research.google.com/github/jhunter11/cse434-fp/blob/main/notebooks/baseline.ipynb).
Choose a GPU runtime and run its cells in order.
The setup cell installs the project automatically. No manual source file copying or web server is required.

The notebook has settings, a real BLIP smoke image, dataset preparation, resumable inference, scoring, an inline gallery, and a rating form.
It also includes an optional image-upload demo inside the notebook.
Start with the default 20-image validation pilot. After settings are fixed, run the original 1,000-image test split.

Use Drive storage for completed captions and experiment records.
The runner saves each completed image and verifies it before resuming.
Different hardware or model settings require a new run folder.
Colab GPU types and runtime limits vary. See the [official FAQ](https://research.google.com/colaboratory/faq.html).

The notebook has local execution checks. A hosted Colab GPU execution remains unverified.

## Optional local workflow

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --locked --extra dev
uv run captionbench demo-pipeline --output outputs/fixture
uv run captionbench-web
```

Open <http://127.0.0.1:8000>. Choose an image and the generic control to check the interface without model downloads.
The control returns one fixed sentence. It does not inspect the image.
The fixture uses original synthetic images. Its scores check software behavior and do not measure Flickr8k performance.
Use a new output folder for each fixture run. The pipeline preserves existing rating files.

To run a real captioning model, install the model dependencies:

```powershell
uv sync --locked --extra dev --extra models
uv run captionbench-web
```

Select **BLIP base** and upload an image. The first request downloads public model weights.
The server processes uploads locally and removes its temporary upload after the request.
Model downloads remain in your Hugging Face cache.
The server binds to the local computer at `127.0.0.1`.

## Study question

Which released model offers useful captions at a practical compute cost?
Which automatic metrics agree with human ratings of accuracy and detail?
Image captioning and metric evaluation already have extensive research.
Our contribution is a reproducible comparison with explicit controls and failure analysis.
Read the [research review](docs/research.md) for prior methods, primary sources, and limitations.

## Scope and status

| Component | Current implementation |
|---|---|
| Colab | One notebook with GPU checks, Drive storage, incremental results, and an inline demo |
| Data | Flickr8k import from local files, fixed split manifests, image hashes, and validation |
| Baselines | Generic nonvisual control and pretrained BLIP base |
| Larger models | BLIP large and BLIP-2 adapters for explicit runs |
| Scoring | Corpus BLEU-4 and per-caption BLEU-4, with optional NLTK METEOR |
| Human study | Blinded rating forms and separate accuracy/detail correlations |
| Demo | Local image upload, model selection, caption output, and saved comparison gallery |
| Planned extensions | CIDEr, BERTScore, original CLIPScore, fine-tuning, and a budgeted GPT vision condition |

The repository does not claim a completed 1,000-image benchmark or completed human study.
Missing metric values mean unmeasured. They do not mean zero.
The [experiment plan](docs/experiment-plan.md) defines the study and promotion gates.
The [validation record](docs/validation.md) records executed checks and evidence limits.

## Prepare Flickr8k

Acquire Flickr8k under the source terms. Keep the images, captions, and split files outside Git.
Use the original test split rather than choosing a new random test set.
The importer accepts the common `captions.txt` CSV and the original `Flickr8k.token.txt` format.

```powershell
uv run captionbench prepare-flickr8k --images data/Flicker8k_Dataset --captions data/Flickr8k.token.txt --train-split data/Flickr_8k.trainImages.txt --val-split data/Flickr_8k.devImages.txt --test-split data/Flickr_8k.testImages.txt --output data/flickr8k.jsonl
uv run captionbench validate --manifest data/flickr8k.jsonl
uv run captionbench generate --manifest data/flickr8k.jsonl --model generic --output outputs/generic.jsonl
uv run captionbench generate --manifest data/flickr8k.jsonl --model blip-base --output outputs/blip-base.jsonl
```

Use `captionbench --help` and each command's `--help` for the exact options.
Run a small pilot with `--limit` before the full split.
Match the image IDs across every compared model and record the model revision from the output.
Generation writes one prediction at a time. Repeating the same command resumes validated completed images.
The runner rejects changed images, model settings, or runtime conditions.

Each manifest row has an `image_id`, `image_path`, `split`, and five `references`.
Each prediction row has an `image_id`, `model`, `caption`, and inference metadata.
The pipeline rejects duplicate IDs, missing captions, and unequal model coverage.
No component trains on test references.

## Evaluate and inspect

```powershell
uv run captionbench evaluate --manifest data/flickr8k.jsonl --predictions outputs/generic.jsonl outputs/blip-base.jsonl --output outputs/comparison.json
uv run captionbench-web --manifest data/flickr8k.jsonl --predictions outputs/blip-base.jsonl
```

Corpus BLEU uses all five references and unsmoothed BLEU-4.
Per-caption BLEU uses a documented smoothing rule for human correlation.
These scores serve different purposes. Do not average sentence BLEU and call it corpus BLEU.
Optional METEOR uses NLTK and WordNet. It is not the Java COCO METEOR scorer.

```powershell
uv run captionbench ratings-template --manifest data/flickr8k.jsonl --predictions outputs/generic.jsonl outputs/blip-base.jsonl --sample-size 100 --output-dir ratings/study-01
uv run captionbench correlate --ratings ratings/study-01/completed-ratings.csv --private-key ratings/study-01/private-model-key.json --evaluation outputs/comparison.json --output outputs/human-correlations.json
```

Each rater fills a separate copy of `ratings.csv` with a pseudonymous rater ID and integer scores from 1 to 5.
Combine the completed rows into `completed-ratings.csv` and retain one header row.
Raters need the selected images and usable paths. Do not send the private key, predictions, or gallery before ratings finish.

## Public repository and data

The MIT license covers this repository's original code. Models and datasets retain their source terms.
Flickr image copyrights remain with their owners. A mirror's license label does not establish image rights.
This repository excludes raw images, reference captions, model weights, personal ratings, credentials, and the private proposal.
See [data handling](docs/data.md) before sharing results or a gallery.

## Team workflow

Divide substantive technical work across data/model inference, metric/statistical evaluation, and human study/demo integration.
Each owner should review another owner's work and reproduce one result.
Keep model selection, prompts, decoding, split IDs, and metric settings fixed before the full test run.

```powershell
uv run pytest -q
uv run ruff check src tests
```

CI runs the same offline fixture and tests. It does not download models or call a paid API.
