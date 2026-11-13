# CaptionBench

Compare image captioning models and test how automatic metrics agree with human ratings.
This repository starts a CSE434 study on Flickr8k. The main workflow runs in one Colab notebook.

## Start here

Open [the study notebook in Colab](https://colab.research.google.com/github/jhunter11/cse434-fp/blob/midterm-checkin/notebooks/study.ipynb).
Choose **Runtime > Run all**. You do not need to edit a setting.
The notebook asks Colab for a T4 GPU. If its GPU check reports no GPU, choose **Runtime > Change runtime type > T4 GPU** and run all cells again.

The default profile is `quick`. It is a short demo that ends with a score table and a page of photos with each model's caption.
To run the study, set `PROFILE` to `midterm` or `full` in the Settings cell.

The notebook needs no account and no uploaded file. Google Drive is optional.
With `SAVE_TO_DRIVE` on, the notebook copies the small result files to `My Drive/CaptionBench` after each stage, and a later run continues from them.
It copies this repository, downloads Flickr8k from a public mirror, and checks the archive hashes.
It then trains and scores every condition and shows the tables, the figures, and a page of real captions.
Near the end, a cell downloads the results as a zip file. The last cell releases the GPU.

| Profile | What runs | Estimated time on a T4 |
|---|---|---|
| `quick` | A wiring check. Its scores are not study results. | 15 to 20 minutes |
| `midterm` | The conditions in the midterm report | 3 to 4 hours |
| `full` | `midterm`, a wider decoder on ConvNeXt features, and a fine-tuned BLIP large | 5 to 6 hours |
| `improve` | Our decoders against released BLIP: a wider decoder, three stronger frozen encoders (MobileNetV3, ConvNeXt, DINOv2), and beam search. No fine-tuning. | 30 to 60 minutes, not measured |

The notebook calls one command, which also runs on a local GPU:

```bash
python scripts/run_study.py --profile quick
```

The command skips each stage whose output exists. Run it again after an interruption.

Status of the checks: a local run confirmed the download, the hash check, the manifest, and each stage script.
A hosted Colab run of the notebook remains unverified. The times are estimates from a local RTX 5070.

The earlier [baseline notebook](notebooks/baseline.ipynb) remains for the single-image demo and the rating forms. It can use Drive.

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
A plain model comparison repeats that research, so the midterm changed the main question.

How much of each automatic caption score can a model earn without its own image?
How does that share change from a decoder trained from scratch, to pretrained BLIP, to fine-tuned BLIP?
`scripts/grounding_gap.py` scores each caption on its own image and on other images to answer this.
The [midterm results](results/midterm/) hold the aggregate scores. The report source is in [report/](report/).
Read the [research review](docs/research.md) for prior methods, primary sources, and limitations.

## Scope and status

| Component | Current implementation |
|---|---|
| Colab | `notebooks/study.ipynb` runs the whole study. Google Drive is optional. `scripts/run_study.py` is its one command |
| Data | Flickr8k import from local files, fixed split manifests, image hashes, and validation |
| Baselines | Generic nonvisual control and pretrained BLIP base |
| Larger models | BLIP large and BLIP-2 adapters for explicit runs |
| Scoring | Corpus and per-caption BLEU-4, CIDEr-D, optional NLTK METEOR, and optional CLIPScore |
| Custom decoders | `scripts/train_decoder.py` trains plain and relation decoders from scratch on frozen ResNet18 or ConvNeXt features |
| Fine-tuning | `scripts/finetune_blip.py` trains BLIP base or BLIP large on the Flickr8k training split |
| Grounding gap | `scripts/grounding_gap.py` scores each caption on its own image and on other images. `scripts/donor_rules.py` selects the nearest and the farthest other image |
| Human study | Blinded rating forms and separate accuracy/detail correlations |
| Demo | Local image upload, model selection, caption output, and saved comparison gallery |
| Planned extensions | BERTScore, BLIP-2 scores, the sealed test run, and the human rating study |

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
