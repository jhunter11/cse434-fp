# Data handling

Acquire the dataset from a source whose access and terms you can inspect.
The [original Flickr8k project page](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html) provides the research provenance.
The [original annotation page](https://hockenmaier.cs.illinois.edu/8k-pictures.html) describes caption licensing.
Image rights and caption rights are separate.

Keep raw photos, reference captions, split files, and source archives under `data/` or another private local directory.
Git ignores `data/`, `outputs/`, `ratings/`, `checkpoints/`, and common credential files.
Review `git status` and the staged diff before each push.

Use original train, development, and test split files.
The importer rejects overlapping image IDs across splits.
It validates exactly five nonempty reference captions and checks local image paths.
Its metadata records input hashes and the manifest hash.
Mirrors can differ in image counts, names, splits, and captions.
Do not infer original split membership from a mirror's row order.

The offline fixture contains code-generated geometric images and original authored references.
These examples are synthetic software fixtures, not Flickr8k samples.
A model's score on them does not establish performance on photographs.

The Salesforce demonstration image can check model loading and inference.
Keep that downloaded image in ignored storage and preserve its source URL in the run record.
It is not part of the Flickr8k study and has no reference-based benchmark score here.

Human forms belong in `ratings/`. Do not commit names, email addresses, or identifiable rater IDs.
Use pseudonymous rater IDs for analysis and keep any identity mapping separately.
Share a gallery publicly only after checking rights for its photos and captions.

The public repository contains original code, public source links, and aggregate validation evidence.
It excludes the school email, the private proposal, and private source packets.
