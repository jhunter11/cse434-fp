# Midterm report

`midterm.tex` is the source of the midterm check-in report.
Git ignores the PDF files. Build them again with these commands from the repository root.

```bash
python scripts/grounding_gap.py --manifest data/flickr8k.jsonl --split val --image-ids outputs/midterm/confirm-ids.json --predictions "outputs/midterm/confirm-*.jsonl" "outputs/midterm/val-*.jsonl" --output results/midterm/grounding-gap.json
python scripts/make_midterm_figures.py --results results/midterm --report report
tectonic -X compile report/midterm.tex --outdir report/build
```

The first command needs the local Flickr8k files and the prediction files.
The second command needs only the tracked files in `results/midterm`.
It writes the figures and `results-table.tex`.

The `\teamname` and `\roles` lines at the top of `midterm.tex` set the title block. `\roles` lists the three names with no roles. Add roles there if the course requires them.
