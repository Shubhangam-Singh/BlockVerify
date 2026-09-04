# Paper build

The paper cites **no hand-typed numbers**. Every quantity is a LaTeX macro defined in
`generated/numbers.tex`, emitted from results files by `analysis/make_paper.py`.

```bash
python3 -m fedverify.analysis.make_tables      # tables 1-6  -> results/tables/
python3 -m fedverify.analysis.plots            # figures 1-6 -> results/figures/
python3 -m fedverify.analysis.make_paper       # macros      -> paper/generated/numbers.tex
python3 -m fedverify.analysis.make_paper --check   # exits non-zero while any [??] remains
cd fedverify/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**A macro with no backing run renders as `[??]` in the PDF.** That is deliberate: an
experiment that has not been run cannot silently become a claim. `--check` is the gate —
do not submit while it fails.
