# Report diagrams

Mermaid sources for the report's design chapter. Every module, class and function named in them is checked against
the code by `fedverify/tests/test_diagrams.py`, so a rename breaks the test rather than leaving a stale diagram.

| File | Diagram | Where in the report |
|---|---|---|
| `architecture.mmd` | System architecture: hospitals, aggregator, ledger, results, dashboard | Design: architecture |
| `dfd_level0.mmd` | Data-flow diagram, level 0 (context) | Design: DFD level 0 (mandatory) |
| `dfd_level1.mmd` | Data-flow diagram, level 1: six processes and six data stores | Design: DFD level 1 (mandatory) |
| `use_case.mmd` | Use cases of hospitals, aggregator, auditor, researcher, adversary | Design: use-case diagram (mandatory) |
| `class_diagram.mmd` | Classes of the core modules and how they relate | Design: class diagram |
| `sequence_round.mmd` | One federated round, from the previous-block check to the new model | Design: sequence diagram |

## Rendering

Each `.mmd` has its renders next to it: `<name>.svg` and `<name>.png` (300 dpi at the diagram's natural size, where
its labels print at 12 pt). `rendered.json` records the SHA-256 of every source and render and the mermaid-cli and
browser versions; `fedverify/tests/test_phase12_wording.py` fails if a render no longer matches its source, and the
report pack (`make_report_pack`) refuses to copy a stale one. To re-render after editing a diagram (npx fetches
mermaid-cli the first time, so that needs network once; it uses the installed Chrome or Chromium, no download):

```bash
python3 -m fedverify.analysis.render_diagrams          # all six: SVG, then PNG at 300 dpi; rewrites rendered.json
```

In a Mermaid sequence diagram a `;` inside a message ends the line (it broke `sequence_round.mmd` once), so message
text uses commas instead. `architecture.mmd` is a wide left-to-right diagram (about 6:1): at the text width of a
portrait page its labels are tiny, so place it landscape or use the SVG. GitHub also renders Mermaid inside Markdown
code blocks.
