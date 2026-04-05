# Paper LaTeX Sources — Beyond Scale

| Submission | Main file | Style |
|---|---|---|
| DMLR 2026 | `DMLR_2026_BeyondScale/00_dmlr_beyond_scale.tex` | `dmlr2e.sty` |
| TMLR (original, do not modify) | `tmlr_original/main.tex` | `tmlr.sty` |

## Overleaf-style workflow in Cursor

1. Install **MacTeX**: `brew install --cask mactex`
2. Install the **LaTeX Workshop** extension in Cursor
3. Open a `.tex` file — **save to compile**, `Cmd+Alt+V` to open PDF preview side-by-side

That's it. Edit → save → PDF updates automatically, just like Overleaf.

> **Tip:** If LaTeX Workshop doesn't find the root file, open the root `.tex` first or add `% !TEX root = main.tex` at the top of any sub-file.
