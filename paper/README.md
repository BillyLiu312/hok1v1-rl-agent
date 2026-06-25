# Paper and Poster

This folder contains the project minipaper, academic poster, and their LaTeX sources.

## Files

- `Honor-of-King-Paper/minipaper.pdf`: compiled project minipaper.
- `Honor-of-King-Paper/minipaper.tex`: minipaper LaTeX source based on the NeurIPS template.
- `Honor-of-King-Paper/neurips_2026.tex`: original NeurIPS template shell.
- `poster.pdf`: compiled project poster.
- `poster.tex`: poster LaTeX source.
- `figures/poster_template_background.jpg`: poster background extracted from the PPTX template.
- `poster模板.pptx`: original poster reference template.
- `Honor-of-King-Poster/`: archived poster source/output copy.

## Compile

Compile the minipaper with Tectonic:

```bash
cd paper/Honor-of-King-Paper
tectonic -X compile minipaper.tex
```

If you use a local TeX Live / MacTeX installation, `xelatex` also works for the minipaper:

```bash
cd paper/Honor-of-King-Paper
xelatex minipaper.tex
```

Preferred poster compile command:

```bash
cd paper
xelatex poster.tex
```

The source now also supports pdfLaTeX. If you only have `pdflatex`, use:

```bash
cd paper
pdflatex poster.tex
```

If `Helvetica Neue` is unavailable under XeLaTeX, replace it in `poster.tex` with another installed sans-serif font, such as `Arial` or `Noto Sans`.
