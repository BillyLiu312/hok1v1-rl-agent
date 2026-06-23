# Poster

This folder contains a LaTeX academic poster based on `poster模板.pptx`.

## Files

- `poster.tex`: main LaTeX source.
- `figures/poster_template_background.jpg`: background extracted from the PPTX template.
- `poster模板.pptx`: original reference template.

## Compile

Preferred compile command:

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

The current workspace did not have a TeX engine installed, so PDF compilation was not performed here.
