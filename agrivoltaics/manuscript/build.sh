#!/bin/bash
# compile the supplement first (cross-references both ways), then the main text, twice each
cd "$(dirname "$0")"
for i in 1 2; do
  pdflatex -interaction=nonstopmode supplement.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null
done
bibtex main >/dev/null; bibtex supplement >/dev/null
for i in 1 2; do pdflatex -interaction=nonstopmode supplement.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; done
