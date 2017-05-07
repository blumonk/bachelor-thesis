#!/bin/bash

pdflatex main
biber main
pdflatex main
pdflatex main

rm -f main.{bib,aux,log,bbl,bcf,blg,run.xml,toc,tct}
