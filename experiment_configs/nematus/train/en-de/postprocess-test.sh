#!/usr/bin/env bash

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/projects/mosesdecoder/

# suffix of target language files
lng=de

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng
