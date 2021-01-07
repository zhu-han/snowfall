#!/usr/bin/env bash

# Copyright 2014 Vassil Panayotov
# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Prepares the dictionary

stage=2

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lm-dir> <dst-dir>"
  echo "e.g.: data/lm data/local/dict"
  exit 1
fi

lm_dir=$1
dst_dir=$2

vocab=$lm_dir/librispeech-vocab.txt
[ ! -f $vocab ] && echo "$0: vocabulary file not found at $vocab" && exit 1;

# this file is a copy of the lexicon we download from openslr.org/11 
lexicon_raw_nosil=$dst_dir/lexicon_raw_nosil.txt
words_raw_nosil=$dst_dir/words_raw_nosil.txt
char_lexicon_raw_nosil=$dst_dir/char_lexicon_raw_nosil.txt

mkdir -p $dst_dir || exit 1;

# The copy operation below is necessary, if we skip the g2p stages(e.g. using --stage 3)
if [[ ! -s "$lexicon_raw_nosil" ]]; then
  cp $lm_dir/librispeech-lexicon.txt $lexicon_raw_nosil || exit 1
fi

if [ $stage -le 2 ]; then
  cat $lexicon_raw_nosil | awk -F " " '{print $1}' - | sort -u > $words_raw_nosil
  paste -d "\t" <(cat $words_raw_nosil) <(cat $words_raw_nosil|sed 's/./& /g') >$char_lexicon_raw_nosil
fi

if [ $stage -le 3 ]; then
  silence_phones=$dst_dir/silence_phones.txt
  optional_silence=$dst_dir/optional_silence.txt
  nonsil_phones=$dst_dir/nonsilence_phones.txt

  echo "Preparing phone lists"
  (echo "<space>"; echo "<SPOKEN_NOISE>";echo "<UNK>";) > $silence_phones
  echo "<space>" > $optional_silence
  # nonsilence phones; on each line is a list of phones that correspond
  # really to the same base phone.
  awk '{for (i=2; i<=NF; ++i) { print $i}}' $char_lexicon_raw_nosil |\
    sort -u > $nonsil_phones || exit 1;
fi

if [ $stage -le 4 ]; then
  (echo '<space> <space>'; echo '<SPOKEN_NOISE> <SPOKEN_NOISE>'; echo '<UNK> <UNK>'; ) |\
  cat - $char_lexicon_raw_nosil | sort | uniq >$dst_dir/lexicon.txt
  echo "Lexicon text file saved as: $dst_dir/lexicon.txt"
fi

exit 0