#!/usr/bin/env bash
# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey);
#                      Arnab Ghoshal
#                2014  Guoguo Chen
#                2015  Hainan Xu
#                2016  FAU Erlangen (Author: Axel Horndasch)
#                2020  Xiaomi Corporation (Author: Junbo Zhang)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script prepares a directory such as data/lang/, in the standard format,
# given a source directory containing a dictionary lexicon.txt in a form like:
# word phone1 phone2 ... phoneN
# per line (alternate prons would be separate lines), or a dictionary with probabilities
# called lexiconp.txt in a form:
# word pron-prob phone1 phone2 ... phoneN
# (with 0.0 < pron-prob <= 1.0); note: if lexiconp.txt exists, we use it even if
# lexicon.txt exists.
# and also files silence_phones.txt, nonsilence_phones.txt, optional_silence.txt
# and extra_questions.txt
# Here, silence_phones.txt and nonsilence_phones.txt are lists of silence and
# non-silence phones respectively (where silence includes various kinds of
# noise, laugh, cough, filled pauses etc., and nonsilence phones includes the
# "real" phones.)
# In each line of those files is a list of phones, and the phones on each line
# are assumed to correspond to the same "base phone", i.e. they will be
# different stress or tone variations of the same basic phone.
# The file "optional_silence.txt" contains just a single phone (typically SIL)
# which is used for optional silence in the lexicon.
# extra_questions.txt might be empty; typically will consist of lists of phones,
# all members of each list with the same stress or tone; and also possibly a
# list for the silence phones.  This will augment the automatically generated
# questions (note: the automatically generated ones will treat all the
# stress/tone versions of a phone the same, so will not "get to ask" about
# stress or tone).
#

# This script adds word-position-dependent phones and constructs a host of other
# derived files, that go in data/lang/.

# Begin configuration section.


sil_prob=0.5
num_extra_phone_disambig_syms=1 # Standard one phone disambiguation symbol is used for optional silence.
                                # Increasing this number does not harm, but is only useful if you later
                                # want to introduce this labels to L_disambig.fst
# end configuration sections

echo "$0 $@"  # Print the command line for logging

. local/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: local/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>"
  echo "e.g.: local/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang"
  echo "<dict-src-dir> should contain the following files:"
  echo " extra_questions.txt  lexicon.txt nonsilence_phones.txt  optional_silence.txt  silence_phones.txt"
  echo "See http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating for more info."
  echo "options: "
  echo "<dict-src-dir> may also, for the grammar-decoding case (see http://kaldi-asr.org/doc/grammar.html)"
  echo "contain a file nonterminals.txt containing symbols like #nonterm:contact_list, one per line."
  echo "     --num-sil-states <number of states>             # default: 5, #states in silence models."
  echo "     --num-nonsil-states <number of states>          # default: 3, #states in non-silence models."
  echo "     --position-dependent-phones (true|false)        # default: true; if true, use _B, _E, _S & _I"
  echo "                                                     # markers on phones to indicate word-internal positions. "
  echo "     --sil-prob <probability of silence>             # default: 0.5 [must have 0 <= silprob < 1]"
  exit 1;
fi

srcdir=$1
oov_word=$2
tmpdir=$3
dir=$4

if [ -d $dir/phones ]; then
  rm -r $dir/phones
fi
mkdir -p $dir $tmpdir $dir/phones

[ -f path.sh ] && . ./path.sh

if [[ ! -f $srcdir/lexicon.txt ]]; then
  echo "**Creating $srcdir/lexicon.txt from $srcdir/lexiconp.txt"
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' < $srcdir/lexiconp.txt > $srcdir/lexicon.txt || exit 1;
fi
if [[ ! -f $srcdir/lexiconp.txt ]]; then
  echo "**Creating $srcdir/lexiconp.txt from $srcdir/lexicon.txt"
  perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $srcdir/lexiconp.txt || exit 1;
fi

cp $srcdir/lexiconp.txt $tmpdir/lexiconp.txt

cat $srcdir/silence_phones.txt $srcdir/nonsilence_phones.txt | \
  awk '{for(n=1;n<=NF;n++) print $n; }' > $tmpdir/phones
paste -d' ' $tmpdir/phones $tmpdir/phones > $tmpdir/phone_map.txt

# Making monophone systems.
cat $srcdir/silence_phones.txt | local/apply_map.pl $tmpdir/phone_map.txt | \
  awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/silence.txt
cat $srcdir/nonsilence_phones.txt | local/apply_map.pl $tmpdir/phone_map.txt | \
  awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/nonsilence.txt
cp $srcdir/optional_silence.txt $dir/phones/optional_silence.txt

# if extra_questions.txt is empty, it's OK.
cat $srcdir/extra_questions.txt 2>/dev/null | local/apply_map.pl $tmpdir/phone_map.txt \
  >$dir/phones/extra_questions.txt


ndisambig=$(local/add_lex_disambig.pl $unk_opt --pron-probs $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt)

ndisambig=$[$ndisambig+$num_extra_phone_disambig_syms]; # add (at least) one disambig symbol for silence in lexicon FST.
echo $ndisambig > $tmpdir/lex_ndisambig

# Format of lexiconp_disambig.txt:
# !SIL	1.0   SIL_S
# <SPOKEN_NOISE>	1.0   SPN_S #1
# <UNK>	1.0  SPN_S #2
# <NOISE>	1.0  NSN_S
# !EXCLAMATION-POINT	1.0  EH2_B K_I S_I K_I L_I AH0_I M_I EY1_I SH_I AH0_I N_I P_I OY2_I N_I T_E

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$dir/phones/disambig.txt

# Create phone symbol table.
echo "<eps>" | cat - $dir/phones/{silence,nonsilence,disambig}.txt | \
  awk '{n=NR-1; print $1, n;}' > $dir/phones.txt

# Create a file that describes the word-boundary information for
# each phone.  5 categories.

# word_boundary.txt might have been generated by another source
[ -f $srcdir/word_boundary.txt ] && cp $srcdir/word_boundary.txt $dir/phones/word_boundary.txt

# Create word symbol table.
# <s> and </s> are only needed due to the need to rescore lattices with
# ConstArpaLm format language model. They do not normally appear in G.fst or
# L.fst.

cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $dir/words.txt || exit 1;

# format of $dir/words.txt:
#<eps> 0
#a 1
#aa 2
#aarvark 3
#...

silphone=`cat $srcdir/optional_silence.txt` || exit 1;
[ -z "$silphone" ] && \
  ( echo "You have no optional-silence phone; it is required in the current scripts"
    echo "but you may use the option --sil-prob 0.0 to stop it being used." ) && \
   exit 1;

grammar_opts=

# Create the basic L.fst without disambiguation symbols, for use
# in training.

local/make_lexicon_fst.py $grammar_opts --sil-prob=$sil_prob --sil-phone=$silphone \
  $tmpdir/lexiconp.txt | \
  local/sym2int.pl -f 3 data/lang_char_nosp/phones.txt | \
  local/sym2int.pl -f 4 data/lang_char_nosp/words.txt > $dir/L.fst.txt || exit 1;

# fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
#   --keep_isymbols=false --keep_osymbols=false | \
# fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "$oov_word" > $dir/oov.txt || exit 1;
cat $dir/oov.txt | local/sym2int.pl $dir/words.txt >$dir/oov.int || exit 1;
# integer version of oov symbol, used in some scripts.

# the file wdisambig.txt contains a (line-by-line) list of the text-form of the
# disambiguation symbols that are used in the grammar and passed through by the
# lexicon.  At this stage it's hardcoded as '#0', but we're laying the groundwork
# for more generality (which probably would be added by another script).
# wdisambig_words.int contains the corresponding list interpreted by the
# symbol table words.txt, and wdisambig_phones.int contains the corresponding
# list interpreted by the symbol table phones.txt.
echo '#0' >$dir/phones/wdisambig.txt

wdisambig_phone=`local/sym2int.pl $dir/phones.txt <$dir/phones/wdisambig.txt`
wdisambig_word=`local/sym2int.pl $dir/words.txt <$dir/phones/wdisambig.txt`

# Create these lists of phones in colon-separated integer list form too,
# for purposes of being given to programs as command-line options.
for f in silence nonsilence optional_silence disambig; do
  local/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt >$dir/phones/$f.int
  local/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/$f.csl || exit 1;
done

if [ -f $dir/phones/word_boundary.txt ]; then
  local/sym2int.pl -f 1 $dir/phones.txt <$dir/phones/word_boundary.txt \
    > $dir/phones/word_boundary.int || exit 1;
fi

silphonelist=`cat $dir/phones/silence.csl`
nonsilphonelist=`cat $dir/phones/nonsilence.csl`

# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.

local/make_lexicon_fst.py $grammar_opts \
  --sil-prob=$sil_prob --sil-phone=$silphone --sil-disambig='#'$ndisambig \
  $tmpdir/lexiconp_disambig.txt | \
  local/sym2int.pl -f 3 data/lang_char_nosp/phones.txt | \
  local/sym2int.pl -f 4 data/lang_char_nosp/words.txt | \
  local/fstaddselfloops.pl $wdisambig_phone $wdisambig_word > $dir/L_disambig.fst.txt || exit 1;

exit 0;
