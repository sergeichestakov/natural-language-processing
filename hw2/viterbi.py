# Noah A. Smith
# 2/21/08
# Runs the Viterbi algorithm (no tricks other than logmath!), given an
# HMM, on sentences, and outputs the best state path.

# Sergei Chestakov
# 2/11/18
# Trigram implementation of Viterbi algorithm ported to Python
# Usage: ./viterbi.py my.hmm ptb.22.txt > my.out

import sys
import re
import math
import itertools
from pprint import pprint
from collections import defaultdict

INIT_STATE = 'init'
FINAL_STATE = 'final'
OOV_SYMBOL = 'OOV'

hmmfile=sys.argv[1]
inputfile=sys.argv[2]

tags = set() # i.e. K in the slides, a set of unique POS tags
trans = {} # transisions
emit = {} # emissions
voc = {} # encountered words

"""
This part parses the my.hmm file you have generated and obtain the transition and emission values.
"""
with open(hmmfile) as hmmfile:
    for line in hmmfile.read().splitlines():
        trans_reg = 'trans\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'
        emit_reg = 'emit\s+(\S+)\s+(\S+)\s+(\S+)'
        trans_match = re.match(trans_reg, line)
        emit_match = re.match(emit_reg, line)
        if trans_match:
            prevprev, prev, curr, probability = trans_match.groups()
            # creating an entry in trans with the POS tag pair
            # e.g. (init, init, NNP) = log(probability for seeing that transition)
            trans[(prevprev, prev, curr)] = math.log(float(probability))
            # add the encountered POS tags to set
            tags.update([prevprev, prev, curr])
        elif emit_match:
            tag, word, probability = emit_match.groups()
            # creating an entry in emit with the tag and word pair
            # e.g. (NNP, "python") = log(probability for seeing that word with that tag)
            emit[(tag, word)] = math.log(float(probability))
            # adding the word to encountered words
            voc[word] = 1
            # add the encountered POS tags to set
            tags.update([tag])
        else:
            pass

"""
This part parses the file with the test sentences, and runs the sentence through the viterbi algorithm.
"""
with open(inputfile) as inputfile:
    for line in inputfile.read().splitlines():
        line = line.split(' ')
        # initialize pi.
        # i.e. set pi(0, *, *) = 1 from slides
        # Format is (k, u, v) = (index, prevtag, currtag)
        pi = {(0, INIT_STATE, INIT_STATE): 0.0} # 0.0 because using logs
        bp = {} # backpointers

        # for each word in sentence and their index
        for index, word in enumerate(line):
            index = index + 1
            if word not in voc:
                # change unseen words into OOV, since OOV is assigned a score in train_hmm. This will give these unseen words a score instead of a mismatch.
                word = OOV_SYMBOL
            for prevprev, prev, curr in itertools.product(tags, tags, tags): #python nested for loop
                # i.e. the first bullet point from the slides.
                # Calculate the scores (p) for each possible combinations of (u, v)
                if (prevprev, prev, curr) in trans and (curr, word) in emit and (index - 1, prevprev, prev) in pi:
                    probability = pi[(index - 1, prevprev, prev)] + trans[(prevprev, prev, curr)] + emit[(curr, word)]
                    if (index, prev, curr) not in pi or probability > pi[(index, prev, curr)]:
                        # here, find the max of all the calculated p, update it in the pi dictionary
                        pi[(index, prev, curr)] = probability
                        # also keeping track of the backpointer
                        bp[(index, prev, curr)] = prevprev

        # second bullet point from the slides. Taking the case for the last word. Find the corrsponding POS tag for that word so we can then start the backtracing.
        foundgoal = False
        goal = float('-inf')
        tag = INIT_STATE
        before_tag = INIT_STATE

        for prev, curr in itertools.product(tags, tags):
            # You want to try each (prevtag, tag, FINAL_STATE) triple for the last word and find which one has max p. That will be the tag you choose.
            if (prev, curr, FINAL_STATE) in trans and (len(line), prev, curr) in pi:
                probability = pi[(len(line), prev, curr)] + trans[(prev, curr, FINAL_STATE)]
                if not foundgoal or probability > goal:
                    # finding tag with max p
                    goal = probability
                    foundgoal = True
                    tag = curr
                    before_tag = prev


        if foundgoal:
            final_tags = [before_tag]
            for index in xrange(len(line) - 2, 0, -1): #start, stop, and step
                # bp[(index,prevtag, tag)] gives you the tag for word[index - 2].
                # we use that and traces through the tags in the sentence.
                final_tags.append(bp[(index + 2, before_tag, tag)])
                before = before_tag
                before_tag = bp[(index + 2, before_tag, tag)]
                tag = before

            # final_tags is appened last tag first. Reverse it.
            final_tags.reverse()
            # print the final output
            print ' '.join(final_tags)
        else:
            # append blank line if something fails so that each sentence is still printed on the correct line.
            print ''
