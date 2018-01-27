import math, collections
from collections import defaultdict

#This class implements Weighted interpolation by using a Bigram and Unigram model in conjunction
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this model"""
    self.unigramCounts = defaultdict(int)
    self.table = defaultdict(lambda: defaultdict(int))
    self.words = set([])
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """
    for sentence in corpus.corpus:
      prevWord = None
      for datum in sentence.data:
        token = datum.word
        self.table[prevWord][token] = self.table[prevWord][token] + 1
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.total += 1
        self.words.add(token)
        prevWord = token

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    score = 0.0
    prevWord = None
    vocab = len(self.words)
    for token in sentence:
      occurances = self.table[prevWord][token]
      countPrev = self.unigramCounts[prevWord]

      #Calculate bigram probability
      probability = float(occurances) / (float(countPrev) + vocab)

      #Give more weight to Bigram depending on how common it is
      interpolation = float(occurances) / float(occurances + 1)

      #Check for bigram first
      if probability > 0:
        score += math.log(interpolation * probability)

      #Add unigram to score
      count = self.unigramCounts[token]
      if count > 0:
        score += math.log((1 - interpolation) * count)
        score -= math.log(self.total)

      prevWord = token
    return abs(score)
