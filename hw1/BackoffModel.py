import math, collections
from collections import defaultdict

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = defaultdict(lambda: 0)
    self.table = defaultdict(lambda: defaultdict(int))
    self.words = set([])
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
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
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    prevWord = None
    vocab = len(self.words)
    for token in sentence:
      occurances = self.table[prevWord][token]
      countPrev = self.unigramCounts[prevWord]

      probability = float(occurances) / (float(countPrev) + vocab)

      #Test results of bigram
      if probability > 0:
        score += math.log(probability)
      else: #Back off to unigram
        count = self.unigramCounts[token]
        if count > 0:
          score += math.log(count)
          score -= math.log(self.total)

      prevWord = token
    return abs(score)
