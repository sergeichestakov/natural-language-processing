import math, collections
from collections import defaultdict

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = defaultdict(lambda: 1)
    self.table = defaultdict(lambda: defaultdict(int))
    self.words = set([])
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
        self.words.add(token)
        prevWord = token


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    prevWord = None
    for token in sentence:
      occurances = self.table[prevWord][token]
      countPrev = self.unigramCounts[prevWord]

      probability = float(occurances) / (float(countPrev) / len(self.words))

      if probability > 0:
        score += math.log(probability)

      prevWord = token
    return abs(score)
