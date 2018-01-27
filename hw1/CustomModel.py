import math, collections
from collections import defaultdict

class CustomModel:

  DISCOUNT = 0.4
  INTERPOLATION = 0.1
  def __init__(self, corpus):
    """Initial custom language model and structures needed by this model"""
    self.unigramCounts = defaultdict(lambda: 1)
    self.table = defaultdict(lambda: defaultdict(lambda: 1))
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

      #bigram = float(occurances - CustomModel.DISCOUNT) / (float(countPrev) / vocab)

      unigramCount = self.unigramCounts[token]
      #unigram = float(count) / float(self.total)

      interpolation = float(occurances) / float(occurances + CustomModel.INTERPOLATION)
      
      score += math.log(interpolation * (occurances - CustomModel.DISCOUNT))
      score -= math.log(float(countPrev) / vocab)

      if unigramCount > 0:
        score += math.log((1 - interpolation) * unigramCount)
        score -= math.log(self.total)
      #score += CustomModel.INTERPOLATION * unigram

      prevWord = token
    return score
