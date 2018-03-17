import sys
import getopt
import os
import operator
from math import log
from collections import defaultdict

class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.numFolds = 10

        #Custom data structures for training and clasifying
        self.posDocCount = 0.0
        self.totalDocCount = 0.0
        self.posFrequency = defaultdict(lambda: 2.0) #Laplace smoothing
        self.negFrequency = defaultdict(lambda: 2.0)
        self.posWordCount = 0.0
        self.negWordCount = 0.0
        self.posWordSet = set()
        self.negWordSet = set()

        #Data structures for highly optimized model
        self.posBigram = defaultdict(lambda: defaultdict(lambda: 1.0)) #Bigram counts and interpolation
        self.negBigram = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.interpolation = 0.05

        self.negateWords = ["not", "didn't", "isn't", "no", "never", "didnt", "isnt"] #Negation in best model
        self.punctuation = ['.', ',', '!', '?', '-']
        self.bonusWords = ['very', 'really', 'exceptionally', 'extremely', 'hugely', 'truly'] #Bonus words give more weight to the following word

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        if self.stopWordsFilter:
            words = self.filterStopWords(words)

        #Probability that a document is positive or negative based on ratio of docs in training set
        probPos = -log(self.posDocCount / self.totalDocCount)
        probNeg = -log( (self.totalDocCount - self.posDocCount) / self.totalDocCount)

        vocab = len(self.posWordSet.union(self.negWordSet))
        if self.naiveBayesBool:
            vocab = self.posWordCount + self.negWordCount
        elif self.bestModel:
            vocab = len(self.posWordSet) + len(self.negWordSet)
        negateFlag = False
        bonus = False
        prevWord = '__START__' #Start of sentence in bigram

        for word in words:
            pos = self.posFrequency[word]
            neg = self.negFrequency[word]

            if self.bestModel: #Negate in best model
                if word in self.negateWords:
                    negateFlag = True
                if negateFlag and word in self.punctuation:
                    negateFlag = False
                if negateFlag:
                    word = 'NOT_' + word

                #Implement weighted interpolation for bigram
                pos = (self.interpolation * self.posFrequency[word] + (1 - self.interpolation) * self.posBigram[prevWord][word])
                neg = (self.interpolation * self.negFrequency[word] + (1 - self.interpolation) * self.negBigram[prevWord][word])

            #Calculate and add probability of each word given the category
            probPosWord = -log( pos / (self.posWordCount + 2 * vocab) )
            probNegWord = -log( neg / (self.negWordCount + 2 * vocab) )

            if self.bestModel: #Increase weight of any word immediately following very, really, etc
                if bonus:
                    if probPosWord < probNegWord:
                        probPosWord *= 0.9
                    else:
                        probNegWord *= 0.9
                    bonus = False
                if word in self.bonusWords:
                    bonus = True

            probPos += probPosWord
            probNeg += probNegWord

            prevWord = word

        #Smallest is most likely because we are adding -log
        if probPos < probNeg:
            return 'pos'
        else:
            return 'neg'

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        if classifier == 'pos':
            self.posDocCount += 1

        self.totalDocCount += 1
        docSet = set() #Avoid duplicates in binary naive bayes
        negateFlag = False #Negate in best model
        prevWord = '__START__' #Start of sentence in bigram model

        for word in words:
            if self.bestModel: #Implement negation in best model
                if word in self.negateWords:
                    negateFlag = True
                if negateFlag and word in self.punctuation:
                    negateFlag = False
                if negateFlag:
                    word = 'NOT_' + word

            if classifier == 'pos':
                if self.naiveBayesBool:
                    if word not in docSet: #Check for duplicates
                        self.posWordCount += 1
                        docSet.add(word)
                        self.posFrequency[word] += 1
                else:
                    self.posWordCount += 1
                    self.posFrequency[word] += 1
                    if self.bestModel: #Keep track of bigram counts
                        self.posBigram[prevWord][word] += 1
                self.posWordSet.add(word)
            else: #classifier is 'neg'
                if self.naiveBayesBool:
                    if word not in docSet:
                        self.negWordCount +=1
                        docSet.add(word)
                        self.negFrequency[word] += 1
                else:
                    self.negWordCount += 1
                    self.negFrequency[word] += 1
                    if self.bestModel:
                        self.negBigram[prevWord][word] += 1
                self.negWordSet.add(word)

            prevWord = word


    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
