from . import Negative_Sampling
from . import Hierarchecal_Software
import logging
logging.basicConfig(level=logging.INFO)

class Embedding:
    def __init__(self, NSG=True, CBOW=True, embedding_dim=100, windows=3, alpha=0.025, epoch=5, k=5, ratio=3/4, alpha_min=0.001):
        """
        :param NCE: True(Negative Sampling), False(Hierarchecal Softmax)
        :param CBOW: True(Continuous Bag of Word), False(Skip Gram)
        :param embedding_dim: word embedding dimension
        :param windows: windows of context
        :param alpha: learning rate
        :param epoch: iteration
        :param k: count of negative samples
        :param ratio: power of word's frequence
        """
        self.NSG = NSG
        self.CBOW = CBOW
        self.embedding_dim = embedding_dim
        self.windows = windows
        self.alpha = alpha
        self.epoch = epoch
        self.k = k
        self.ratio = ratio
        self.alpha_min = alpha_min
        self.corpus = []

    def preprocessing(self, corpus_path):
        """
         Main purpose is to remove stop words
        :param corpus_path: path of corpus
        :return:
        """
        # load stop word table
        stop_words_path = "../dict/stop_words.txt"
        stop_word = []
        if stop_words_path is not None:
            try:
                with open(stop_words_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        stop_word.append(line.strip())
            except IOError:
                logging.info("Opening stop words file failed...")

        # remove stop word from corpus
        try:
            with open(corpus_path, "r", encoding="utf-8") as w:
                line = w.readline()
                while line:
                    temp_line = []
                    line = line.strip().split()
                    if len(line) < 2:
                        line = w.readline()
                        continue
                    for word in line:
                        if word not in stop_word:
                            temp_line.append(word)
                    self.corpus.append(temp_line)
                    line = w.readline()
        except IOError:
            raise IOError(" Opening corpus file failed...")

    def freq(self, n=1):
        """
        statistics frequencies of words
        """
        from collections import defaultdict
        dictionary = defaultdict(int)

        # frequencies of words
        for line in self.corpus:
            for word in line:
                dictionary[word] += 1

        # remove low frequency words from corpus and dictionary
        temp_corpus = []
        if n >= 1:
            low_time_words = set()
            for word, time in dictionary.items():
                if time <= n:
                    low_time_words.add(word)
            for index in range(len(self.corpus)):
                line = [word for word in self.corpus[index] if word not in low_time_words]
                if len(line) >= 2:
                    temp_corpus.append(line)
            self.corpus = temp_corpus

            for word in low_time_words:
                del (dictionary[word])

        if self.NSG:
            self.frequencies = dictionary
        else:
            self.frequencies = [(value, word) for word, value in dictionary.items()]

    def fit(self, corpus_path):
        """
        train corpus
        :param corpus_path: path of corpus
        :return:
        """
        self.preprocessing(corpus_path)
        self.freq()
        if self.NSG:
            self.model = Negative_Sampling.Embedding(self.corpus, self.frequencies, self.embedding_dim, self.windows,
                                                     self.alpha, self.epoch, self.k, self.ratio,
                                                     self.alpha_min)
        else:
            self.model = Hierarchecal_Software.Embedding(self.corpus, self.frequencies, self.embedding_dim, self.windows,
                                                         self.alpha, self.epoch, self.alpha_min)
        if self.CBOW:
            self.model.fit_cbow()
        else:
            self.model.fit_skip()

    def similar(self, word1, word2):
        return self.model.similar(word1, word2)

    def most_similar(self, word, n=10):
        return self.model.most_similar(word, n)

if __name__ == "__main__":
    pass