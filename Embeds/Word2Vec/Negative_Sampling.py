from .component import *
import numpy as np


class Embedding:
    def __init__(self, corpus, frequencies, embedding_dim, windows, alpha, epoch, k, ratio, alpha_min):
        """
        :param corpus: corpus
        :param frequencies: word and frequence
        :param embedding_dim: word embedding dimension
        :param windows: count of context
        :param alpha: learning rate
        :param epoch: iteration
        :param k: count of negative samples
        :param ratio: power of word's frequence
        :param alpha_min: minimum learning rate
        """
        self.corpus = corpus
        self.frequencies = frequencies
        self.embedding_dim = embedding_dim
        self.windows = windows
        self.alpha = alpha
        self.epoch = epoch
        self.k = k
        self.ratio = ratio
        self.alpha_min = alpha_min

    def find_context(self, sentence, cursor):
        """
        Find out the context words
        :param sentence: a line of corpus
        :param cursor: index of central word
        :return: context
        """
        length = len(sentence)

        # process start of sentence
        if cursor >= self.windows:
            left_distance = self.windows
        else:
            left_distance = cursor

        # process end of sentence
        if length - cursor >= self.windows:
            right_distance = self.windows
        else:
            right_distance = length - cursor

        words = sentence[cursor - left_distance: cursor] + sentence[
                                                           cursor + 1: cursor + right_distance + 1]
        return words

    def make_table(self):
        """
        Generate a random sampling table
        """
        from math import pow
        self.table = []
        for word, times in self.frequencies.items():
            idx = self.word_to_id[word]
            times = int(pow(times, self.ratio))
            temp = [idx] * times
            self.table.extend(temp)

    def initialize(self):
        """
        Initialize embedding table of words、weights
        """
        # self.embeds = (np.random.rand(self.embedding_dim, len(self.frequencies))- 0.5)/self.embedding_dim
        self.embeds = np.random.randn(self.embedding_dim, len(self.frequencies)) / 100
        self.word_to_id = {word: index for index, word in enumerate(self.frequencies.keys())}
        self.word_weights = np.zeros((len(self.frequencies), self.embedding_dim))

        self.make_table()

    def sum_embeds(self, sentence, cursor):
        """
        Sum context embedding
        """
        words = self.find_context(sentence, cursor)
        ids = [self.word_to_id[word] for word in words]
        total = 0
        for idx in ids:
            total += self.embeds[:, idx]
        return total, ids

    def sampling(self, positive_id, k):
        """
        return sampling IDS(include the right id)
        :param right_id: positive sample
        :param k: N negative samples
        :return: samples IDs
        """
        import random
        wrong_ids = []
        while len(wrong_ids) < k:
            idx = self.table[random.randint(0, len(self.table) - 1)]
            if idx == positive_id:
                continue
            else:
                wrong_ids.append(idx)
        wrong_ids.append(positive_id)  # including positive id and negative ids
        return wrong_ids

    def fit_skip(self):
        """
        Train corpus using Skip Gram + Negative Sampling
        :return:
        """
        self.initialize()

        for epoch in range(self.epoch):
            temp = int(len(self.corpus) / 10)  # to calculate perception
            for num, sentence in enumerate(self.corpus):
                # rate of processing and modify learning rate
                if num % temp == 0:
                    print("Epoch: %d , finishing ---> %d %%" % (epoch, num // temp * 10))
                    self.alpha = dynamic_alpha(self.alpha, num, len(self.corpus), self.alpha_min)
                    print("Learning rate updates at %.7f" %self.alpha)
                    print("------------------------------------------")

                for i in range(len(sentence)):
                    context_words = self.find_context(sentence, i)

                    positive_ids = [self.word_to_id[word] for word in
                                    context_words]  # find out ids to context as the positive IDs
                    central_word_id = self.word_to_id[sentence[i]]
                    all_ids_plus_context = []  # put positive ids(context) into negative ids list one by one
                    for right_id in positive_ids:
                        all_ids = self.sampling(right_id, self.k)
                        all_ids_plus_context.append(all_ids)

                    central_word_embedding = self.embeds[:, central_word_id]

                    for m in range(len(context_words)):
                        e = 0
                        for u in all_ids_plus_context[m]:
                            q = sigmoid(np.dot(self.word_weights[u], central_word_embedding))
                            if u == positive_ids[m]:
                                l_w = 1
                            else:
                                l_w = 0
                            g = self.alpha * (l_w - q)
                            e = e + g * self.word_weights[u]
                            self.word_weights[u] = self.word_weights[u] + (g * central_word_embedding).reshape(1, self.embedding_dim)

                        central_word_embedding = central_word_embedding + e
        print("Skip-Gram NCE train has finished ...")

    def fit_cbow(self):
        """
        Train corpus by using CBOW + Negative Sampling
        """
        self.initialize()
        for epoch in range(self.epoch):
            temp = int(len(self.corpus) / 10)  # to calculate perception
            for num, sentence in enumerate(self.corpus):
                # rate of processing and modify learning rate
                if num % temp == 0:
                    print("Epoch: %d , finishing ---> %d %%" % (epoch, num // temp * 10))
                    self.alpha = dynamic_alpha(self.alpha, num, len(self.corpus), self.alpha_min)
                    print("Learning rate updates at %.7f" %self.alpha)
                    print("------------------------------------------")

                for i in range(len(sentence)):
                    if len(sentence) < 2:
                        print(sentence, num)
                    X_w, words_ids = self.sum_embeds(sentence, i)
                    right_id = self.word_to_id[sentence[i]]
                    all_ids = self.sampling(right_id, self.k)

                    e = 0
                    for index in all_ids:
                        q = sigmoid(np.dot(self.word_weights[index], X_w))
                        if index == right_id:
                            l_w = 1
                        else:
                            l_w = 0
                        g = self.alpha * (l_w - q)
                        e = e + g * self.word_weights[index]
                        self.word_weights[index] = self.word_weights[index] + (g * X_w).reshape(1, self.embedding_dim)

                    for j in words_ids:
                        self.embeds[:, j] = self.embeds[:, j] + e.reshape(1, self.embedding_dim)
        print("CBOW NCE train has finished ...")

    def similar(self, word1, word2):
        """
        Using words to calculate similarity
        """
        embeds1 = self.embeds[:, self.word_to_id[word1]]
        embeds2 = self.embeds[:, self.word_to_id[word2]]
        return np.dot(embeds1, embeds2) / (np.linalg.norm(embeds1) * np.linalg.norm(embeds2))

    def most_similar(self, word, n):
        """
        Store the ids of the n most similar words
        """
        idx = self.word_to_id[word]
        word_embedding = self.embeds[:, idx]

        max_list = {}  # store the ids of the n most similar words
        for i in range(len(self.word_to_id)):
            if idx == i:  # self
                continue
            else:
                other_embedding = self.embeds[:, i]
                similarity = np.dot(word_embedding, other_embedding) / (
                            np.linalg.norm(word_embedding) * np.linalg.norm(other_embedding))

                if len(max_list) < n:
                    max_list[i] = similarity
                else:
                    max_list[i] = similarity
                    min_key, _ = min(max_list.items(), key=lambda x: x[1])
                    del (max_list[min_key])
        # find out words to ids
        result_words = []
        for word, index in self.word_to_id.items():
            if index in max_list.keys():
                result_words.append((word, max_list[index]))
        return sorted(result_words, key=lambda x: x[1], reverse=True)
