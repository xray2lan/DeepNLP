import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class Embedding:
    def __init__(self, embedding_dim=100, windows=3, alpha=0.025, epoch=5, n=1, use_distance=True):
        """
        Glove embedding
        :param embedding_dim: word embedding dimension
        :param windows: context words
        :param alpha: learning rate
        :param epoch: iteration
        :param n: lowest frequence
        :param use_distance: context distance
        """
        self.embedding_dim = embedding_dim
        self.windows = windows
        self.alpha = alpha
        self.epoch = epoch
        self.n = n
        self.use_distance = use_distance
        self.vocabulary = set()
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

    def freq(self, n=0):
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
        self.vocabulary = [word for word in dictionary.keys()]
        self.vob_length = len(dictionary)

    def co_occurrence(self):
        """
        Generate co_occurrence
        """
        print(self.vob_length)
        self.matrix = np.zeros((self.vob_length, self.vob_length))

        for line in self.corpus:
            for i in range(len(line)):
                context_ids, distances = self.find_context(line, i)
                for index, idx in enumerate(context_ids):
                    if self.use_distance:
                        self.matrix[self.word_to_id[line[i]], idx] += (1 - distances[index] / 10)  # 越接近中心词权重越大
                    else:
                        self.matrix[self.word_to_id[line[i]], idx] += 1

    def dynamic_alpha(self, i, alpha_min=0.001):
        """
        Modify learning rate dynamic
        """
        new_alpha = self.alpha * (1 - i / (self.vob_length * 10 + 1))
        if new_alpha < alpha_min:
            self.alpha = alpha_min
        else:
            self.alpha = new_alpha

    def find_context(self, sentence, cursor):
        """
        Find out the context words
        :param sentence: a line of corpus
        :param cursor: index of central word
        :return: context
        """
        length = len(sentence)
        distances = []  # distance of every word to center word in context
        # process start of sentence
        if cursor >= self.windows:
            left_distance = self.windows
        else:
            left_distance = cursor
        for i in range(left_distance - 1, -1, -1):
            distances.append(i)

        # process end of sentence
        if length - cursor >= self.windows:
            right_distance = self.windows
        else:
            right_distance = length - cursor
        for j in range(0, right_distance):
            distances.append(j)

        words = sentence[cursor - left_distance: cursor] + sentence[cursor + 1: cursor + right_distance + 1]
        ids = [self.word_to_id[word] for word in words]
        return ids, distances

    def initialization(self):
        """
        Initialize word embedding、weights、dictionary including IDs for words
        """

        self.U = (np.random.rand(self.vob_length, self.embedding_dim) - 0.5) / 100
        self.V = (np.random.rand(self.vob_length, self.embedding_dim) - 0.5) / 100
        self.b_U = (np.random.rand(self.vob_length) - 0.5) / 100
        self.b_V = (np.random.rand(self.vob_length) - 0.5) / 100
        self.word_to_id = {word: index for index, word in enumerate(self.vocabulary)}

    def f_i_j(self, i, j, xmax=100):
        """
        Guaranteed weight is not too big
        :param i: row in co_occurrence
        :param j: column in co_occurrence
        :param xmax:
        :return: weight
        """
        X_i_j = self.matrix[i, j]
        if X_i_j < xmax:
            return (X_i_j / xmax) ** (3 / 4)
        else:
            return 1

    def fit(self, corpus_path):
        """
        train corpus
        :param corpus_path:
        :return:
        """
        self.preprocessing(corpus_path)
        self.freq()
        self.initialization()
        self.co_occurrence()

        for epoch in range(self.epoch):
            for i in range(self.vob_length):
                temp = self.vob_length // 50
                # percent of process and modify learning rate
                if i % temp == 0:
                    print("Epoch: %d , finishing ---> %d %%" % (epoch, i // temp * 2))
                    self.dynamic_alpha(i)
                    print(self.alpha)

                for j in range(self.vob_length):
                    f = self.f_i_j(i, j)
                    if f != 0.:  # not calculate when f=0
                        if self.matrix[i, j] == 0:
                            mid = np.dot(self.U[i], self.V[j]) + self.b_U[i] + self.b_V[j]
                        else:
                            mid = np.dot(self.U[i], self.V[j]) + self.b_U[i] + self.b_V[j] - np.log(self.matrix[i, j])

                        g = f * mid
                        delta_u_i = g * self.V[j]
                        delta_v_j = g * self.U[i]
                        delta_b_i = g
                        delta_b_j = g

                        self.U[i] = self.U[i] - self.alpha * delta_u_i
                        self.V[j] = self.V[j] - self.alpha * delta_v_j
                        self.b_U[i] = self.b_U[i] - self.alpha * delta_b_i
                        self.b_V[j] = self.b_V[j] - self.alpha * delta_b_j

        self.U = self.U + self.V

    def similar(self, word1, word2):
        """
        Using words to calculate similarity
        :param word1: word1
        :param word2: word2
        :return: similarity
        """
        embeds1 = self.U[self.word_to_id[word1]]
        embeds2 = self.U[self.word_to_id[word2]]
        return np.dot(embeds1, embeds2) / (np.linalg.norm(embeds1) * np.linalg.norm(embeds2))

    def most_similar(self, word, n=10):
        """
        the Topn most similar words
        """
        idx = self.word_to_id[word]
        embedding = self.U[idx]

        max_list = {}
        for i in range(len(self.word_to_id)):
            if idx == i:
                continue
            else:
                other = self.U[i]
                similarity = np.dot(embedding, other) / (np.linalg.norm(embedding) * np.linalg.norm(other))

                if len(max_list) < n:
                    max_list[i] = similarity
                else:
                    max_list[i] = similarity
                    min_key, _ = min(max_list.items(), key=lambda x:x[1])
                    del(max_list[min_key])
        result_words = []
        for word, index in self.word_to_id.items():
            if index in max_list.keys():
                result_words.append((word, max_list[index]))
        return sorted(result_words, key=lambda x: x[1], reverse=True)