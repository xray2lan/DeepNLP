from .component import *
import numpy as np

class Embedding:
    def __init__(self, corpus, frequencies, embedding_dim, windows, alpha, epoch, alpha_min):
        """
        :param corpus: corpus
        :param frequencies: word and frequence
        :param embedding_dim: word embedding dimension
        :param windows: count of context
        :param alpha: learning rate
        :param epoch: iteration
        :param alpha_min: minimum learning rate
        """
        self.corpus = corpus
        self.frequencies = frequencies
        self.embedding_dim = embedding_dim
        self.windows = windows
        self.alpha = alpha
        self.epoch = epoch
        self.alpha_min = alpha_min

    def make_dict(self):
        """
        make assis vectors and dictionary
        """
        huff = Huffman()
        huff.create_tree(self.frequencies)
        huff.huff_encode(huff.phead)

        self.word_to_code = huff.encode_dictionary  # dictionary of word to Huffman coding
        self.word_to_id = {char: num for num, char in enumerate(self.word_to_code.keys())}  # dicrionary including ID to word
        self.code_to_word = {key: value for key, value in zip(self.word_to_code.values(), self.word_to_code.keys())}  # dictionary inclueding huffman coding to word
        self.assis = huff.assis_set  # set including non-leaf node path
        self.assis_to_id = {char: num for num, char in enumerate(["START"] + list(self.assis)[1:])}  # dictionary include ID to non-leaf node

    def initialize(self):
        """
        Initialize embedding table of words、weights
        """
        self.embeds = (np.random.rand(len(self.word_to_code), self.embedding_dim) - 0.5) / self.embedding_dim  # words vectories
        self.embeds_assis = np.zeros((len(self.assis), self.embedding_dim))  # assist vectories


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

        context_words = sentence[cursor - left_distance: cursor] + sentence[cursor + 1: cursor + right_distance + 1]

        return context_words

    def sum_embeds(self, sentence, cursor):
        """
        Sum context embedding
        """
        words = self.find_context(sentence, cursor)
        ids = [self.word_to_id[word] for word in words]

        X_w = np.zeros((1,100))
        for idx in ids:
            X_w += self.embeds[idx]
        return X_w.reshape(1, self.embedding_dim), ids

    def fit_skip(self):
        """
        Train corpus using CBOW + Skip Gram
        :return:
        """
        self.make_dict()
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
                    # get context words
                    context_words = self.find_context(sentence, i)

                    # get the huffman coding to target word
                    target_codes = []
                    for word in context_words:
                        target_codes.append(self.word_to_code[word])

                    # get non-leaf nodes table of IDs
                    context_words_path = []
                    a = ["START"]
                    for target_code in target_codes:
                        for k in range(1, len(target_code)):
                            a.append(target_code[0:k])
                        assis_ids = [self.assis_to_id[index] for index in a]
                        context_words_path.append(assis_ids)

                    e = 0
                    for u in range(len(context_words)):
                        for j in range(len(target_codes[u])):
                            central_word_embedding = self.embeds[self.word_to_id[sentence[i]]]
                            context_node_embedding = self.embeds_assis[context_words_path[u][j]].reshape(self.embedding_dim, 1)

                            q = sigmoid(np.dot(central_word_embedding, context_node_embedding))
                            g = self.alpha * (1 - int(target_codes[u][j]) - q)     # Think of 1 as a negative example, 0 positive
                            e = e + g * self.embeds_assis[context_words_path[u][j]]     # update weights
                            self.embeds_assis[context_words_path[u][j]] = self.embeds_assis[context_words_path[u][j]] + g * central_word_embedding

                        # update embedding vectories
                        self.embeds[self.word_to_id[sentence[i]]] = self.embeds[self.word_to_id[sentence[i]]] + e
        print("Skip-Gram HS train has finished ...")

    def fit_cbow(self):
        """
        Train corpus by using CBOW + Hierarchecal Softmax
        """
        self.make_dict()
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
                    # get sum of context embeddig, IDs to context words
                    X_w, words_ids = self.sum_embeds(sentence, i)

                    # get Huffman coding to target word
                    target_code = self.word_to_code[sentence[i]]

                    # get non-leaf nodes ids
                    a = ["START"]
                    for k in range(1, len(target_code)):
                        a.append(target_code[0:k])
                    assis_ids = [self.assis_to_id[index] for index in a]

                    e = 0
                    for j in range(len(target_code)):
                        q = sigmoid(np.dot(X_w, self.embeds_assis[assis_ids[j]].reshape(self.embedding_dim, 1)))
                        g = self.alpha * (1 - int(target_code[j]) - q)
                        e = e + g * self.embeds_assis[assis_ids[j]]     # update weights
                        self.embeds_assis[assis_ids[j]] = self.embeds_assis[assis_ids[j]] + g * X_w

                    # update embedding vectories
                    for i in words_ids:
                        self.embeds[i] = self.embeds[i] + e
        print("CBOW HS train has finished ...")

    def most_similar(self, word, n=3):
        """
        Store the ids of the n most similar words
        """
        idx = self.word_to_id[word]
        embedding = self.embeds[idx]

        max_list = {}
        for i in range(len(self.word_to_id)):
            if idx == i:
                continue
            else:
                # similarity
                other = self.embeds[i]
                similarity = np.dot(embedding, other) / (np.linalg.norm(embedding) * np.linalg.norm(other))

                # store the ids of the n most similar words
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

    def similar(self, word1, word2):
        """
        Using words to calculate similarity
        :param word1: word1
        :param word2: word2
        :return: similarity
        """
        embeds1 = self.embeds[self.word_to_id[word1]]
        embeds2 = self.embeds[self.word_to_id[word2]]
        return np.dot(embeds1, embeds2) / (np.linalg.norm(embeds1) * np.linalg.norm(embeds2))

