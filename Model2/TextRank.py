import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import math

class TextRank:

    def __init__(self, stopwords_file_path):
        self.stopwords_from_file = []

        stopword_file = open(stopwords_file_path, "r")
        self.stopwords_from_file = []
        for line in stopword_file.readlines():
            self.stopwords_from_file.append(str(line.strip()))

        self.wordnet_lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        text = text.lower()
        printable = set(string.printable)
        text = filter(lambda x: x in printable, text)  # filter funny characters, if any.

        clean_text = "".join(list(text))

        return clean_text

    def generate_tokens(self, clean_text):
        tokens = word_tokenize(clean_text)
        return tokens

    def lemmatize(self, tokens):
        pos_tags = nltk.pos_tag(tokens)
        adjective_tags = ['JJ', 'JJR', 'JJS']

        lemmatized_text = []
        for word in pos_tags:
            if word[1] in adjective_tags:
                lemmatized_text.append(str(self.wordnet_lemmatizer.lemmatize(word[0], pos="a")))
            else:
                lemmatized_text.append(str(self.wordnet_lemmatizer.lemmatize(word[0])))  # default POS = noun

        return lemmatized_text

    def stopwords_based_on_pos_tags(self, lemmatized_text):
        pos_tags = nltk.pos_tag(lemmatized_text)

        stopwords = []
        wanted_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBG', 'FW']

        for word in pos_tags:
            if word[1] not in wanted_POS:
                stopwords.append(word[0])

        punctuations = list(str(string.punctuation))

        stopwords = stopwords + punctuations

        return stopwords

    def stopwords_based_filtering(self, lemmatized_text):
        stopwords_based_on_pos_tags = self.stopwords_based_on_pos_tags(lemmatized_text)

        final_stopwords = self.stopwords_from_file + stopwords_based_on_pos_tags

        processed_text = []
        for word in lemmatized_text:
            if word not in final_stopwords:
                processed_text.append(word)

        return processed_text

    def build_graph(self, vocabulary, processed_text, window_size, score):
        vocab_len = len(vocabulary)
        weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

        covered_coocurrences = []

        for i in range(0, vocab_len):
            score[i] = 1
            for j in range(0, vocab_len):
                if j == i:
                    weighted_edge[i][j] = 0
                else:
                    for window_start in range(0, (len(processed_text) - window_size)):

                        window_end = window_start + window_size

                        window = processed_text[window_start:window_end]

                        if (vocabulary[i] in window) and (vocabulary[j] in window):

                            index_of_i = window_start + window.index(vocabulary[i])
                            index_of_j = window_start + window.index(vocabulary[j])

                            # index_of_x is the absolute position of the xth term in the window
                            # (counting from 0)
                            # in the processed_text

                            if [index_of_i, index_of_j] not in covered_coocurrences:
                                weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                                covered_coocurrences.append([index_of_i, index_of_j])

        return weighted_edge, score

    def rank(self, vocab_len, weighted_edge, score):
        inout = np.zeros((vocab_len), dtype=np.float32)

        for i in range(0, vocab_len):
            for j in range(0, vocab_len):
                inout[i] += weighted_edge[i][j]

        MAX_ITERATIONS = 50
        d = 0.85
        threshold = 0.0001  # convergence threshold

        for iter in range(0, MAX_ITERATIONS):
            prev_score = np.copy(score)

            for i in range(0, vocab_len):

                summation = 0
                for j in range(0, vocab_len):
                    if weighted_edge[i][j] != 0:
                        summation += (weighted_edge[i][j] / inout[j]) * score[j]

                score[i] = (1 - d) + d * (summation)

            if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
#                 print("Converging at iteration " + str(iter) + "....")
                break

        return score

    def execute(self, sentence, window_size=3):
        clean_text = self.clean(sentence)
        tokens = self.generate_tokens(clean_text)
        lemmatized_text = self.lemmatize(tokens)
        processed_text = self.stopwords_based_filtering(lemmatized_text)

        vocabulary = list(set(processed_text))
        vocab_len = len(vocabulary)

        score = np.zeros((vocab_len), dtype=np.float32)
        weighted_edge, score = self.build_graph(vocabulary, processed_text, window_size, score)

        score = self.rank(vocab_len, weighted_edge, score)

        vocab_to_id = {}
        for index, word in enumerate(vocabulary):
            vocab_to_id[word] = index

        original_tokens = self.generate_tokens(sentence)
        word_scores = []
        for index, word in enumerate(original_tokens):
            if word in vocabulary:
                word_scores.append((word, score[vocab_to_id[word]]))
            else:
                word_scores.append((word, 0.0))

        return word_scores
