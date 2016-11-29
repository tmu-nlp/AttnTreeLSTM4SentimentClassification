from nltk.tree import Tree
from collections import defaultdict
from gensim.models.word2vec import Word2Vec


class Vocablary:
    def __init__(self):
        self.__word2id = defaultdict(lambda: len(self.__word2id))
        self.__id2word = dict()
        self.embed_model = None

    def read_vocab(self, vocab):
        for word in vocab:
            self.__word2id[word]
            self.__id2word[self.__word2id[word]] = word
        self.__word2id = dict(self.__word2id)

    def read_embed(self, model_file):
        self.embed_model = Word2Vec.load_word2vec_format(model_file)

    def has_word(self, word):
        return word in self.__word2id

    def word2id(self, word):
        if word not in self.__word2id:
            return self.__word2id['##unk##']
        return self.__word2id[word]

    def is_unk_id(self, wordid):
        return wordid == self.__word2id['##unk##']

    def id2word(self, num):
        assert num in self.__id2word, 'index {} out of range'.format(num)
        return self.__id2word[num]

    def dep2id(self, dep):
        assert dep in self.__dep2id, '{} not in vocablary'.format(dep)
        return self.__dep2id[dep]

    def cat2id(self, cat):
        assert cat in self.__cat2id, '{} not in vocablary'.format(cat)
        return self.__cat2id[cat]

    def is_topk_cat(self, cat, k):
        return cat in sorted(self.__cat2count, key=self.__cat2count.get)[:k]

    def is_topk_dep(self, dep, k):
        return dep in sorted(self.__dep2count, key=self.__dep2count.get)[:k]

    def get_vocab_size(self):
        return len(self.__word2id)

    def get_cat_size(self):
        return len(self.__cat2id)
