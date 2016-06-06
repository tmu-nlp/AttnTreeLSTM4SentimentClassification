import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import utils


class TERNN(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, cat_units, train=True):
        n_vocab = vocab.get_vocab_size()
        n_cat = vocab.get_cat_size()
        super().__init__(
            embed=L.EmbedID(n_vocab, n_units),
            catembed=L.EmbedID(n_cat, cat_units),
            embed2hidden=L.Linear(n_units, mem_units),
            hidden=L.Linear(mem_units * 2 + cat_units * 2, mem_units)
        )
        self.__train = train
        self.__vocab = vocab

    def __call__(self, tree, evaluate=None):
        if tree.is_leaf():
            word = tree.get_word()
            wordid = self.__vocab.word2id(word)
            one_hot = utils.num2onehot(wordid)
            embed = self.embed(one_hot)
            vector = self.embed2hidden(embed)
            total_loss = 0
        else:
            left_tree, right_tree = tree.get_children()
            left_vec, left_cat_vec, left_loss = self(left_tree, evaluate)
            right_vec, right_cat_vec, right_loss = self(right_tree, evaluate)
            concat = F.concat(
                (left_vec, left_cat_vec, right_vec, right_cat_vec))
            vector = self.hidden(concat)
            total_loss = left_loss + right_loss
        cat = tree.get_label()[tree.get_label().find(':') + 1:]
        catid = self.__vocab.cat2id(cat)
        one_hot = utils.num2onehot(catid)
        cat_vec = self.catembed(one_hot)

        if evaluate is not None:
            label = int(tree.get_label()[:tree.get_label().find(':')])
            x = F.concat((vector, cat_vec))
            y, loss = evaluate(x, tree.is_root(), label)
            total_loss += loss
        return vector, cat_vec, total_loss

    def read_glove(self, glovef):
        for line in open(glovef):
            word = line[:line.find(' ')]
            if self.__vocab.has_word(word):
                index = self.__vocab.word2id(word)
                strvec = ' '.join(line.split()[1:])
                vec = np.fromstring(strvec, dtype=np.float32, sep=' ')
                self.embed.W.data[index] = vec
