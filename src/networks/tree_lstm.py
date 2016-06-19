import chainer
import chainer.functions as F
import chainer.links as L
from chainer.variable import Variable
import numpy as np
import utils
from enum import Enum


class Composition(Enum):
    tree_lstm = 1
    slstm = 2

    def forget_in_size(self):
        return [2, 4][self.value - 1]


class TreeLSTM(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, comp_type=Composition.tree_lstm, is_regression=False, train=True, is_leaf_as_chunk=False):
        n_vocab = vocab.get_vocab_size()
        super().__init__(
            embed=L.EmbedID(n_vocab, n_units),
            embed2hidden=L.Linear(n_units, mem_units),
            updatel=L.Linear(mem_units * 4, mem_units),
            updater=L.Linear(mem_units * 4, mem_units),
            inputl=L.Linear(mem_units * 4, mem_units),
            inputr=L.Linear(mem_units * 4, mem_units),
            forgetl=L.Linear(mem_units * comp_type.forget_in_size(), mem_units),
            forgetr=L.Linear(mem_units * comp_type.forget_in_size(), mem_units),
            outputl=L.Linear(mem_units * 4, mem_units),
            outputr=L.Linear(mem_units * 4, mem_units)
        )
        self.__train = train
        self.__vocab = vocab
        self.is_leaf_as_chunk = is_leaf_as_chunk
        self.mem_units = mem_units
        self.is_regression = is_regression
        self.comp_type = comp_type

        # init embed
        if self.__vocab.embed_model is not None:
            for i in range(self.__vocab.get_vocab_size()):
                word = self.__vocab.id2word(i)
                if word in self.__vocab.embed_model:
                    vec = self.__vocab.embed_model[word]
                    self.embed.W.data[i] = vec

    def __call__(self, tree):
        # skip the node if whose child is only one
        while len(tree.children) == 1 and not tree.is_leaf():
            tree = tree.children[0]
        if tree.is_leaf():
            word = tree.get_word()
            # avg
            if self.is_leaf_as_chunk:
                vector = None
                for tok in word.split('/'):
                    wordid = self.__vocab.word2id(tok)
                    one_hot = utils.num2onehot(wordid)
                    embed = self.embed(one_hot)
                    if vector is None:
                        vector = self.embed2hidden(embed)
                    else:
                        vector += self.embed2hidden(embed)
                vector /= len(word.split('/'))
            else:
                wordid = self.__vocab.word2id(word)
                one_hot = utils.num2onehot(wordid)
                embed = self.embed(one_hot)
                vector = self.embed2hidden(embed)
            c = Variable(np.zeros((1, self.mem_units), dtype=np.float32))
        else:
            left_tree, right_tree = tree.children
            leftc = self(left_tree)
            rightc = self(right_tree)
            # skip the node if whose child is only one
            while len(left_tree.children) == 1 and not left_tree.is_leaf():
                left_tree = left_tree.children[0]
            while len(right_tree.children) == 1 and not right_tree.is_leaf():
                right_tree = right_tree.children[0]
            left_vec = left_tree.data['vector']
            right_vec = right_tree.data['vector']

            # composition by tree lstm
            concat = F.concat((left_vec, right_vec, leftc, rightc))
            u_l = self.updatel(concat)
            u_r = self.updater(concat)
            i_l = self.inputl(concat)
            i_r = self.inputr(concat)
            if self.comp_type == Composition.tree_lstm:
                concatl = F.concat((left_vec, leftc))
                concatr = F.concat((right_vec, rightc))
                f_l = self.forgetl(concatr)
                f_r = self.forgetr(concatl)
            elif self.comp_type == Composition.slstm:
                f_l = self.forgetl(concat)
                f_r = self.forgetr(concat)
            o_l = self.outputl(concat)
            o_r = self.outputr(concat)
            l_v = F.concat((u_l, i_l, f_l, o_l))
            r_v = F.concat((u_r, i_r, f_r, o_r))
            c, vector = F.slstm(leftc, rightc, l_v, r_v)

        tree.data['vector'] = vector
        return c
