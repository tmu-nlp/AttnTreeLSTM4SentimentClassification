import chainer
import chainer.functions as F
import chainer.links as L
from chainer.variable import Variable
import numpy as np
import utils


class TreeLSTM(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, is_regression=False, train=True, is_leaf_as_chunk=False):
        n_vocab = vocab.get_vocab_size()
        super().__init__(
            embed=L.EmbedID(n_vocab, n_units),
            embed2hidden=L.Linear(n_units, mem_units),
            updatel=L.Linear(mem_units * 4, mem_units),
            updater=L.Linear(mem_units * 4, mem_units),
            inputl=L.Linear(mem_units * 4, mem_units),
            inputr=L.Linear(mem_units * 4, mem_units),
            forgetl=L.Linear(mem_units * 2, mem_units),
            forgetr=L.Linear(mem_units * 2, mem_units),
            outputl=L.Linear(mem_units * 4, mem_units),
            outputr=L.Linear(mem_units * 4, mem_units)
        )
        self.__train = train
        self.__vocab = vocab
        self.is_leaf_as_chunk = is_leaf_as_chunk
        self.mem_units = mem_units
        self.is_regression = is_regression

        # init embed
        if self.__vocab.embed_model is not None:
            for i in range(self.__vocab.get_vocab_size()):
                word = self.__vocab.id2word(i)
                if word in self.__vocab.embed_model:
                    vec = self.__vocab.embed_model[word]
                    self.embed.W.data[i] = vec

    def __call__(self, tree, attention_mems, classifier=None, visualize=False):
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
            total_loss = 0
        else:
            left_tree, right_tree = tree.children
            left_vec, leftc, left_loss = self(left_tree, attention_mems, classifier, visualize)
            right_vec, rightc, right_loss = self(right_tree, attention_mems, classifier, visualize)

            concat = F.concat((left_vec, right_vec, leftc, rightc))
            concatl = F.concat((left_vec, leftc))
            concatr = F.concat((right_vec, rightc))
            u_l = self.updatel(concat)
            u_r = self.updater(concat)
            i_l = self.inputl(concat)
            i_r = self.inputr(concat)
            f_l = self.forgetl(concatr)
            f_r = self.forgetr(concatl)
            o_l = self.outputl(concat)
            o_r = self.outputr(concat)
            l_v = F.concat((u_l, i_l, f_l, o_l))
            r_v = F.concat((u_r, i_r, f_r, o_r))
            c, vector = F.slstm(leftc, rightc, l_v, r_v)

            total_loss = left_loss + right_loss

        if classifier is not None:
            label = None
            if 'label' in tree.get_label():
                typ = int
                if self.is_regression:
                    typ = float
                label = typ(tree.get_label().split('label:')[1].split(',')[0])
            elif tree.get_label().isdigit():
                typ = int
                if self.is_regression:
                    typ = float
                label = typ(tree.get_label())
            if label is not None:
                y, loss = classifier(vector, label, attention_mems, tree.is_root(), self.is_regression, tree if visualize else None)
                total_loss += loss
        if attention_mems is not None:
            attention_mems.append(vector)
            if visualize:
                tree.data[tuple(vector.data.flatten().tolist())] = None
        return vector, c, total_loss
