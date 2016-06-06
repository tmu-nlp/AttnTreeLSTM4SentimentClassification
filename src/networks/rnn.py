import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import utils


class RNN(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, train=True, is_leaf_as_chunk=False):
        n_vocab = vocab.get_vocab_size()
        super().__init__(
            embed=L.EmbedID(n_vocab, n_units),
            embed2hidden=L.Linear(n_units, mem_units),
            hidden=L.Linear(mem_units * 2, mem_units)
        )
        self.__train = train
        self.__vocab = vocab
        self.is_leaf_as_chunk = is_leaf_as_chunk

    def init_embed(self):
        if self.__vocab.embed_model is None:
            return
        for i in range(self.__vocab.get_vocab_size()):
            word = self.__vocab.id2word(i)
            if word in self.__vocab.embed_model:
                vec = self.__vocab.embed_model[word]
                self.embed.W[i] = vec

    def __call__(self, tree, evaluate=None):
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
            total_loss = 0
        else:
            left_tree, right_tree = tree.get_children()
            left_vec, left_loss = self(left_tree, evaluate)
            right_vec, right_loss = self(right_tree, evaluate)
            concat = F.concat((left_vec, right_vec))
            vector = self.hidden(concat)
            total_loss = left_loss + right_loss
        if evaluate is not None:
            label = int(tree.get_label())
            y, loss = evaluate(vector, tree.is_root(), label)
            total_loss += loss
        return vector, total_loss

    def read_glove(self, glovef):
        for line in utils.progressbar(
                open(glovef), maxval=utils.get_line_num(glovef),
                message='read glove'):
            word = line[:line.find(' ')]
            if self.__vocab.has_word(word):
                index = self.__vocab.word2id(word)
                strvec = ' '.join(line.split()[1:])
                vec = np.fromstring(strvec, dtype=np.float32, sep=' ')
                self.embed.W.data[index] = vec
