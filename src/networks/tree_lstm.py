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
    tree_attention_lstm = 3
    attention_slstm = 4

    def forget_in_size(self):
        return [1, 2, 2, 4][self.value - 1]


class TreeLSTM(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, comp_type=Composition.tree_lstm, is_regression=False, train=True, forget_bias=False, is_leaf_as_chunk=False):
        n_vocab = vocab.get_vocab_size()
        if forget_bias:
            super().__init__(
                embed=L.EmbedID(n_vocab, n_units),
                embed2hidden=L.Linear(n_units, mem_units),
                updatel=L.Linear(mem_units * 4, mem_units),
                updater=L.Linear(mem_units * 4, mem_units),
                inputl=L.Linear(mem_units * 4, mem_units),
                inputr=L.Linear(mem_units * 4, mem_units),
                forgetl=L.Linear(mem_units * comp_type.forget_in_size(), mem_units, initial_bias=np.ones(mem_units)),
                forgetr=L.Linear(mem_units * comp_type.forget_in_size(), mem_units, initial_bias=np.ones(mem_units)),
                outputl=L.Linear(mem_units * 4, mem_units),
                outputr=L.Linear(mem_units * 4, mem_units)
            )
        else:
            super().__init__(
                embed=L.EmbedID(n_vocab, n_units),
                embed2hidden=L.Linear(n_units, mem_units),
                updatel=L.Linear(mem_units * 2, mem_units),
                updater=L.Linear(mem_units * 2, mem_units),
                inputl=L.Linear(mem_units * 2, mem_units),
                inputr=L.Linear(mem_units * 2, mem_units),
                forgetl=L.Linear(mem_units * comp_type.forget_in_size(), mem_units),
                forgetr=L.Linear(mem_units * comp_type.forget_in_size(), mem_units),
                outputl=L.Linear(mem_units * 2, mem_units),
                outputr=L.Linear(mem_units * 2, mem_units)
            )
        self.__train = train
        self.__vocab = vocab
        self.is_leaf_as_chunk = is_leaf_as_chunk
        self.mem_units = mem_units
        self.n_units = n_units
        self.is_regression = is_regression
        self.comp_type = comp_type

        # init embed
        if self.__vocab.embed_model is not None:
            for i in range(self.__vocab.get_vocab_size()):
                word = self.__vocab.id2word(i)
                if word in self.__vocab.embed_model:
                    vec = self.__vocab.embed_model[word]
                    self.embed.W.data[i] = vec

    def get_word_vec(self, word):
        # TODO
        ### for old model
        if not self.__vocab.has_word(word):
            if self.__vocab.embed_model is not None:
                if word in self.__vocab.embed_model:
                    embed = self.__vocab.embed_model[word]
                    return chainer.Variable(np.array([embed]).astype('float32'))
                else:
                    return chainer.Variable(np.random.randn(1,self.n_units).astype('float32'))
            else:
                return chainer.Variable(np.random.randn(1,self.n_units, dtype=np.float32))
        ### for old model
        #if self.__vocab.is_unk_id(wordid):
        #    if self.__vocab.embed_model is not None:
        #        if word in self.__vocab.embed_model:
        #            embed = self.__vocab.embed_model[word]
        #            return chainer.Variable(np.array([embed], dtype=np.float32))
        wordid = self.__vocab.word2id(word)
        one_hot = utils.num2onehot(wordid)
        embed = self.embed(one_hot)
        return embed

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
                    embed = self.get_word_vec(tok)
                    if vector is None:
                        vector = self.embed2hidden(embed)
                    else:
                        vector += self.embed2hidden(embed)
                vector /= len(word.split('/'))
            else:
                embed = self.get_word_vec(word)
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
            concat = F.concat((left_vec, right_vec))
            u_l = self.updatel(concat)
            u_r = self.updater(concat)
            i_l = self.inputl(concat)
            i_r = self.inputr(concat)
            if self.comp_type == Composition.tree_lstm:
                f_l = self.forgetl(right_vec)
                f_r = self.forgetr(left_vec)
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


class TreeAttentionLSTM(chainer.Chain):
    def __init__(self, vocab, n_units, mem_units, attention_method, is_regression=False, train=True, forget_bias=False, is_leaf_as_chunk=False):
        n_vocab = vocab.get_vocab_size()
        comp_type = Composition.tree_attention_lstm
        if forget_bias:
            super().__init__(
                embed=L.EmbedID(n_vocab, n_units),
                embed2hidden=L.Linear(n_units, mem_units),
                updatel=L.Linear(mem_units * 4, mem_units),
                updater=L.Linear(mem_units * 4, mem_units),
                inputl=L.Linear(mem_units * 4, mem_units),
                inputr=L.Linear(mem_units * 4, mem_units),
                forgetl=L.Linear(mem_units * comp_type.forget_in_size(), mem_units, initial_bias=np.ones(mem_units)),
                forgetr=L.Linear(mem_units * comp_type.forget_in_size(), mem_units, initial_bias=np.ones(mem_units)),
                outputl=L.Linear(mem_units * 4, mem_units),
                outputr=L.Linear(mem_units * 4, mem_units)
            )
        else:
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
                outputr=L.Linear(mem_units * 4, mem_units),

                atw1=L.Linear(mem_units, mem_units),
                atw1con=L.Linear(2 * mem_units, mem_units),
                atw1gate=L.Linear(2 * mem_units, mem_units),
                atw1bi=L.Bilinear(mem_units, mem_units, 1),
                atw2=L.Linear(mem_units, 1),
            )
        self.__attention_method = attention_method
        self.__train = train
        self.__vocab = vocab
        self.is_leaf_as_chunk = is_leaf_as_chunk
        self.mem_units = mem_units
        self.n_units = n_units
        self.is_regression = is_regression
        self.comp_type = comp_type

        # init embed
        if self.__vocab.embed_model is not None:
            for i in range(self.__vocab.get_vocab_size()):
                word = self.__vocab.id2word(i)
                if word in self.__vocab.embed_model:
                    vec = self.__vocab.embed_model[word]
                    self.embed.W.data[i] = vec

    def get_word_vec(self, word):
        # TODO
        ### for old model
        if not self.__vocab.has_word(word):
            if self.__vocab.embed_model is not None:
                if word in self.__vocab.embed_model:
                    embed = self.__vocab.embed_model[word]
                    return chainer.Variable(np.array([embed]).astype('float32'))
                else:
                    return chainer.Variable(np.random.randn(1,self.n_units).astype('float32'))
            else:
                return chainer.Variable(np.random.randn(1,self.n_units, dtype=np.float32))
        ### for old model
        #if self.__vocab.is_unk_id(wordid):
        #    if self.__vocab.embed_model is not None:
        #        if word in self.__vocab.embed_model:
        #            embed = self.__vocab.embed_model[word]
        #            return chainer.Variable(np.array([embed], dtype=np.float32))
        wordid = self.__vocab.word2id(word)
        one_hot = utils.num2onehot(wordid)
        embed = self.embed(one_hot)
        return embed

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
                    embed = self.get_word_vec(tok)
                    if vector is None:
                        vector = self.embed2hidden(embed)
                    else:
                        vector += self.embed2hidden(embed)
                vector /= len(word.split('/'))
            else:
                embed = self.get_word_vec(word)
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
            left_attention_vec = self.calc_attention(left_tree)
            right_attention_vec = self.calc_attention(right_tree)
            concat = F.concat((left_vec, right_vec, left_attention_vec, right_attention_vec))
            u_l = self.updatel(concat)
            u_r = self.updater(concat)
            i_l = self.inputl(concat)
            i_r = self.inputr(concat)
            if self.comp_type == Composition.tree_attention_lstm:
                concatl = F.concat((left_vec, left_attention_vec))
                concatr = F.concat((right_vec, right_attention_vec))
                f_l = self.forgetl(concatr)
                f_r = self.forgetr(concatl)
            elif self.comp_type == Composition.attention_slstm:
                f_l = self.forgetl(concat)
                f_r = self.forgetr(concat)
            o_l = self.outputl(concat)
            o_r = self.outputr(concat)
            l_v = F.concat((u_l, i_l, f_l, o_l))
            r_v = F.concat((u_r, i_r, f_r, o_r))
            c, vector = F.slstm(leftc, rightc, l_v, r_v)

        tree.data['vector'] = vector
        if tree.is_root():
            self.calc_attention(tree)
        return c

    def calc_attention(self, tree):
        sume = chainer.Variable(np.array([[0]], dtype=np.float32))
        root_vec = tree.data['vector']
        for subtree in tree.subtrees():
            # skip the node whose child is only one
            if 'vector' not in subtree.data:
                continue
            phrase_vec = subtree.data['vector']
            #if self.__attention_target == 'word' and not subtree.is_leaf():
            #    subtree.data['attention_weight'] = chainer.Variable(np.array([[0]], dtype=np.float32))
            #    continue
            #elif self.__attention_target == 'phrase' and subtree.is_leaf():
            #    subtree.data['attention_weight'] = chainer.Variable(np.array([[0]], dtype=np.float32))
            #    continue
            if self.__attention_method == 'only':
                tmp = F.tanh(self.atw1(phrase_vec))
                e = F.exp(self.atw2(tmp))
            elif self.__attention_method == 'concat':
                tmp = F.tanh(self.atw1con(F.concat((phrase_vec, root_vec))))
                e = F.exp(self.atw2(tmp))
            elif self.__attention_method == 'bilinear':
                e = F.exp(F.tanh(self.atw1bi(phrase_vec, root_vec)))
            elif self.__attention_method == 'dot':
                e = F.exp(F.tanh(F.matmul(phrase_vec, root_vec, transb=True)))
            elif self.__attention_method == 'gate':
                gate = F.sigmoid(self.atw1gate(F.concat((phrase_vec, root_vec))))
                tmp = F.tanh(self.atw1con(F.concat((phrase_vec, root_vec))))
                e = F.exp(self.atw2(tmp * gate))
            else:
                assert False, 'illegal attention method name'
            subtree.data['attention_weight'] = e
            sume += e
        attention_vec = chainer.Variable(np.zeros(root_vec.data.shape, dtype=np.float32))

        top5 = list()
        for subtree in tree.subtrees():
            # skip the node whose child is only one
            if 'vector' not in subtree.data:
                continue
            if float(subtree.data['attention_weight'].data) != 0:
                attention_weight = subtree.data['attention_weight'] / sume
                subtree.data['attention_weight'] = attention_weight
                attention_vec += F.matmul(attention_weight, subtree.data['vector'])

                # create top5 attentted list
                if tree.is_root():
                    if len(top5) < 5:
                        top5.append(subtree)
                    elif min(float(t.data['attention_weight'].data) for t in top5) < float(attention_weight.data):
                        minf = min(float(t.data['attention_weight'].data) for t in top5)
                        flist = [float(t.data['attention_weight'].data) for t in top5]
                        i = flist.index(minf)
                        top5.pop(i)
                        top5.append(subtree)
        if tree.is_root():
            for i, tree in enumerate(sorted(top5, key=lambda t: -float(t.data['attention_weight'].data))):
                tree.data['top{}'.format(i + 1)] = True

        return attention_vec
