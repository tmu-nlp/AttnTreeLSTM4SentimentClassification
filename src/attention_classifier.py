import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import utils


class AttentionClassifier(chainer.Chain):
    def __init__(self, mem_units, label_num, method='only'):
        super().__init__(
            atw1=L.Linear(mem_units, mem_units),
            atw1con=L.Linear(2 * mem_units, mem_units),
            atw1gate=L.Linear(2 * mem_units, mem_units),
            atw1bi=L.Bilinear(mem_units, mem_units, 1),
            atw2=L.Linear(mem_units, 1),
            atout_class=L.Linear(mem_units + mem_units, label_num),
            atout_reg=L.Linear(mem_units + mem_units, 1),
            out_class=L.Linear(mem_units, label_num),
            out_reg=L.Linear(mem_units, 1))
        self.__count = {'total_root': 0, 'correct_root': 0}
        self.__method = method

    def __call__(self, x, correct=None, attention_mems=None, is_root=True, regression=False, tree=None):
        attention = self.calc_attention(x, attention_mems, tree)
        y = self.classify(x, attention, regression)

        loss = None
        if correct is not None:
            self.label = correct
            if regression:
                correct = chainer.Variable(numpy.array([[correct]], dtype=numpy.float32))
                loss = F.mean_squared_error(y, correct)
                self.dist = y.data
                self.pred = 1 if float(y.data) >= 0.5 else 0
            else:
                correct = utils.num2onehot(correct)
                loss = F.softmax_cross_entropy(y, correct)
                self.dist = F.softmax(y).data
                self.pred = numpy.argmax(self.dist)

            if is_root:
                self.__count['total_root'] += 1
                if tree is not None:
                    tree.data['render_label'].append('T:{},Y:{}'.format(correct.data[0], self.pred))
            if correct.data[0] == self.pred:
                if is_root:
                    self.__count['correct_root'] += 1
        return y, loss

    def classify(self, x, attention, regression):
        if attention is not None:
            if regression:
                y = F.sigmoid(self.atout_reg(F.concat((x, attention))))
            else:
                y = self.atout_class(F.concat((x, attention)))
        else:
            if regression:
                y = F.sigmoid(self.out_reg(x))
            else:
                y = self.out_class(x)
        return y

    def calc_attention(self, x, attention_mems, tree):
        if attention_mems is None:
            return
        sume = chainer.Variable(numpy.array([[0]], dtype=numpy.float32))
        e_list = list()
        for phmem in attention_mems:
            if self.__method == 'only':
                tmp = F.tanh(self.atw1(phmem))
                e = F.exp(self.atw2(tmp))
            elif self.__method == 'concat':
                tmp = F.tanh(self.atw1con(F.concat((phmem, x))))
                e = F.exp(self.atw2(tmp))
            elif self.__method == 'bilinear':
                e = F.exp(F.tanh(self.atw1bi(phmem, x)))
            elif self.__method == 'dot':
                e = F.exp(F.tanh(F.matmul(phmem, x, transb=True)))
            elif self.__method == 'gate':
                gate = F.sigmoid(self.atw1gate(F.concat((phmem, x))))
                tmp = F.tanh(self.atw1con(F.concat((phmem, x))))
                e = F.exp(self.atw2(tmp * gate))
            else:
                assert False, 'illegal attention method name'
            e_list.append(e)
            sume += e
        attention = chainer.Variable(numpy.zeros(x.data.shape, dtype=numpy.float32))
        for phmem, e in zip(attention_mems, e_list):
            a = e / sume
            if tree is not None:
                for subt in tree.subtrees():
                    if tuple(phmem.data.flatten().tolist()) in subt.data:
                        subt.data['render_label'].append('({:0>.3})'.format(float(a.data)))
            attention += F.matmul(a, phmem)
        return attention

    def init_count(self):
        self.__count = {'total_root': 0, 'correct_root': 0}

    def calc_acc(self):
        if self.__count['total_root'] == 0:
            root_acc = 0
        else:
            root_acc = self.__count['correct_root'] / self.__count['total_root']
        return root_acc
