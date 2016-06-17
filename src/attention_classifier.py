import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import utils


class AttentionClassifier(chainer.Chain):
    def __init__(self, mem_units, label_num, attention_method, attention_target, dropout_ratio, is_regression=False):
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
        self.__attention_method = attention_method
        self.__attention_target = attention_target
        self.__is_regression = is_regression
        self.__dropout_ratio = dropout_ratio

    def __call__(self, tree, is_train=True):
        total_loss = 0
        for subtree in tree.subtrees():
            loss, is_correct = self.classify_one(subtree, is_train)
            if loss is not None:
                total_loss += loss

        # to log the attention weights for root node, root node should be classified lastly
        loss, is_correct = self.classify_one(tree, is_train)
        total_loss += loss
        self.__count['total_root'] += 1
        if is_correct:
            self.__count['correct_root'] += 1
        return total_loss

    def classify_one(self, tree, is_train):
        label = None
        # get label
        if 'label' in tree.get_label():
            typ = int
            if self.__is_regression:
                typ = float
            label = typ(tree.get_label().split('label:')[1].split(',')[0])
        elif tree.get_label().isdigit():
            typ = int
            if self.__is_regression:
                typ = float
            label = typ(tree.get_label())

        # classify
        if label is not None:
            y = self.decode_one(tree, is_train)
            loss, pred, dist = self.calc_loss(y, label)
            tree.data['correct'] = label
            tree.data['dist'] = dist
            tree.data['correct_pred'] = 'T:{},Y:{}'.format(label, pred)
            tree.data['is_correct'] = pred == label
            return loss, tree.data['is_correct']
        return None, None

    def calc_loss(self, y, t):
        if self.__is_regression:
            correct = chainer.Variable(numpy.array([[t]], dtype=numpy.float32))
            loss = F.mean_squared_error(y, correct)
            dist = y.data
            pred = 1 if float(y.data) >= 0.5 else 0
        else:
            correct = utils.num2onehot(t)
            loss = F.softmax_cross_entropy(y, correct)
            dist = F.softmax(y).data
            pred = numpy.argmax(dist)
        return loss, pred, dist

    def decode_one(self, tree, is_train=True):
        in_vec = tree.data['vector']
        if self.__attention_method is not None:
            attention_vec = self.calc_attention(tree)
            if self.__is_regression:
                y = F.sigmoid(self.atout_reg(F.dropout(F.concat((in_vec, attention_vec)), ratio=self.__dropout_ratio, train=is_train)))
            else:
                y = self.atout_class(F.dropout(F.concat((in_vec, attention_vec)), ratio=self.__dropout_ratio, train=is_train))
        else:
            if self.__is_regression:
                y = F.sigmoid(self.out_reg(F.dropout(in_vec, ratio=self.__dropout_ratio, train=is_train)))
            else:
                y = self.out_class(F.dropout(in_vec, ratio=self.__dropout_ratio, train=is_train))
        return y

    def calc_attention(self, tree):
        sume = chainer.Variable(numpy.array([[0]], dtype=numpy.float32))
        root_vec = tree.data['vector']
        for subtree in tree.subtrees():
            phrase_vec = subtree.data['vector']
            if self.__attention_target == 'word' and not subtree.is_leaf():
                subtree.data['attention_weight'] = chainer.Variable(numpy.array([[0]], dtype=numpy.float32))
                continue
            elif self.__attention_target == 'phrase' and subtree.is_leaf():
                subtree.data['attention_weight'] = chainer.Variable(numpy.array([[0]], dtype=numpy.float32))
                continue
            elif self.__attention_method == 'only':
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
        attention_vec = chainer.Variable(numpy.zeros(root_vec.data.shape, dtype=numpy.float32))

        top5 = list()
        for subtree in tree.subtrees():
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

    def init_count(self):
        self.__count = {'total_root': 0, 'correct_root': 0}

    def calc_acc(self):
        if self.__count['total_root'] == 0:
            root_acc = 0
        else:
            root_acc = self.__count['correct_root'] / self.__count['total_root']
        return root_acc

    def get_count(self):
        return self.__count['total_root'], self.__count['correct_root']
