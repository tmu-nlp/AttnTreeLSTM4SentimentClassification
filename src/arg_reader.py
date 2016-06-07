import decoparser
import utils
import pickle
import os
from chainer import optimizers


@decoparser.option('--kind', required=True, choices=('constituency', 'dependency'))
@decoparser.option('--lang', required=True, choices=('Ja', 'En'))
@decoparser.option('--logdir', required=True)
@decoparser.option('--optimizer', choices=('SGD', 'AdaDelta', 'AdaGrad', 'Adam', 'NesterovAG'), default='Adam')
@decoparser.option('--composition', choices=('SLSTM', 'TreeLSTM'), default='SLSTM')
@decoparser.option('--lr', type=float, default=0.01)
@decoparser.option('--l2', type=float, default=0.0001)
@decoparser.option('--clip-grad', type=float, default=5)
@decoparser.option('--mem-units', type=int, default=100)
@decoparser.option('--attention', choices=('only', 'concat', 'bilinear', 'dot', 'gate'))
@decoparser.option('--regression', action='store_true')
@decoparser.option('--toy', action='store_true')
@decoparser.option('--pol-dict', choices=('pn', 'pnn'))
@decoparser.option('--not-render', action='store_false', default=True)
def get_arg(arg, kind, logdir, optimizer, lr, l2, clip_grad, mem_units, attention, toy, pol_dict, regression, composition, lang, not_render):
    args = ['kind', 'logdir', 'lr', 'l2', 'clip_grad', 'attention', 'toy', 'pol_dict', 'optimizer', 'mem_units', 'regression', 'composition', 'lang', 'not_render']
    d = dict()
    for a in args:
        d[a] = eval(a)
    if arg in d:
        return d[arg]
    elif arg is 'all':
        return d
    else:
        return None


class ArgReader:
    def __init__(self):
        self.__fold = 10
        self.__batch_size = 32
        self.__n_epoch = 10
        self.__label_num = 2
        self.__mem_units = get_arg('mem_units')
        self.__mode = get_arg('kind')
        self.__lang = get_arg('lang')
        self.__is_toy = get_arg('toy')
        self.__dict = get_arg('pol_dict')
        self.__logdir = get_arg('logdir')
        self.__is_regression = get_arg('regression')
        self.__attention = get_arg('attention')
        self.__lr = get_arg('lr')
        self.__l2 = get_arg('l2')
        self.__clip_grad = get_arg('clip_grad')
        self.__composition = get_arg('composition')
        self.__n_units = 200
        self.__render_graph = get_arg('not_render')
        if self.__lang == 'En':
            self.__n_units = 300

        # optimizer
        self.__opt_name = get_arg('optimizer')
        if self.__opt_name == 'SGD':
            self.__opt = lambda: optimizers.SGD(lr=self.__lr)
        elif self.__opt_name == 'AdaDelta':
            self.__opt = lambda: optimizers.AdaDelta()
        elif self.__opt_name == 'Adam':
            self.__opt = lambda: optimizers.Adam(alpha=self.__lr)
        elif self.__opt_name == 'NesterovAg':
            self.__opt = lambda: optimizers.NesterovAG(lr=self.__lr)

        # data
        data_dir = utils.get_data_path()
        mecab_embedf = data_dir + '/vector/word2vec/wikiDump_mecab_size200_cbow.w2vModel'
        kytea_embedf = data_dir + '/vector/word2vec/wikiDump_kytea_size200_skipgram.w2vModel'
        en_embedf = data_dir + '/vector/glove/glove.840B.300d.txt'
        if self.__lang == 'Ja':
            if self.__mode == 'constituency':
                data = data_dir + '/ntcirj/ckylark/data.pkl.bz2'
                self.__embedf = kytea_embedf
            elif self.__mode == 'dependency':
                data = data_dir + '/ntcirj/cabocha/data.pkl.bz2'
                self.__embedf = mecab_embedf
        elif self.__lang == 'En':
            self.__embedf = en_embedf
            data = data_dir + '/ntcire/ckylark/data.pkl.bz2'
        data = utils.read_pkl_bz2(data)
        if self.__is_toy:
            data = data['toy']
        if self.__dict == 'pn':
            data = data['poldict']
        elif self.__dict == 'pnn':
            data = data['poldict_neutral']
        self.__data = data
        self.mk_logfiles()
        self.print_params()

    def print_params(self):
        with open(self.__logdir + '/params.txt', 'w') as f:
            print('toy data: {}'.format(self.__is_toy), file=f)
            print('mem_units: {}'.format(self.__mem_units), file=f)
            print('optimizer: {}'.format(self.__opt_name), file=f)
            print('compositoin network: {}'.format(self.__composition), file=f)
            print('learning rate: {}'.format(self.__lr), file=f)
            print('l2: {}'.format(self.__l2), file=f)
            print('clip grad: {}'.format(self.__clip_grad), file=f)
            print('attention: {}'.format(self.__attention), file=f)
            print('regression: {}'.format(self.__is_regression), file=f)
            print('batch size: {}'.format(self.__batch_size), file=f)
            print('pol dict: {}'.format(self.__dict), file=f)
            print('lang: {}'.format(self.__lang), file=f)
            print('mode: {}'.format(self.__mode), file=f)

    def get_fold(self):
        return self.__fold

    def get_mode(self):
        return self.__mode

    def get_lang(self):
        return self.__lang

    def get_embedf(self):
        return self.__embedf

    def get_data(self):
        return self.__data

    def mk_logfiles(self):
        logf = self.__logdir + '/log.txt'
        progf = self.__logdir + '/progress.txt'
        model_dir = self.__logdir + '/models'
        ex_dir = self.__logdir + '/example'
        fold_dir = self.__logdir + '/each_fold'
        graph_dir = self.__logdir + '/graph'
        graph_dirs = [graph_dir + '/fold{}'.format(i) for i in range(self.__fold)]
        if not os.path.exists(self.__logdir):
            os.mkdir(self.__logdir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(ex_dir):
            os.mkdir(ex_dir)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)
        for d in graph_dirs:
            if not os.path.exists(d):
                os.mkdir(d)

    def get_logfiles(self):
        logf = self.__logdir + '/log.txt'
        progf = self.__logdir + '/progress.txt'
        model_dir = self.__logdir + '/models'
        ex_dir = self.__logdir + '/example'
        fold_dir = self.__logdir + '/each_fold'
        graph_dir = self.__logdir + '/graph'
        graph_dirs = [graph_dir + '/fold{}'.format(i) for i in range(self.__fold)]
        if not os.path.exists(self.__logdir):
            os.mkdir(self.__logdir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(ex_dir):
            os.mkdir(ex_dir)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)
        for d in graph_dirs:
            if not os.path.exists(d):
                os.mkdir(d)
        return logf, progf, model_dir, ex_dir, fold_dir, graph_dirs

    def is_regression(self):
        return self.__is_regression

    def get_attention(self):
        return self.__attention

    def get_n_units(self):
        return self.__n_units

    def get_mem_units(self):
        return self.__mem_units

    def get_batch_size(self):
        return self.__batch_size

    def get_n_epoch(self):
        return self.__n_epoch

    def get_label_num(self):
        return self.__label_num

    def get_lr(self):
        return self.__lr

    def get_l2(self):
        return self.__l2

    def get_clip_grad(self):
        return self.__clip_grad

    def get_optimizer(self):
        return self.__opt

    def get_composition(self):
        return self.__composition

    def is_toy(self):
        return self.__is_toy

    def render_grpah(self):
        return self.__render_graph

    def output_result(self, test, dev):
        resultf = self.__logdir + '/result.pkl'
        d = get_arg('all')
        d['test'] = test
        d['dev'] = dev
        pickle.dump(d, open(resultf, 'wb'))
