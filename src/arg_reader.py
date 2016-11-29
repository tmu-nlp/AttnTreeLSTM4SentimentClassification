import decoparser
import utils
import pickle
import os
from chainer import optimizers
from networks import Composition
import time

@decoparser.option('--data', required=True, choices=('ntcirj_con', 'ntcirj_dep', 'ntcire', 'tsukuba', 'sst_all', 'sst_cut'))
@decoparser.option('--logdir', required=True)
@decoparser.option('--optimizer', choices=('SGD', 'AdaDelta', 'AdaGrad', 'Adam', 'NesterovAG'), default='Adam')
@decoparser.option('--composition', type=decoparser.Enum, choices=Composition, default=Composition.tree_lstm)
@decoparser.option('--lr', type=float, default=0.01)
@decoparser.option('--l2', type=float, default=0.0001)
@decoparser.option('--clip-grad', type=float, default=5)
@decoparser.option('--dropout', type=float, default=0.0)
@decoparser.option('--mem-units', type=int, default=200)
@decoparser.option('--attention', choices=('only', 'concat', 'bilinear', 'dot', 'gate'))
@decoparser.option('--attention-target', choices=('all', 'word', 'phrase'), default='all')
@decoparser.option('--regression', action='store_true')
@decoparser.option('--only-attn', action='store_true')
@decoparser.option('--toy', action='store_true')
@decoparser.option('--pol-dict', choices=('pn', 'pnn'))
@decoparser.option('--not-render', action='store_false', default=True)
@decoparser.option('--not-embed', action='store_true')
@decoparser.option('--forget-bias', action='store_true')
@decoparser.option('--always-attn', action='store_true')
def get_arg(arg, logdir, optimizer, lr, l2, clip_grad, mem_units, attention, toy, pol_dict, regression, composition, data, not_render, not_embed, attention_target, dropout, forget_bias, only_attn, always_attn):
    args = ['logdir', 'lr', 'l2', 'clip_grad', 'attention', 'toy', 'pol_dict', 'optimizer', 'mem_units', 'regression', 'composition', 'data', 'not_render', 'not_embed', 'attention_target', 'dropout', 'forget_bias', 'only_attn', 'always_attn']
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
        self.__start_time = time.time()
        self.__always_attn = get_arg('always_attn')
        self.__fold = 10
        self.__batch_size = 32
        self.__n_epoch = 10
        self.__label_num = 2
        self.__mem_units = get_arg('mem_units')
        self.__data_mode = get_arg('data')
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
        self.__dropout = get_arg('dropout')
        if self.__data_mode == 'ntcire' or self.__data_mode.startswith('sst'):
            self.__n_units = 300
        self.__not_embed = get_arg('not_embed')
        self.__attention_target = get_arg('attention_target')
        self.__forget_bias = get_arg('forget_bias')
        self.__only_attn = get_arg('only_attn')

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
        elif self.__opt_name == 'AdaGrad':
            self.__opt = lambda: optimizers.AdaGrad(lr=self.__lr)

        # data
        data_dir = utils.get_data_path()
        mecab_embedf = data_dir + '/vector/word2vec/wikiDump_mecab_size200_cbow.w2vModel'
        kytea_embedf = data_dir + '/vector/word2vec/wikiDump_kytea_size200_skipgram.w2vModel'
        en_embedf = data_dir + '/vector/glove/glove.840B.300d.txt'
        if self.__data_mode== 'ntcirj_con':
            data = data_dir + '/ntcirj/ckylark/data.pkl.bz2'
            self.__embedf = kytea_embedf
        if self.__data_mode == 'ntcirj_dep':
            data = data_dir + '/ntcirj/cabocha/data.pkl.bz2'
            self.__embedf = mecab_embedf
        elif self.__data_mode == 'ntcire':
            self.__embedf = en_embedf
            data = data_dir + '/ntcire/ckylark/data.pkl.bz2'
        elif self.__data_mode == 'tsukuba':
            self.__embedf = kytea_embedf
            data = data_dir + '/tsukuba/ckylark/data.pkl.bz2'
        elif self.__data_mode == 'sst_all':
            self.__embedf = en_embedf
            self.__label_num = 5
            data = data_dir + '/sst_all/data.pkl.bz2'
        elif self.__data_mode == 'sst_cut':
            self.__embedf = en_embedf
            self.__label_num = 5
            data = data_dir + '/sst_cut/data.pkl.bz2'
        data = utils.read_pkl_bz2(data)
        if self.__is_toy:
            data = data['toy']
            self.__n_epoch = 3
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
            print('compositoin network: {}'.format(self.__composition.name), file=f)
            print('learning rate: {}'.format(self.__lr), file=f)
            print('l2: {}'.format(self.__l2), file=f)
            print('clip grad: {}'.format(self.__clip_grad), file=f)
            print('dropout ratio: {}'.format(self.__dropout), file=f)
            print('attention: {}'.format(self.__attention), file=f)
            print('attention target: {}'.format(self.__attention_target), file=f)
            print('forget_bias: {}'.format(self.__forget_bias), file=f)
            print('regression: {}'.format(self.__is_regression), file=f)
            print('batch size: {}'.format(self.__batch_size), file=f)
            print('pol dict: {}'.format(self.__dict), file=f)
            print('data: {}'.format(self.__data_mode), file=f)
            print('only attn: {}'.format(self.__only_attn), file=f)
            print('always attn: {}'.format(self.__always_attn), file=f)

    def get_fold(self):
        return self.__fold

    def get_mode(self):
        if self.__data_mode == 'ntcirj_con':
            return 'constituency'
        elif self.__data_mode == 'ntcirj_dep':
            return 'dependency'

    def get_data(self):
        return self.__data

    def get_embedf(self):
        return self.__embedf

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

    def get_attention_target(self):
        return self.__attention_target

    def get_dropout(self):
        return self.__dropout

    def is_toy(self):
        return self.__is_toy

    def is_only_attn(self):
        return self.__only_attn

    def is_always_attn(self):
        return self.__always_attn

    def render_grpah(self):
        return self.__render_graph

    def forget_bias(self):
        return self.__forget_bias

    def output_result(self, test, dev):
        resultf = self.__logdir + '/result.pkl'
        elapsed_time = time.time() - self.__start_time
        d = get_arg('all')
        d['test'] = test
        d['dev'] = dev
        d['composition'] = d['composition'].name
        d['time'] = utils.time_format(elapsed_time)
        pickle.dump(d, open(resultf, 'wb'))

        script = 'import pickle\n' +\
        "for k, v in pickle.load(open('result.pkl', 'rb')).items():\n" +\
        "    print('{}: {}'.format(k, v))"
        print(script, file=open(self.__logdir + '/print_result.py', 'w'))

    def use_embed(self):
        return not self.__not_embed
