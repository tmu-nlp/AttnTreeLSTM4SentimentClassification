import copy
import pickle
import random
import re
from arg_reader import ArgReader

from mprogressbar import ProgressManager
import multiprocessing as mp

from tree import Tree
from attention_classifier import AttentionClassifier
from vocablary import Vocablary
from networks import SLSTM
from networks import TreeLSTM
import chainer


def log_writer(results):
    none_count = 0
    on_way_acc = dict()
    for e in range(n_epoch):
        on_way_acc[e] = list()
    dev_acc_re = re.compile('(?P<fold>[0-9])dev ep(?P<epoch>[0-9]*?): acc=(?P<acc>[0-9.]*)')

    with open(logf, 'w') as f:
        while True:
            item = logs.get()
            if item is None:
                none_count += 1
                if none_count == fold:
                    break
            else:
                print(item, file=f, flush=True)
                match = dev_acc_re.search(item)
                if match is not None:
                    # fold_num = int(match.group('fold'))
                    epoch_num = int(match.group('epoch'))
                    acc_num = float(match.group('acc'))
                    on_way_acc[epoch_num].append(acc_num)
                    # output on way dev acc
                    if len(on_way_acc[epoch_num]) == fold:
                        ave = sum(on_way_acc[epoch_num]) / fold
                        print('dev average ep{}: acc={:0>2}'.format(epoch_num, ave), file=f, flush=True)
        print('dev', file=f)
        devs = [val for key, val in results.items() if 'dev' in key]
        for acc in devs:
            print(acc, file=f)
        ave = sum(devs) / len(devs)
        print('ave: {}'.format(ave), file=f)
        results['dev_ave'] = ave

        print('', file=f)
        print('test', file=f)
        tests = [val for key, val in results.items() if 'test' in key]
        for acc in tests:
            print(acc, file=f)
        ave = sum(tests) / len(tests)
        print('ave: {}'.format(ave), file=f)
        results['test_ave'] = ave


def get_progress_manager():
    width = 80
    prog = open(progf, 'w')
    return ProgressManager(target=prog, width=width)


def one_fold(t, i, results, logs):
    train_data = data['train'][i]
    test_data = data['test'][i]
    dev_data = data['dev'][i]
    train_len = len(train_data)
    test_len = len(test_data)
    dev_len = len(dev_data)

    if composition == 'SLSTM':
        rnn = SLSTM(vocab, n_units, mem_units, is_regression, is_leaf_as_chunk=mode == 'dependency')
    elif composition == 'TreeLSTM':
        rnn = TreeLSTM(vocab, n_units, mem_units, is_regression, is_leaf_as_chunk=mode == 'dependency')

    optimizer = opt()
    optimizer.setup(rnn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(l2))
    optimizer.add_hook(chainer.optimizer.GradientClipping(clip_grad))

    classifier = AttentionClassifier(mem_units, label_num, method=attention)
    optimizer2 = opt()
    optimizer2.setup(classifier)
    optimizer2.add_hook(chainer.optimizer.WeightDecay(l2))
    optimizer2.add_hook(chainer.optimizer.GradientClipping(clip_grad))

    flog = open(fold_dir + '/{}.txt'.format(i), 'w')
    # training
    best_root = -100
    print('start train')
    for epoch in t('Fold{}: Epoch'.format(i), maxv=n_epoch, order=1, nest=0)(range(n_epoch)):
        # shuufle data
        random.shuffle(train_data)

        accum_loss = 0
        total_loss = 0
        count = 0
        classifier.init_count()
        for line in t('Training', maxv=train_len, order=2, nest=1)(train_data):
            tree = Tree(line)
            attention_mems = None
            if attention is not None:
                attention_mems = list()
            vec, c, loss = rnn(tree, attention_mems, classifier)
            accum_loss += loss
            count += 1
            if batch_size <= count:
                count = 0
                rnn.zerograds()
                classifier.zerograds()
                accum_loss.backward()
                optimizer.update()
                optimizer2.update()
                total_loss += float(accum_loss.data)
                print('epoch{}, accum loss: {}, total loss: {}'.format(epoch, float(accum_loss.data), total_loss), flush=True, file=flog)
                accum_loss = 0
        if count != 0:
            rnn.zerograds()
            classifier.zerograds()
            accum_loss.backward()
            optimizer.update()
            optimizer2.update()
            total_loss += float(accum_loss.data)
            print('epoch{}, accum loss: {}, total loss: {}'.format(epoch, float(accum_loss.data), total_loss), flush=True, file=flog)
        acc = classifier.calc_acc() * 100
        logs.put('{}train ep{}: loss={:0>3}, acc={:0>2}'.format(i, epoch, total_loss, acc))
        classifier.init_count()

        # dev
        classifier.init_count()
        for line in t('Develop', maxv=dev_len, order=3, nest=1)(dev_data):
            tree = Tree(line)
            attention_mems = None
            if attention is not None:
                attention_mems = list()
            vec, c, loss = rnn(tree, attention_mems, classifier)
        root_acc = classifier.calc_acc()
        logs.put('{}dev ep{}: acc={:0>2}'.format(i, epoch, root_acc))
        if best_root < root_acc:
            best_ep = epoch
            best_root = root_acc
            best_rnn = copy.deepcopy(rnn)
            best_classifier = copy.deepcopy(classifier)
            pickle.dump((best_rnn, best_classifier), open(model_dir + '/{}.pkl'.format(i), 'wb'))
    flog.close()

    # test
    best_classifier.init_count()
    fw = open(ex_dir + '/test{}.txt'.format(i), 'w')
    print('best epoch: {}'.format(best_ep), file=fw)
    print('dev best acc: {:0>3}'.format(best_root), file=fw)
    results['dev{}'.format(i)] = best_root
    for testi, line in enumerate(t('Test', maxv=test_len, order=4, nest=1)(test_data)):
        tree = Tree(line)
        attention_mems = None
        if attention is not None:
            attention_mems = list()
        vec, c, loss = best_rnn(tree, attention_mems, best_classifier, True)
        tree.render_graph(graph_dirs[i], 'test{}'.format(testi), True)

        label = best_classifier.label
        dist = best_classifier.dist
        pred = best_classifier.pred
        res = 'True' if pred == label else 'False'
        print('result:{}, label:{}, dist:{}, {}'.format(res, label, dist, line[:-1]), file=fw)
    fw.close()
    acc = best_classifier.calc_acc()
    logs.put('{}test: acc={:0>2}'.format(i, acc))
    results['test{}'.format(i)] = best_classifier.calc_acc()
    logs.put(None)


# get args
args = ArgReader()
fold = args.get_fold()
mode = args.get_mode()
embedf = args.get_embedf()
data = args.get_data()
logf, progf, model_dir, ex_dir, fold_dir, graph_dirs = args.get_logfiles()
is_regression = args.is_regression()
attention = args.get_attention()
n_units = args.get_n_units()
mem_units = args.get_mem_units()
batch_size = args.get_batch_size()
n_epoch = args.get_n_epoch()
label_num = args.get_label_num()
learning_rate = args.get_lr()
l2 = args.get_l2()
clip_grad = args.get_clip_grad()
opt = args.get_optimizer()
composition = args.get_composition()
is_toy = args.is_toy()

vocab = Vocablary()
vocab.read_vocabl(data['vocab'])
if not is_toy:
    vocab.read_embed(embedf)


results = mp.Manager().dict()
logs = mp.Queue()


# exec
fold = 1
offset = 4
progress_m = get_progress_manager()
pts = [progress_m.new_tree(offset * i) for i in range(fold)]
processes = list()
progress_m.start()
for i in range(fold):
    p = mp.Process(target=one_fold, args=(pts[i], i, results, logs))
    processes.append(p)
    p.start()
log_writer(results)
for p in processes:
    p.join()
progress_m.finish()

args.output_result(test=results['test_ave'], dev=results['dev_ave'])
