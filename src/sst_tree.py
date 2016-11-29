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
from networks import TreeLSTM
from networks import TreeAttentionLSTM
import chainer


def get_progress_manager():
    width = 80
    prog = open(progf, 'w')
    return ProgressManager(target=prog, width=width)


def train():
    i = 0
    train_data = data['train']
    test_data = data['test']
    dev_data = data['dev']
    train_len = len(train_data)
    test_len = len(test_data)
    dev_len = len(dev_data)

    flog = open(fold_dir + '/{}.txt'.format(i), 'w')

    print('reading vocablary', file=flog, flush=True)
    vocab = Vocablary()
    vocab.read_vocab(data['vocab'])
    if not is_toy and use_embed:
        print('reading initial embedding', file=flog, flush=True)
        vocab.read_embed(embedf)
    print('vocab init fin', file=flog, flush=True)
    if is_always_attn:
        rnn = TreeAttentionLSTM(vocab, n_units, mem_units, attention_method, is_regression, forget_bias, is_leaf_as_chunk=mode == 'dependency')
    else:
        rnn = TreeLSTM(vocab, n_units, mem_units, composition, is_regression, forget_bias, is_leaf_as_chunk=mode == 'dependency')

    optimizer = opt()
    optimizer.setup(rnn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(l2))
    optimizer.add_hook(chainer.optimizer.GradientClipping(clip_grad))
    if is_always_attn:
        attn_method = None
    else:
        attn_method = attention_method
    classifier = AttentionClassifier(mem_units, label_num, attn_method, attention_target, dropout, is_regression, is_only_attn)
    optimizer2 = opt()
    optimizer2.setup(classifier)
    optimizer2.add_hook(chainer.optimizer.WeightDecay(l2))
    optimizer2.add_hook(chainer.optimizer.GradientClipping(clip_grad))

    # training
    best_root = -100
    print('start train', file=flog, flush=True)
    for epoch in t('Epoch', maxv=n_epoch, order=1, nest=0)(range(n_epoch)):
        # shuufle data
        random.shuffle(train_data)

        accum_loss = 0
        total_loss = 0
        count = 0
        classifier.init_count()
        for line in t('Training', maxv=train_len, order=2, nest=1)(train_data):
            tree = Tree(line)
            # compute and assign vector to each node of tree
            rnn(tree)
            # classify polarity for each node who has correct label
            accum_loss += classifier(tree)
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
        print('{}train ep{}: loss={:0>3}, acc={:0>2}'.format(i, epoch, total_loss, acc), flush=True, file=flog)
        classifier.init_count()

        # dev
        classifier.init_count()
        for line in t('Develop', maxv=dev_len, order=3, nest=1)(dev_data):
            tree = Tree(line)
            # compute and assign vector to each node of tree
            rnn(tree)
            # classify polarity for each node who has correct label
            loss = classifier(tree, is_train=False)
        root_acc = classifier.calc_acc()
        print('{}dev ep{}: acc={:0>2}'.format(i, epoch, root_acc), flush=True, file=flog)
        if best_root < root_acc:
            best_ep = epoch
            best_root = root_acc
            best_rnn = copy.deepcopy(rnn)
            best_classifier = copy.deepcopy(classifier)
            pickle.dump((best_rnn, best_classifier), open(model_dir + '/{}.pkl'.format(i), 'wb'))

    # test
    best_classifier.init_count()
    fw = open(ex_dir + '/test{}.txt'.format(i), 'w')
    print('best epoch: {}'.format(best_ep), file=fw)
    print('dev best acc: {:0>3}'.format(best_root), file=fw)
    dev_best = best_root
    for testi, line in enumerate(t('Test', maxv=test_len, order=4, nest=1)(test_data)):
        tree = Tree(line)
        # compute and assign vector to each node of tree
        rnn(tree)
        # classify polarity for each node who has correct label
        loss = best_classifier(tree, is_train=False)

        # logging
        if render_flag:
            def get_label(tree):
                label = list()
                if 'correct_pred' in tree.data:
                    label.append(tree.data['correct_pred'])
                if 'attention_weight' in tree.data:
                    label.append('{:.5f}'.format(float(tree.data['attention_weight'].data)))
                if tree.is_leaf():
                    label.append(tree.get_word())
                if 'dist' in tree.data:
                    n = tree.data['dist'][0][0]
                    p = tree.data['dist'][0][1]
                    label.append('P:{:.3f},N:{:.3f}'.format(p, n))
                return '\n'.join(label)
            def get_fill_color(tree):
                if 'top1' in tree.data:
                    return '#ff0000'
                if 'top2' in tree.data:
                    return '#ff3f3f'
                if 'top3' in tree.data:
                    return '#ff7e7e'
                if 'top4' in tree.data:
                    return '#ffbdbd'
                if 'top5' in tree.data:
                    return '#ffc8c8'
                return '#ffffff'
            def get_color(tree):
                if 'is_correct' in tree.data and tree.data['is_correct']:
                    return '#0033ff'
                if 'correct_pred' in tree.data:
                    return '#dd0000'
                return '#000000'
            def get_peripheries(tree):
                if 'correct_pred' in tree.data:
                    return '3'
                return '1'
            tree.render_graph(graph_dirs[i], 'test{}{}'.format(testi, tree.data['is_correct']), get_label, get_fill_color, get_color, get_peripheries)
        print('{}, {}, dist:{}, {}'.format(tree.data['is_correct'], tree.data['correct_pred'], tree.data['dist'], line[:-1]), file=fw)
    fw.close()
    acc = best_classifier.calc_acc()
    print('{}test: acc={:0>2}'.format(i, acc), flush=True, file=flog)
    test_best = best_classifier.calc_acc()
    flog.close()
    return dev_best, test_best


# get args
args = ArgReader()
#fold = args.get_fold()
mode = args.get_mode()
embedf = args.get_embedf()
data = args.get_data()
logf, progf, model_dir, ex_dir, fold_dir, graph_dirs = args.get_logfiles()
is_regression = args.is_regression()
attention_method = args.get_attention()
attention_target = args.get_attention_target()
n_units = args.get_n_units()
mem_units = args.get_mem_units()
batch_size = args.get_batch_size()
n_epoch = args.get_n_epoch()
label_num = args.get_label_num()
learning_rate = args.get_lr()
l2 = args.get_l2()
clip_grad = args.get_clip_grad()
dropout = args.get_dropout()
opt = args.get_optimizer()
composition = args.get_composition()
is_toy = args.is_toy()
render_flag = args.render_grpah()
use_embed = args.use_embed()
forget_bias = args.forget_bias()
is_only_attn = args.is_only_attn()
is_always_attn = args.is_always_attn()

# exec
progress_m = get_progress_manager()
t = progress_m.new_tree(0)
progress_m.start()
dev_best, test_best = train()
progress_m.finish()

args.output_result(test=test_best, dev=dev_best)
