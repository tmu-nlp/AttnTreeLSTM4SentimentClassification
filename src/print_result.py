import sys
import pickle
from prettytable import PrettyTable

dirs = sys.argv[1:]
results = dict()
for dir in dirs:
    name = dir + '/result.pkl'
    d = pickle.load(open(name, 'rb'))
    dev = d['dev']
    test = d['test']
    opt = d['optimizer']
    att = d['attention']
    lr = d['lr']
    comp = d['composition']
    pold = d['pol_dict']
    dropout = None
    if 'dropout' in d:
        dropout = d['dropout']
    att_tgt = None
    if 'attention_target' in d:
        att_tgt = d['attention_target']
    results[name] = (test, dev, opt, lr, att, att_tgt, pold, dropout, comp)
keys = list(d.keys())
print('keys')
print(keys)
print()
print('sort key = dev')
table = PrettyTable(['test', 'dev', 'optimizer', 'lr', 'attention', 'attention_target', 'pol_dict', 'dropout', 'composition', 'name'])
table.align['optimizer'] = 'l'
table.align['lr'] = 'l'
table.align['attention'] = 'l'
table.align['attention_target'] = 'l'
table.align['pol_dict'] = 'l'
table.align['dropout'] = 'l'
table.align['composition'] = 'l'
table.align['name'] = 'l'
for name, (test, dev, opt, lr, att, att_tgt, pold, drop, comp) in sorted(results.items(), key=lambda x: -x[1][1]):
    dev = '{:.5f}'.format(dev)
    test = '{:.5f}'.format(test)
    table.add_row([test, dev, opt, lr, att, att_tgt, pold, drop, comp, name])
print(table)
