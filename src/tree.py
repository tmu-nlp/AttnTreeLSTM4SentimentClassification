from nltk.tree import Tree as nlTree
from graphviz import Digraph


class Tree:
    def __init__(self, s_fomula=None, root=None, parent=None, nltree=None):
        self.data = dict()
        self.data['render_label'] = list()
        self.children = list()
        if s_fomula is not None:
            self.__nltree = nlTree.fromstring(s_fomula)
            self.__root = self
            if not self.is_leaf():
                for i in range(len(self.__nltree)):
                    self.children.append(Tree(root=self.__root, parent=self, nltree=self.__nltree[i]))
        else:
            self.__nltree = nltree
            self.__root = root
            self.parent = parent
            if not self.is_leaf():
                for i in range(len(self.__nltree)):
                    self.children.append(Tree(root=self.__root, parent=self, nltree=self.__nltree[i]))

    def render_graph(self, fdir, fname, get_label=lambda tree: tree.get_label(), get_fill_color=lambda tree: '#ffffff', get_color=lambda tree: '#000000', get_peripheries=lambda tree: '1'):
        def recursive(t, g):
            for c in t.children:
                label = get_label(c)
                g.node(str(id(c)), label, fontsize='20', style='filled', fillcolor=get_fill_color(c), color=get_color(c), peripheries=get_peripheries(c))
                g.edge(str(id(t)), str(id(c)), dir='back')
                recursive(c, g)

        # define graph
        g = Digraph(format='png', filename=fname, directory=fdir)
        g.attr('node', shape='box')
        g.attr('graph', randkdir='BT', concentrate='true', charset='UTF-8')

        # make graph
        label = get_label(self)
        g.node(str(id(self)), label, fontsize='20', style='filled', fillcolor=get_fill_color(self), color=get_color(self), peripheries=get_peripheries(self))
        g.node('sent', ' '.join(self.get_leaves()), fontsize='20')
        recursive(self, g)
        g.render(cleanup=True)

    def s_fomula(self):
        return ' '.join(tmp for tmp in ' '.join(
            self.__nltree.pformat().split('\n')).split() if tmp != '')

    def is_root(self):
        return self.__root == self

    def is_leaf(self):
        return sum(1 for _ in self.__nltree.subtrees()) == 1

    def get_label(self):
        return self.__nltree.label()

    def subtrees(self):
        l = list()
        l += self.children
        for c in l:
            l += c.children
            yield c

    def get_leaves(self):
        yield from self.__nltree.leaves()

    def get_word(self):
        assert self.is_leaf(), 'current is not leaf node'
        if len(self.__nltree) == 0:
            return '　'
        try:
            return self.__nltree[0]
        except IndexError:
            print(self.__nltree, self.__nltree.leaves(), file=open('error.log', 'a'))
