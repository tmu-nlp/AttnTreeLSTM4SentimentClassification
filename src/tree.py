from nltk.tree import Tree as nlTree
from graphviz import Digraph


class Tree:
    count = 0

    def __init__(self, s_fomula=None, root=None, parent=None, nltree=None, i=None):
        self.i = Tree.count
        Tree.count += 1
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
            self.__parent = parent
            if not self.is_leaf():
                for i in range(len(self.__nltree)):
                    self.children.append(Tree(root=self.__root, parent=self, nltree=self.__nltree[i]))

    def render_graph(self, fdir, fname, with_data=False):
        def recursive(t, g):
            for c in t.children:
                additional = ''
                if with_data:
                    additional = '|'.join(c.data['render_label'])
                if c.is_leaf():
                    g.node(str(c.i), c.get_word() + additional, fontsize='20')
                else:
                    g.node(str(c.i), additional)
                g.edge(str(t.i), str(c.i), dir='back')
                recursive(c, g)

        g = Digraph(format='png', filename=fname, directory=fdir)
        g.attr('node', shape='box')
        g.attr('graph', randkdir='BT', concentrate='true', charset='UTF-8')
        additional = ''
        if with_data:
            additional = '|'.join(self.data['render_label'])
        g.node(str(self.i), additional)
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
