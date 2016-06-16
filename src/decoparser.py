import os
import sys
import argparse
from argparse import ArgumentTypeError
from enum import Enum as origEnum


class FileType:
    def __init__(self, mode='r', encoding=None):
        self.mode = mode
        self.encoding = encoding

    def __call__(self, arg):
        self.name = arg
        return self


class FilePath:
    def __init__(self, exists=False, absolute=False):
        self.exists = exists
        self.absolute = absolute

    def __call__(self, arg):
        if self.exists and not os.path.exists(arg):
            errmsg = 'No such file or directory'
            message = "can't open '{}': {}".format(arg, errmsg)
            raise ArgumentTypeError(message)
        if self.absolute:
            return os.path.abspath(arg)
        return arg

    def __repr__(self):
        return 'hoge'


class Enum:
    pass


class Cmd:
    parser = argparse.ArgumentParser()
    first_call = True
    args = None
    pre_args = list()
    pre_options = list()

    @classmethod
    def get_args(cls):
        if Cmd.first_call:
            for args, kwargs in Cmd.pre_args + Cmd.pre_options:
                try:
                    Cmd.parser.add_argument(*args, **kwargs)
                except argparse.ArgumentError as e:
                    print(e.__class__.__name__, file=sys.stderr)
                    print('message:\n {0}'.format(e.message), file=sys.stderr)
                    print(' there are many same options', file=sys.stderr)
                    exit()
            Cmd.args = Cmd.parser.parse_args()
            Cmd.first_call = False

    def __init__(self, f, name, name2, action, nargs, const, default, type,
                 choices, required, help, metavar, dest):
        self.f = f
        self.name = name
        self.name2 = name2
        self.action = action
        self.nargs = nargs
        self.const = const
        self.default = default
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        self.metavar = metavar
        self.dest = dest

        if type is FileType:
            self.type = None

    def add_option(self):
        args = [self.name]
        kwargs = {'default': self.default, 'help': self.help}
        if self.name2 is not None:
            args.append(self.name2)
        for kw in ['action', 'nargs', 'const', 'type', 'choices', 'required',
                   'metavar', 'dest']:
            slf = eval('self.{}'.format(kw))
            if slf is not None:
                kwargs.update({kw: slf})
        if self.type is Enum:
            if not issubclass(self.choices, origEnum):
                assert False, 'if type is decoparser.Enum, choices should be enum.Enum type'
            del kwargs['type']
            kwargs['choices'] = list(self.choices.__members__)
            if self.default is not None:
                if not isinstance(self.default, self.choices):
                    assert False, 'default value should be a value of Enum'
                kwargs['default'] = self.default.name

        Cmd.pre_options.append((args, kwargs))

    def add_argument(self):
        args = [self.name]
        kwargs = {'metavar': self.name.upper(), 'help': self.help}
        if self.default is not None:
            kwargs.update({'default': self.default, 'nargs': '?'})
        for kw in ['action', 'nargs', 'const', 'type', 'choices', 'required',
                   'metavar', 'dest']:
            slf = eval('self.{}'.format(kw))
            if slf is not None:
                kwargs.update({kw: slf})
        Cmd.pre_args.append((args, kwargs))

    def __call__(self, *args, **kwargs):
        Cmd.get_args()
        kname = self.name[2:] if self.name.startswith('--') else self.name
        kname = kname.replace('-', '_')
        arg = Cmd.args.__dict__[kname]
        if self.type is Enum:
            arg = self.choices[arg]
        # file type
        if isinstance(arg, FileType):
            with open(arg.name, arg.mode, encoding=arg.encoding) as f:
                kwargs[kname] = f
                return self.f(*args, **kwargs)
        else:
            kwargs[kname] = arg
            return self.f(*args, **kwargs)


def option(name, name2=None, action=None, nargs=None, const=None, default=None,
           type=None, choices=None, required=None, help=None, metavar=None,
           dest=None):
    def iner(f):
        cmd = Cmd(f, name, name2, action, nargs, const, default, type, choices,
                  required, help, metavar, dest)
        cmd.add_option()
        return cmd
    return iner


def argument(name, action=None, nargs=None, const=None, default=None,
             type=None, choices=None, required=None, help=None, metavar=None,
             dest=None):
    def iner(f):
        cmd = Cmd(f, name, None, action, nargs, const, default, type, choices,
                  required, help, metavar, dest)
        cmd.add_argument()
        return cmd
    return iner


def add_description(massage):
    Cmd.parser.description = massage


def add_version(version):
    Cmd.pre_args.insert(0, (['--version'],
                            {'action': 'version', 'version': version}))
