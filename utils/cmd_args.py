"""
Command-line argument wrapper.

Singleton implementation. Each class declare a subset of flags. Flags all need 
to be declared before calling "make".

Use "get" for all external usage. Do not call CmdArgs class constructor unless
absolutely necessary.
"""

import argparse

cmd_args = None


def init(description):
    """Get a singleton instance of CmdArgs."""
    global cmd_args
    if cmd_args is None:
        cmd_args = CmdArgs(description)
        pass
    return cmd_args


def add(name, dtype='str', default=None, help_str=None):
    """Add an argument."""
    global cmd_args
    if cmd_args is None:
        cmd_args = init(None)
    cmd_args.add(name, dtype=dtype, default=default, help_str=help_str)
    pass


def make():
    """Make command-line options."""
    return cmd_args.make()


class Arg(object):
    """A data struct for storing command line argument details."""

    def __init__(self, name, dtype='str', default=None, help_str=None):
        self.name = name
        self.default = default
        self.dtype = dtype
        self.help_str = help_str
        if dtype == 'str':
            self.stype = str
        elif dtype == 'int':
            self.stype = int
        elif dtype == 'float':
            self.stype = float
        elif dtype == 'bool':
            self.stype = bool
            self.default = False
        elif dtype == 'list<int>' or dtype == 'list<float>':
            self.stype = str
        else:
            raise Exception('Unknown dtype: {}'.format(dtype))
        pass

    def convert(self, value):
        """Convert raw values into usable dynamic data structures."""
        if self.dtype == 'list<int>':
            return [int(v) for v in value.split(',')]
        elif self.dtype == 'list<float>':
            return [float(v) for v in value.split(',')]
        else:
            if value is not None:
                return self.stype(value)
            else:
                return None

    def register(self, args_parser):
        """Register a command-line argument."""
        if self.dtype == 'bool':
            args_parser.add_argument('--{}'.format(self.name),
                                     action='store_true', help=self.help_str)
        else:
            args_parser.add_argument(
                '--{}'.format(self.name), default=self.default,
                type=self.stype, help=self.help_str)


class CmdArgs(object):
    """Wrapper class for ArgumentParser."""

    def __init__(self, description=None):
        self.args = {}
        self.args_parser = argparse.ArgumentParser(description=description)
        self.converted = None
        pass

    def add(self, name, dtype='str', default=None, help_str=None):
        """Add an argument."""
        if dtype == 'list<int>' or dtype == 'list<float>':
            default = ','.join([str(d) for d in default])
            pass
        arg = Arg(name, dtype=dtype, default=default, help_str=help_str)
        if name not in self.args:
            self.args[name] = arg
            arg.register(self.args_parser)
            pass
        pass

    def make(self):
        """Parse all arguments."""
        converted = {}
        if self.converted is None:
            parsed = self.args_parser.parse_args()
            for arg in self.args.itervalues():
                name = arg.name
                converted[name] = arg.convert(parsed.__getattribute__(name))

        return converted
