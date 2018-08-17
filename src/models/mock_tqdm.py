import sys


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)


class tqdm(object):

    def __init__(self, iterable=None, desc=None, **kwargs):
        self.iterable = iterable
        self.desc = desc

    def __enter__(self):
        print('Starting', self.desc)
        return self

    def __iter__(self):
        for item in self.iterable:
            yield item

    def __exit__(self, *exc):
        print('Completed', self.desc)
        return False

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if ordered_dict:
            print(ordered_dict)

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        fp = file if file is not None else sys.stdout
        # Write the message
        fp.write(s)
        fp.write(end)
