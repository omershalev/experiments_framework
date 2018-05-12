import logging
import inspect
import os

logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

def get_logger(name=None, level=None):
    if name is None:
        mod = inspect.getmodule(inspect.stack()[1][0])
        name = mod.__file__.split(os.path.sep)[-1].rstrip('.py') if mod.__name__ == '__main__' else mod.__name__
    if level is None:
        level = logging.DEBUG
    logging.getLogger(name).setLevel(logging.getLevelName(level))
    return logging.getLogger(name)
