from os.path import dirname, basename, isfile
import glob
from inspect import isabstract
from .base import _ModelBase
from inspect import getmembers
from importlib import import_module


package_directory = dirname(__file__)
module_root = basename(package_directory)
modules = glob.glob(package_directory + '/*.py')

__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]



"""
    Make a dictionary of all available models. Model objects must inherit from
    ModelBase in order to be added to this dictionary.

    { name : class }
"""
available_models = {}

for module_name in __all__:
    module = import_module('{}.{}'.format(module_root, module_name))
    for name, obj in getmembers(module):
        try:
            if (obj != _ModelBase) and issubclass(obj, _ModelBase) and not isabstract(obj):
                available_models[name.lower()] = obj
        except TypeError:
            pass


class UnknownModelException(Exception):
    pass


def get_model(name):
    """
    Returns a model object, derived from ModelBase
    :param name: string, model name, case insensitive
    :return: ModelBase descendant
    """
    try:
        obj = available_models[name.lower()]
        return obj()
    except KeyError:
        raise UnknownModelException('Unknown model {}. Available models: {}'
                                    .format(name, ', '.join(k for k in available_models.keys())))
