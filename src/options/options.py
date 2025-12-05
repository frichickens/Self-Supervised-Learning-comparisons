import os.path as osp
import yaml
from utils.utils import OrderedYaml, get_timestamp
Loader, Dumper = OrderedYaml()


def parse(opt_path, root):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt