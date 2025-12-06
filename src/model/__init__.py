from .mobilenet import MobileNetV1
from .resnet import *

def create_model(opt):
    if opt['model'] == 'mobilenet':
        return MobileNetV1(c_in=3, c_out=opt['out_nc'])
    elif opt['model'] == 'resnet18':
        return ResNet18(c_out=opt['out_nc'])
    elif opt['model'] == 'resnet34':
        return ResNet34(c_out=opt['out_nc'])
    else:
        raise NotImplementedError('Model [{:s}] is not recognized.'.format(opt['model']))