from .mobilenet import MobileNetV1
from .resnet import ResNet50

def create_model(opt):
    if opt['model'] == 'mobilenet':
        return MobileNetV1(opt['out_nc'])
    elif opt['model'] == 'resnet':
        return ResNet50(opt['out_nc'])
    else:
        raise NotImplementedError('Model [{:s}] is not recognized.'.format(opt['model']))