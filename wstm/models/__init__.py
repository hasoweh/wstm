from .pcm import DeepLabModel, PCMBackDeepLab
from pydoc import locate
from .sem_cams import *
from .deeplab import *
from .resnet import *


def get_classification_model(modelname, classes, config, debug = False):
    model_dict = {'Resnet': create_resnet,
                  'Sem_resnet': create_sem_resnet,
                  'Pcm_deeplab': create_pcm_deeplab,
                  'Sem_deeplab': create_sem_deeplab
                 }
    model_builder = model_dict[modelname]
    
    return model_builder(classes, config, debug)

def load_pytorch_model(config):
    pretrained = False
    if 'pretrained' in config:
        pretrained = config['pretrained']
    
    # get the base resnet model
    if config['model'] is not None:
        model_base = locate(config['model'])(pretrained)
    return model_base


def create_resnet(classes, config, debug):
    
    unfreeze = None if 'unfreeze' not in config else config['unfreeze']
    
    return Resnet(load_pytorch_model(config), 
                  len(classes),
                  n_bands = len(config['means']),
                  p_dropout = config['prob_drop'],
                  unfreeze = unfreeze)

def create_sem_resnet(classes, config, debug):
    unfreeze = None if 'unfreeze' not in config else config['unfreeze']
    
    # use resnet as backbone
    backbone = Backbone(load_pytorch_model(config), 
                        len(classes),
                        n_bands = len(config['means']),
                        p_dropout = config['prob_drop'],
                        unfreeze = unfreeze)
    
    return SEM_Resnet(backbone, len(classes))

def create_pcm_deeplab(classes, config, debug):
    # use Deeplabv3+ as main architecture
    backbone = PCMBackDeepLab(config['model'], 
                              decoder_channels = 128,
                              decoder_atrous_rates = (12, 24, 36),
                              in_channels = len(config['means']),
                              classes = len(classes),
                              upsampling = 4,
                              p_dropout = config['prob_drop'])
    
    return DeepLabModel(backbone, len(classes), 128, debug = debug)

def create_sem_deeplab(classes, config, debug):
    # use Deeplabv3+ as main architecture
    backbone = DeepLabBackbone(config['model'], 
                               decoder_channels = 128,
                               decoder_atrous_rates = (12, 24, 36),
                               in_channels = len(config['means']),
                               classes = len(classes),
                               upsampling = 4,
                               p_dropout = config['prob_drop'])
    
    return SEM_DeepLab(backbone, len(classes), debug = debug)
