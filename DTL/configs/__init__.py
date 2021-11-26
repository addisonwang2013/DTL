from .configurations import *
from .config_event import *
from .config_depth import *


config_factory = {'resnet_cityscapes': Config(),
        }

config_event = {'resnet_ddd17': Config(),
        }


configs_depth = {'deeplab_depth': Config_depth(),

                 }