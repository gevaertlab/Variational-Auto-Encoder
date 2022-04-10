from typing import List, Union
import torch.nn as nn
import torch
from .models.resnet import resnet10


class MedicalNet(nn.Module):

    def __init__(self,
                 path_to_weights,
                 input_D=None,  # NOTE: these are used eventually
                 input_H=None,
                 input_W=None,
                 num_seg_classes=2,
                 fea_layers: List = ["layer4"],
                #  device='0',
                 **kwargs):
        super(MedicalNet, self).__init__()
        self.model = resnet10(sample_input_D=input_D,
                              sample_input_H=input_H,
                              sample_input_W=input_W,
                              num_seg_classes=num_seg_classes)
        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.1)
        )
        self.fea_layers = fea_layers
        net_dict = self.model.state_dict()
        pretrained_weights = torch.load(path_to_weights) # , map_location=torch.device(device)
        pretrain_dict = {
            k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
        }
        net_dict.update(pretrain_dict)
        self.model.load_state_dict(net_dict)

        # output features
        self.register_hook(fea_layers)

        pass

    def forward(self, x):
        features = self.model(x)
        return features

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output
        return hook

    def register_hook(self, fea_layers: List[str]):
        self._features = {}
        for name, layer in self.model.named_children():
            layer.__name__ = name
            self._features[name] = torch.empty(0)
            if name in fea_layers:
                layer.register_forward_hook(self.forward_hook(name))
        pass
