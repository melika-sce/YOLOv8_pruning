from ultralytics.nn.modules import Bottleneck, Detect
import torch
from torch import nn


def get_ignore_bn(model):
    ignore_bn_list = []
    for k, m in model.named_modules():
        if isinstance(m, Bottleneck):
            if m.add:
                ignore_bn_list.append(
                    k.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(k + '.cv1.bn')
                ignore_bn_list.append(k + '.cv2.bn')
        if isinstance(m, (Detect,)):
                ignore_bn_list.append(k+".cv2.0.0.bn")
                ignore_bn_list.append(k+".cv2.0.1.bn")
                ignore_bn_list.append(k+".cv2.1.0.bn")
                ignore_bn_list.append(k+".cv2.1.1.bn")
                ignore_bn_list.append(k+".cv2.2.0.bn")
                ignore_bn_list.append(k+".cv2.2.1.bn")

                ignore_bn_list.append(k+".cv3.0.0.bn")
                ignore_bn_list.append(k+".cv3.0.1.bn")
                ignore_bn_list.append(k+".cv3.1.0.bn")
                ignore_bn_list.append(k+".cv3.1.1.bn")
                ignore_bn_list.append(k+".cv3.2.0.bn")
                ignore_bn_list.append(k+".cv3.2.1.bn")

                ignore_bn_list.append(k+".cv4.0.0.bn")
                ignore_bn_list.append(k+".cv4.0.1.bn")
                ignore_bn_list.append(k+".cv4.1.0.bn")
                ignore_bn_list.append(k+".cv4.1.1.bn")
                ignore_bn_list.append(k+".cv4.2.0.bn")
                ignore_bn_list.append(k+".cv4.2.1.bn")
    return ignore_bn_list

def get_bn_weights(model, ignore_bn_list):
    module_list = []
    for j, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d) and j not in ignore_bn_list:
            bnw = layer.state_dict()['weight']
            module_list.append(bnw)

    size_list = [idx.data.shape[0] for idx in module_list]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)
                   ] = module_list[idx].data.abs().clone()
        index += size
    return bn_weights