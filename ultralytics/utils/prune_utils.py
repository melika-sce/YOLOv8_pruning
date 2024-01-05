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

def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask

def get_mask_bn(model, ignore_bn_list, thre_prune):
    remain_num = 0
    mask_bn = {}
    for bnname, bnlayer in model.named_modules():
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            mask = obtain_bn_mask(bn_module, thre_prune)
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size()).cuda()
            mask_bn[bnname] = mask
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            bn_module.bias.data.mul_(mask)
            print(f"|\t{bnname:<25}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
            assert int(mask.sum(
            )) > 0, "Current remaining channel must be greater than 0!!! please set prune percent to lower thresh, or you can retrain a more sparse model..."
    print("=" * 94)
    return model, mask_bn

def gather_bn_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights

def get_prune_threshold(model_list, percent):
    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]


    # Avoid pruning the highest threshold of all channels (the minimum value of 
    # the maximum gamma value of each BN layer is the upper limit of the threshold)
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())

    highest_thre = min(highest_thre)
    # Find the percentage corresponding to the subscript corresponding to highest_thre
    threshold_index = (sorted_bn == highest_thre).nonzero().squeeze()
    if len(threshold_index.shape) > 0:
        threshold_index = threshold_index[0]
    percent_threshold = threshold_index.item() / len(bn_weights)
    print('Suggested Gamma threshold should be less than {}'.format(highest_thre))
    print('The corresponding prune ratio is {}, but you can set higher'.format(percent_threshold))
    thre_index = int(len(sorted_bn) * percent)
    thre_prune = sorted_bn[thre_index]
    print('Gamma value that less than {} are set to zero'.format(thre_prune))
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    return thre_prune

def get_bn_list(model):
    model_list = {}
    ignore_bn_list = []

    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:
                ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
        if isinstance(layer, Detect):
                ignore_bn_list.append(i+".cv2.0.0.bn")
                ignore_bn_list.append(i+".cv2.0.1.bn")
                ignore_bn_list.append(i+".cv2.1.0.bn")
                ignore_bn_list.append(i+".cv2.1.1.bn")
                ignore_bn_list.append(i+".cv2.2.0.bn")
                ignore_bn_list.append(i+".cv2.2.1.bn")

                ignore_bn_list.append(i+".cv3.0.0.bn")
                ignore_bn_list.append(i+".cv3.0.1.bn")
                ignore_bn_list.append(i+".cv3.1.0.bn")
                ignore_bn_list.append(i+".cv3.1.1.bn")
                ignore_bn_list.append(i+".cv3.2.0.bn")
                ignore_bn_list.append(i+".cv3.2.1.bn")

                ignore_bn_list.append(i+".cv4.0.0.bn")
                ignore_bn_list.append(i+".cv4.0.1.bn")
                ignore_bn_list.append(i+".cv4.1.0.bn")
                ignore_bn_list.append(i+".cv4.1.1.bn")
                ignore_bn_list.append(i+".cv4.2.0.bn")
                ignore_bn_list.append(i+".cv4.2.1.bn")
        if isinstance(layer, torch.nn.BatchNorm2d):
            model_list[i] = layer

    model_list = {k: v for k, v in model_list.items() if k not in ignore_bn_list}

    return model_list, ignore_bn_list