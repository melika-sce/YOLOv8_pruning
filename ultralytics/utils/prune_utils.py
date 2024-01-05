from ultralytics.nn.modules import Bottleneck, Detect


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