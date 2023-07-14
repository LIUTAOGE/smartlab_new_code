import re

import mmcv
from mmcv.runner import load_checkpoint
from mmcls.models import build_classifier
import warnings
import torch
import numpy as np
from torchvision import transforms
import cv2


def init_model(config, nb, checkpoint=None, device='cuda:0', options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
        config._cfg_dict['model']['head']['num_classes'] = nb
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class classification:

    def __init__(self, device, model_path, nb=None):
        self.model = init_model("state_cls/mobilenet-v2_8xb32_in1k.py", nb, model_path, device=device)
        self.model.eval()

        self.mean = np.asarray([123.675, 116.28, 103.53])
        self.std = np.asarray([58.395, 57.12, 57.375])

    def inference_model(self, img, component, classnames, args):
        # forward the model
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            cv2_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = mmcv.imnormalize(cv2_img, self.mean, self.std,True)
            img = trans(img)
            img = torch.unsqueeze(img, 0)

            data = {}
            data['img_metas'] = []
            img_metas = {}
            img_metas['filename'] = "./"
            img_metas['ori_filename'] = "./"
            img_metas['ori_shape'] = (224,224,3)
            img_metas['img_shape'] = (224, 224, 3)
            img_metas['img_norm_cfg'] = {}
            img_metas['img_norm_cfg']['mean'] = np.array([123.675, 116.28, 103.53])
            img_metas['img_norm_cfg']['std'] = np.array([58.395, 57.12, 57.375])
            data['img_metas'].append(img_metas)

            data['img'] = img

            scores = self.model(return_loss=False, **data)

            if args.mode_cls == "multi":
                pred_score = np.max(scores, axis=1)[0]
                pred_label = np.argmax(scores, axis=1)[0]
            else:
                new_score = np.array([])
                for index, model_class in enumerate(self.model.CLASSES):
                    if re.match(component+"#", model_class) != None:
                       new_score = np.append(new_score, scores[0][index])

                pred_score = max(new_score)
                pred_label = int(np.where(scores == pred_score)[1])

            result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        result['pred_class'] = self.model.CLASSES[result['pred_label']]
        return result

