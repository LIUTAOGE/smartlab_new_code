import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import cv2
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torchvision.transforms import transforms



from state_cls import models_vit
import torch.nn.functional as F
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class mae_model:
    def __init__(self, device, s_class, nb_classes=2):
        model_temple = "vit_base_patch16"
        drop_path = 0.1
        global_pool = True


        self.model_mae = models_vit.__dict__[model_temple](num_classes=nb_classes, drop_path_rate=drop_path, global_pool=global_pool)
        checkpoint = torch.load(s_class, map_location='cpu')
        checkpoint_model = checkpoint['model']
        self.model_mae.load_state_dict(checkpoint_model, strict=False)
        self.model_mae.to(device)
        self.model_mae.eval()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])




    def inference_model(self, img, component, classnames, args):
        with torch.no_grad():

            cv2_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            # img = cv2.resize(cv2_img, (224, 224))
            img = self.trans(cv2_img)
            img = torch.unsqueeze(img, 0)

            output = self.model_mae(img)
            scores = F.softmax(output, dim=1)
            pred_score = np.max(list(scores.detach().numpy()), axis=1)[0]
            pred_label = np.argmax(list(scores.detach().numpy()), axis=1)[0]
            result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        result['pred_class'] = classnames[component][result['pred_label']]
        return result


