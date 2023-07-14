import re

import mmcv
import torch
import numpy as np
from torchvision import transforms
import cv2

class cls_openvino:
    def __init__(self, core,  device, model_path, nb=None):
        self.core = core
        self.model_path = model_path
        self.device = device
        self.input_size = (224, 224)
        self.inode, self.onode, self.input_shape, self.model = self.get_openvino_model()

        self.mean = np.asarray([123.675, 116.28, 103.53])
        self.std = np.asarray([58.395, 57.12, 57.375])
        self.infer_request = self.model.create_infer_request()

    def get_openvino_model(self):
        net = self.core.read_model(self.model_path)
        compiled_model = self.core.compile_model(model=net, device_name=self.device)
        input_name = compiled_model.inputs[0]
        output_name = compiled_model.outputs[0]

        return (input_name, output_name, (self.input_size[0], self.input_size[1]), compiled_model)

    def inference_model(self, img, component, classnames, args):
        # forward the model
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            cv2_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = mmcv.imnormalize(cv2_img, self.mean, self.std,True)
            img = trans(img)
            img = torch.unsqueeze(img, 0)

            data = {}
            data['img'] = img
            res_numpy = self.infer_request.infer({self.inode: np.expand_dims(data['img'].squeeze(), axis=0)})[self.onode]

            if args.mode_cls == "multi":
                pred_score = np.max(res_numpy, axis=1)[0]
                pred_label = np.argmax(res_numpy, axis=1)[0]
            else:
                new_score = np.array([])
                classnames = classnames[list(classnames.keys())[0]]
                for index, model_class in enumerate(classnames):
                    if re.match(component+"#", model_class) != None:
                       new_score = np.append(new_score, res_numpy[0][index])

                pred_score = max(new_score)
                pred_label = int(np.where(res_numpy == pred_score)[1])

            result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        result['pred_class'] = classnames[result['pred_label']]
        return result