import os.path

from state_cls.classification import classification
from state_cls.classification_openvino import cls_openvino


from state_cls.mae_model import mae_model
from sys_config import config
from utils.img_util import Img_util

from utils.logger_config import get_logger

logger = get_logger(config.LOG_PATH)
class Clstor:

    def __init__(self, core, device, mae_models, mobilev2_models):
        self.core = core
        self.clstor = {}
        self.image_dir = {}
        self.classnames = {}

        for mae_item in mae_models:
            device = device.lower()
            self.classnames[mae_item] = mae_models[mae_item]['classname']
            self.clstor[mae_item] = mae_model(device, mae_models[mae_item]['path'], nb_classes=mae_models[mae_item]['nb'])
            self.image_dir[mae_item] = './cls_output/{}'.format(mae_item)


        for mobile_item in mobilev2_models:
            if len(mobilev2_models) == 1 and mobilev2_models[mobile_item]['mode_is_openvino']:
                logger.debug("************use openvino********************")
                self.classnames[mobile_item] = mobilev2_models[mobile_item]['classname']
                self.clstor[mobile_item] = cls_openvino(self.core, device=device, model_path=mobilev2_models[mobile_item]['openvino_path'],
                                                        nb=mobilev2_models[mobile_item]['nb'])
            else:
                logger.debug("************use pth********************")
                device = device.lower()
                self.clstor[mobile_item] = classification(device=device, model_path=mobilev2_models[mobile_item]['path'],
                                                          nb=mobilev2_models[mobile_item]['nb'])
            self.image_dir[mobile_item] = './cls_output/{}'.format(mobile_item)

        self.counter = 1

        self.img_util = Img_util()

        self.create_file_name()



    """
    构建存储图片文件夹
    """
    def create_file_name(self):
        for p in self.image_dir:
            if not os.path.exists(self.image_dir[p]):
                os.makedirs(self.image_dir[p])

    """
    获取预测结果
    """
    def get_component_pred(self, frame_top, frame_side, future_detector, component, clstor_res_value, frame_counter, args, view):
        pred = clstor_res_value
        if view == "top":
            frame = frame_top
        else:
            frame = frame_side


        if "-" in component:
            list_component = component.split("-")
            img_crop = self.img_util.get_multi_crop_img(frame, future_detector, list_component, view=view)
        else:
            img_crop = self.img_util.get_crop_img(frame, future_detector, component, view=view)
        if img_crop:

            if "." in component:
                component = component.split(".")[1]

            if args.mode_cls == "multi":
                result = self.clstor[component].inference_model(img_crop, component, self.classnames, args)
                pred = result["pred_class"]
                pre_dir = self.image_dir[component] + f"/{pred}"
            else:
                result = self.clstor[list(self.clstor.keys())[0]].inference_model(img_crop, component, self.classnames,args)
                pred = result["pred_class"]
                pre_dir = self.image_dir[list(self.image_dir.keys())[0]] + f"/{pred}"
            try:
                if not os.path.exists(pre_dir):
                    os.mkdir(pre_dir)
                img_crop.save(f"{pre_dir}/{frame_counter}_{result['pred_score']:.2f}.jpg")
            except:
                logger.error("error img save dir")

            self.counter += 1
        return pred




