from copy import deepcopy


# obj det 全局设置
class MwGlobalExp:
    def __init__(self, core, device, mw_classes, model_path,
                 nms_thresh, conf_thresh, parent_obj=''):

        self.parent_cat = parent_obj
        self.is_cascaded_det = len(parent_obj) > 0
        self.input_size = (416, 416)

        self.mw_classes = mw_classes
        # create reverse map of cls -> category_id
        self.cls2id = {name: i + 1 for i, name in enumerate(self.mw_classes)}
        # define children objects if necessary
        if self.is_cascaded_det:
            self.children_cats = deepcopy(self.mw_classes)
            self.children_cats.remove(parent_obj)

        # define model file
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = len(mw_classes)
        self.core = core
        self.device = device

    def get_openvino_model(self):
        net = self.core.read_model(self.model_path)
        compiled_model = self.core.compile_model(model=net, device_name=self.device)
        input_name = compiled_model.inputs[0]
        output_name = compiled_model.outputs[0]

        return (input_name, output_name, (self.input_size[0], self.input_size[1]), compiled_model)
