
import numpy as np

from utils.thread_argument import ThreadWithReturnValue
from .settings import MwGlobalExp
from .subdetectors import SubDetector


class Detector:
    def __init__(self,
        core,
        device,
        top_num_class: list,
        side_num_class: list,
        yaml_params):

        top_models = [yaml_params['object_model']['top']]
        side_models = [yaml_params['object_model']['side']]


        "配置2个模型在顶层"
        self.top_glb_exp = MwGlobalExp(
            core=core,
            device=device,
            mw_classes=top_num_class,
            model_path=top_models[0],
            conf_thresh=yaml_params['top_conf_thresh'],
            nms_thresh=yaml_params['top_nms_thresh'],
        )

        self.side_glb_exp = MwGlobalExp(
            core=core,
            device=device,
            mw_classes=side_num_class,
            model_path=side_models[0],
            conf_thresh=yaml_params['side_conf_thresh'],
            nms_thresh=yaml_params['side_nms_thresh'],
        )

        self.all_classes = list(self.top_glb_exp.mw_classes)
        self.all_classes += list(self.side_glb_exp.mw_classes)
        self.all_classes = sorted(list(set(self.all_classes)))

        ### load models for top view
        self.top_glb_subdetector = SubDetector(self.top_glb_exp, self.all_classes)
        ### load models for side view
        self.side_glb_subdetector = SubDetector(self.side_glb_exp, self.all_classes)


    def _get_parent_roi(self, preds, parent_id):
        for pred in preds:
            if parent_id == int(pred[-1]):
                res = pred[: 4]
                return res
        return None


    def _detect_one(self, img, view='top'):
        if view == 'top': # top view
            glb_subdet = self.top_glb_subdetector
        else: # side view
            glb_subdet = self.side_glb_subdetector

        # global detector inference
        preds, _ = glb_subdet.inference(img) # Nx7 (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        if preds is None or len(preds) == 0: return None, [], None
        all_preds = preds

        # cast class id integer
        for r, pred in enumerate(all_preds):
            all_preds[r, -1] = int(pred[-1])

        # remap to original image scale
        bboxes = all_preds[:, :4]
        cls = all_preds[:, 6]
        scores = all_preds[:, 4] * all_preds[:, 5]

        return bboxes, cls, scores

    def inference(self, img_top, img_side):
        """
        Given input arrays for two view, need to generate and save
            the corresponding detection results in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_front: img_array of H x W x C for the front view
        Returns:
        prediction results for the two images
        """

        ### sync mode ###
        top_bboxes, top_cls_ids, top_scores = self._detect_one(img_top, view='top')
        side_bboxes, side_cls_ids, side_scores = self._detect_one(img_side, view='side')
        side_bboxes_type = str(type(side_bboxes))
        top_bboxes_type = str(type(top_bboxes))
        if side_bboxes_type == "<class 'NoneType'>":
            side_bboxes = np.zeros((32,4))
            side_cls_ids = np.zeros((32,))
            side_scores = np.zeros((32,))
            side_labels = np.zeros((32,))
        else:
            # get class label
            side_labels = [self.all_classes[int(i) - 1] for i in side_cls_ids]

        if top_bboxes_type == "<class 'NoneType'>":
            top_bboxes = np.zeros((32, 4))
            top_cls_ids = np.zeros((32,))
            top_scores = np.zeros((32,))
            top_labels = np.zeros((32,))
        else:
            top_labels = [self.all_classes[int(i) - 1] for i in top_cls_ids]

        return [top_bboxes, top_cls_ids, top_labels, top_scores], [side_bboxes, side_cls_ids, side_labels, side_scores]

    def inference_multithread(self, img_top, img_side, frame_index):
        """
        Given input arrays for two view, need to generate and save the corresponding detection results
            in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_side: img_array of H x W x C for the side view
        Returns:
        prediction results for the two images
        """

        # creat detector thread and segmentor thread
        tdetTop = ThreadWithReturnValue(target = self._detect_one, args = (img_top, 'top',))
        tdetSide = ThreadWithReturnValue(target = self._detect_one, args = (img_side, 'side',))

        tdetTop.start()
        tdetSide.start()

        top_bboxes, top_cls_ids, top_scores = tdetTop.join()
        side_bboxes, side_cls_ids, side_scores = tdetSide.join()

        # get class label
        top_labels = None
        side_labels = None
        if top_cls_ids is not None and len(top_cls_ids) > 0:
            top_labels = [self.all_classes[int(i) - 1] for i in top_cls_ids]
        if side_cls_ids is not None and len(side_cls_ids) > 0:
            side_labels = [self.all_classes[int(i) - 1] for i in side_cls_ids]

        return [top_bboxes, top_cls_ids, top_labels, top_scores], \
                [side_bboxes, side_cls_ids, side_labels, side_scores], \
                frame_index