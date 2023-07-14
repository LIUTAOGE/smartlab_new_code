import cv2
from PIL import Image

class Img_util():
    def get_crop_img(self, frame, future_detector, obj_c, view):
        if view == "side":
            detector_res = future_detector[1]
        else:
            detector_res = future_detector[0]
        obj_idx = detector_res[2].index(obj_c) if obj_c in detector_res[2] else -1
        if obj_idx != -1:
            object_bbox = detector_res[0][obj_idx]
            x_1, y_1, x_2, y_2 = object_bbox[0], object_bbox[1], \
                                 object_bbox[2], object_bbox[3]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 测试使用，真实去掉
            if obj_c in ["power", "switch", "bulb", "ammeter", "bulb_module"]:
                x_min = x_1 - 30
                y_min = y_1 - 30
                x_max = x_2 + 30
                y_max = y_2 + 30
            else:
                x_min = x_1
                y_min = y_1
                x_max = x_2
                y_max = y_2

            img_crop = img.crop((x_min, y_min, x_max, y_max))
            img_crop = self.resize(img_crop, 224, 224)
            return img_crop


    def get_multi_crop_img(self, frame, future_detector, obj_c_list, view='top'):
        if view == "side":
            detector_res = future_detector[1]
        else:
            detector_res = future_detector[0]

        obj_all_bboxes = []
        for c in obj_c_list:
            obj_idxs = [idx for idx in range(len(detector_res[2])) if detector_res[2][idx] == c]
            if obj_idxs:
                obj_bboxes = [detector_res[0][idx] for idx in obj_idxs]
                obj_all_bboxes += obj_bboxes
            else:
                return

        x_1, y_1, x_2, y_2 = [], [], [], []
        for x1, y1, x2, y2 in obj_all_bboxes:
            x_1.append(x1)
            y_1.append(y1)
            x_2.append(x2)
            y_2.append(y2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        x_min = min(x_1)
        y_min = min(y_1)
        x_max = max(x_2)
        y_max = max(y_2)

        img_crop = img.crop((x_min, y_min, x_max, y_max))
        img_crop = self.resize(img_crop, 224, 224)

        return img_crop


    def resize(self, image_pil, width, height):
        '''
        Resize PIL image keeping ratio and using white background.
        '''
        ratio_w = width / image_pil.width
        ratio_h = height / image_pil.height
        if ratio_w < ratio_h:
            # It must be fixed by width
            resize_width = width
            resize_height = round(ratio_w * image_pil.height)
        else:
            # Fixed by height
            resize_width = round(ratio_h * image_pil.width)
            resize_height = height
        image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
        background.paste(image_resize, offset)
        return background.convert('RGB')


    def is_inside(self, small_item_center_coor, big_item_coor):
        if small_item_center_coor[0] >= big_item_coor[0] and small_item_center_coor[0] <= big_item_coor[2] \
                and small_item_center_coor[1] > big_item_coor[1] and small_item_center_coor[1] < big_item_coor[3]:
            return True
        else:
            return False