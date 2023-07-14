import re

import numpy as np

from sys_config import config
from utils.exception_utils import DataFormatException
from utils.logger_config import get_logger

logger = get_logger(config.LOG_PATH)

class Evaluator(object):

    "初始化打分变量"
    def __init__(self, yaml):

        self.init_yaml(yaml)



    def init_yaml(self, yaml):
        yaml_def = yaml['yaml_def']
        yaml_ruler = yaml['yaml_rule']
        self.yaml_params = yaml['yaml_params']

        self.action_segment = {} # action片段
        self.keyname_action = {} # keyname和action映射

        self.yaml_keyframe = yaml_ruler['exp_rules']['keyframe']
        self.yaml_action_class = self.yaml_params['action']['class']
        self.yaml_action_tran = self.yaml_params['action']['transition']
        self.yaml_action_key_frame = self.yaml_params['action']['key_frame']


        self.exp_class = yaml_def['exp_class']

        self.exp_device = list(self.yaml_params['exp_filter'])
        self.exp_range_state = self.yaml_params['exp_range_state']

        self.get_device_relation()
        self.get_componenet_num_state()
        self.get_keyframe_init()
        self.get_rule_action_init()

        self.buffer_size = 0
        self.frame_counter = 0
        self.state = self.yaml_action_class[0]

    """
    抽出设备关系, exp_def -->exp_class, scale-zero
    """
    def get_device_relation(self):
        self.exp_class_relation = {}
        for exp_class_item in self.exp_class:

            exp_class_item_name = exp_class_item.split("#")[0]
            exp_class_item_state = exp_class_item.split("#")[1]

            if exp_class_item_name in self.exp_class_relation:
                self.exp_class_relation[exp_class_item_name] = self.exp_class_relation[exp_class_item_name] + [
                    exp_class_item_state]
            else:
                self.exp_class_relation[exp_class_item_name] = [exp_class_item_state]



    """
    根据模型预测结果，记录组件状态
    """
    def get_componenet_num_state(self):
        self.component_num_state = {}
        for exp in self.exp_class:
            self.component_num_state[exp] = {"num":0, "state":False}


    """
    关键帧初始化
    """
    def get_keyframe_init(self):
        self.keyframe_dict = {}
        for item_keyframe in self.yaml_keyframe:
            score_init = {"scoring":0, "clip_start":-1, "clip_end":-1, "clip_keyframe":[]}
            score_init['state'] = self.yaml_action_key_frame[item_keyframe]
            score_init['start'] = self.yaml_keyframe[item_keyframe]['start']
            score_init['end'] = self.yaml_keyframe[item_keyframe]['end']

            start_condition = self.sub_relation_extract(score_init['start'])
            score_init['start_condition'] = start_condition
            end_condition = self.sub_relation_extract(score_init['end'])
            score_init['end_condition'] = end_condition

            self.keyframe_dict[self.yaml_keyframe[item_keyframe]['name']] = score_init

            self.keyname_action[self.yaml_keyframe[item_keyframe]['name']] = self.yaml_action_key_frame[item_keyframe]

    """
    规则rule信息初始化, action
    """
    def get_rule_action_init(self):
        self.action_dict = {}
        for item_action in self.yaml_action_tran:
            action_init = {}
            action_init['o_state'] = self.yaml_action_tran[item_action]['o_state']
            action_init['e_state'] = self.yaml_action_tran[item_action]['e_state']
            action_init['trigger'] = self.yaml_action_tran[item_action]['trigger']
            trigger_condition = self.sub_relation_extract(action_init['trigger'])
            action_init['trigger_condition'] = trigger_condition
            action_init['name'] = self.yaml_action_tran[item_action]['name']

            self.action_dict[action_init['name']] = action_init




    def inference(self,
                  top_det_results,
                  side_det_results,
                  clstor_res_dict,
                  frame_top,
                  frame_side,
                  frame_counter,
                  frames_num,
                  yaml_params,
                  args,
                  clstor,
                  top_exp_angle,
                  side_exp_angle):


        self.frames_num = frames_num
        self.frame_counter = frame_counter
        self.yaml_params = yaml_params


        self.clstor_res_dict = clstor_res_dict

        top_det_results, side_det_results = self.filter_object(top_det_results, side_det_results)
        future_detector = []
        future_detector.append(top_det_results)
        future_detector.append(side_det_results)
        future_detector = tuple(future_detector)

        if frame_counter % 5 == 0:
            for t_e_d in top_exp_angle:
                self.clstor_res_dict[t_e_d] = clstor.get_component_pred(frame_top, frame_side, future_detector, t_e_d, self.clstor_res_dict[t_e_d], frame_counter, args, view="top")
            for s_e_d in side_exp_angle:
                self.clstor_res_dict[s_e_d] = clstor.get_component_pred(frame_top, frame_side, future_detector, s_e_d, self.clstor_res_dict[s_e_d], frame_counter, args, view="side")

        if frame_counter > self.buffer_size:

            self.classify_state()

            self.sub_evaluate(args)

            for score_item in self.keyframe_dict:
                state = self.keyframe_dict[score_item]['state']
                start_condition = self.keyframe_dict[score_item]['start_condition']
                end_condition = self.keyframe_dict[score_item]['end_condition']

                if "-" in state:
                    state_list = state.split("-")
                else:
                    state_list = [state]
                self.key_frame_extract(frame_counter, score_item, state_list, start_condition, end_condition)
            self.key_frame_end_score_tidy()

        display_frame_counter = frame_counter - self.buffer_size



        return self.keyframe_dict, self.state, top_det_results, side_det_results, frame_top, frame_side, display_frame_counter,\
               self.keyname_action, self.action_segment, self.frames_num


    """
    对关键帧条件组合
    """
    def sub_relation_extract(self, end):
        i = 0
        condition_list = []
        if end != 'None':
            if type(end).__name__ != "dict":
                condition_list.append(end)
            else:
                for e_c in end:
                    i += 1
                    if end[e_c] != 'None' and i > 1:
                        condition_list.append("#")
                        # condition_list.append("or") # 并列条件默认是or关系
                    if end[e_c] != 'None':
                        for e_sub_c in end[e_c]:
                            condition_list.append(e_sub_c)
                condition_list.append("#")
        return condition_list


    """
    抽取关键帧
    """
    def key_frame_extract(self, frame_counter, name, state_list, start_condition, end_condition):
        if self.state not in state_list:
            return

        # 在两个阶段都存在，
        if len(state_list) >= 2 and len(self.keyframe_dict[name]['clip_keyframe']) >= self.yaml_params["two_phase_max_num"]:
            return

        if name == list(self.keyframe_dict.keys())[len(self.keyframe_dict) - 1]:
            return

        if len(end_condition) == 0 and len(start_condition) > 0:
            if self.keyframe_dict[name]['clip_start'] == -1:
                res_start = self.get_rule_result(start_condition)

                if res_start and frame_counter > self.keyframe_dict[name]['clip_end'] + self.yaml_params['frame_range']:
                    self.keyframe_dict[name]['clip_start'] = frame_counter - self.yaml_params['frame_range']
                    self.keyframe_dict[name]['clip_end'] = frame_counter + self.yaml_params['frame_range']
                    self.keyframe_dict[name]['clip_keyframe'].append((self.keyframe_dict[name]['clip_start'], self.keyframe_dict[name]['clip_end']))
                    self.keyframe_dict[name]['clip_start'] = -1

        elif len(end_condition) > 0 and len(start_condition) > 0:
            if self.keyframe_dict[name]['clip_start'] == -1:
                res_start = self.get_rule_result(start_condition)

                if res_start and frame_counter > self.keyframe_dict[name]['clip_end'] + self.yaml_params['frame_range']:
                    self.keyframe_dict[name]['clip_start'] = frame_counter - self.yaml_params['frame_range']

            else:
                res_end = self.get_rule_result(end_condition)
                if res_end:
                    self.keyframe_dict[name]['clip_end'] = frame_counter + self.yaml_params['frame_range']
                    self.keyframe_dict[name]['clip_keyframe'].append((self.keyframe_dict[name]['clip_start'], self.keyframe_dict[name]['clip_end']))
                    self.keyframe_dict[name]['clip_start'] = -1
        else:
            logger.error("format error")
            raise DataFormatException("format error")


    """
    得到规则结果
    """
    def get_rule_result(self, start_condition):
        state_list = []
        for start_con_item in start_condition:
            if start_con_item not in ["and", "or", "#"]:
                if start_con_item in self.component_num_state.keys():
                    component_state = self.component_num_state[start_con_item]['state']
                    state_list.append(component_state)
            else:
                state_list.append(start_con_item)

        result = False
        index_result = []
        state_list_all = []
        state_list_new = []
        # 将 condition_1 和 condition_2 分开并列，默认为or
        for idx, state_item in enumerate(state_list):
            if state_item == "#":
                state_list_all.append(state_list_new)
                state_list_new = []
            else:
                state_list_new.append(state_item)

        for state_list_item in state_list_all:
            if len(state_list_item) == 1:
                result = state_list_item[0]
            elif len(state_list_item) >= 2:
                if state_list_item[1] == "and":
                    result = state_list_item[0] and state_list_item[2]
                else:
                    result = state_list_item[0] or state_list_item[2]
                for idx, state_item in enumerate(state_list_item):
                    if idx > 2 and (idx % 2) != 0:
                        if state_item == "and":
                            result = result and state_list_item[idx + 1]
                        else:
                            result = result or state_list_item[idx + 1]
            index_result.append(result)

        # 将并列条件进行组合
        for idx, in_res in enumerate(index_result):
            if idx == 0:
                result = in_res
            elif idx >= 1:
                result = result or in_res
            else:
                logger.error("get_rule_result error")
        return result

    """
    关键帧--end_score_tidy  最后一帧
    """
    def key_frame_end_score_tidy(self):
        last_name = list(self.keyframe_dict.keys())[len(self.keyframe_dict) -1]
        if self.keyframe_dict[last_name]['clip_start'] == -1:
            self.keyframe_dict[last_name]['clip_start'] = self.frames_num
            self.keyframe_dict[last_name]['clip_end'] = self.frames_num
            self.keyframe_dict[last_name]['clip_keyframe'].append((
                self.keyframe_dict[last_name]['clip_start'],
                self.keyframe_dict[last_name]['clip_end']))

    """
    切换状态
    """
    def classify_state(self):
        for action_item in self.action_dict:
            o_state = self.action_dict[action_item]['o_state']
            e_state = self.action_dict[action_item]['e_state']
            trigger_condition = self.action_dict[action_item]['trigger_condition']

            if self.state == o_state and self.state != e_state:
                res_end = self.get_rule_result(trigger_condition)
                if res_end:
                    if len(self.action_segment) == 0:
                        self.action_segment[o_state] = [0, self.frame_counter]
                    else:
                        key = list(self.action_segment.keys())[len(self.action_segment) - 1]
                        self.action_segment[o_state] = [self.action_segment[key][1], self.frame_counter]
                    if len(self.action_segment) == len(self.yaml_action_class) - 1:
                        self.action_segment[e_state] = [self.frame_counter, self.frames_num]
                    logger.debug("*************".format(self.frame_counter))
                    self.state = e_state


    """
    底部检测
    """
    def sub_evaluate(self, args):
        for exp in self.exp_class_relation:
            if exp not in self.clstor_res_dict:
                continue

            predict = self.clstor_res_dict[exp] # 模型预测值
            if predict == "":
                continue

            if args.mode_cls == "one":
                predict = predict.split("#")[len(predict.split("#"))-1]

            copy_class_realtion_list = self.exp_class_relation[exp].copy()
            if predict in copy_class_realtion_list:
                copy_class_realtion_list.remove(predict)

            key = exp + "#" + predict

            if exp in self.exp_range_state:
                if key in self.component_num_state:
                    self.set_component_num_state_true(exp, key, copy_class_realtion_list)

                if key in self.component_num_state:
                    for class_item_not in copy_class_realtion_list:
                        not_key = exp + "#" + class_item_not
                        if not_key in self.component_num_state:
                            self.component_num_state[not_key]['num'] = 0
            else:
                # 没有true_num和false_num
                self.component_num_state[key]['state'] = True
                for class_item_not in copy_class_realtion_list:
                    not_key = exp + "#" + class_item_not
                    if not_key in self.component_num_state:
                        self.component_num_state[not_key]['state'] = False


    def set_component_num_state_true(self, exp, key, copy_class_realtion_list):
        self.component_num_state[key]['num'] += 1
        if self.component_num_state[key]['num'] >= self.yaml_params['true_num']:

            for class_item_not in copy_class_realtion_list:
                not_key = exp + "#" + class_item_not
                if not_key in self.component_num_state:
                    # self.component_num_state[not_key]['num'] = 0
                    self.component_num_state[not_key]['state'] = False

            self.component_num_state[key]['state'] = True


    def filter_object(self, top_det_results, side_det_results):
        # filter the object based on logic (eliminate not logic item,eg. 2 riders)
        first_top_det_results = self.filter_one_object(top_det_results)
        first_side_det_results = self.filter_one_object(side_det_results)
        return first_top_det_results, first_side_det_results

    def filter_one_object(self, det_results):
        coor_dict = {}
        for exp_device_item in self.exp_device:
            coor_dict[exp_device_item] = []

        if det_results[0] is not None:
            unwanted_list = []
            for index, (obj, coor, confidence) in enumerate(zip(det_results[2], det_results[0], det_results[-1])):
                if obj in coor_dict.keys():
                    unwanted_list = self.get_unwanted_list(coor_dict[obj], index, coor, confidence,
                                                           unwanted_list)
            for ele in sorted(unwanted_list, reverse=True):
                det_results[0] = np.delete(det_results[0], ele, axis=0)
                det_results[1] = np.delete(det_results[1], ele)
                det_results[2] = np.delete(det_results[2], ele)
                det_results[3] = np.delete(det_results[3], ele)
        det_results[2] = list(det_results[2])
        return det_results

    def get_unwanted_list(self, item_coor, index, coor, confidence, unwanted_list):
        if len(item_coor) == 0:
            item_coor.append([index, coor, confidence])
        else:
            if item_coor[0][2] > confidence:
                unwanted_list.append(index)
            else:
                unwanted_list.append(item_coor[0][0])
        return unwanted_list