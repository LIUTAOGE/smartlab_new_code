import copy
import json

import cv2
import time

import yaml
from openvino.runtime import Core

from evaluator import Evaluator
from object_detection.detector import Detector
import os
from state_cls.clstor import Clstor
from sys_config import config
from argparse import ArgumentParser

from utils.exception_utils import DataFormatException
from utils.logger_config import get_logger

logger = get_logger(config.LOG_PATH)

def video_loop(args, yaml, cap_top, cap_side, detector, clstor, frames_num, evaluator, json_name, video_id):

    yaml_params = yaml['yaml_params']
    clstor_res_dict = {}
    in_time = time.time()
    old_time = time.time()
    frame_counter = 0
    fps = 0.0
    interval_second = 1
    interval_start_frame = 0
    detector_result = None
    buffer_display = None
    future_detector = None

    top_exp_angle = yaml['yaml_def']['exp_angle']['top']
    side_exp_angle = yaml['yaml_def']['exp_angle']['side']

    for t_e_d in top_exp_angle:
        clstor_res_dict[t_e_d] = ""
    for s_e_d in side_exp_angle:
        clstor_res_dict[s_e_d] = ""

    while cap_top.isOpened() and cap_side.isOpened():
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()

        if not ret_top or not ret_side:

            evaluator.init_yaml(yaml)

            json_dir = config.public_config["public_path"] + config.public_config["json_path"]
            logger.debug("video_loop json_dir, {}".format(json_dir))
            create_dir_not_exist(json_dir)
            record_file_path = json_dir + "/" + json_name
            if os.path.exists(record_file_path):
                with open(record_file_path, 'r') as f:
                    score_data = json.load(f)
            else:
                score_data = {}

            key = video_id

            clip_keyframe_new = {}
            if buffer_display is not None:
                top_det_results, side_det_results, frame_top, frame_side, display_frame_counter, keyframe_dict, keyname_action, action_segment, frames_num = buffer_display
                for dict in keyframe_dict:

                    clip_keyframe_new[dict] = keyframe_dict[dict]['clip_keyframe']
                    if len(clip_keyframe_new[dict]) > 0:
                        clip_keyframe_new[dict + "_score"] = 1
                    else:
                        clip_keyframe_new[dict + "_score"] = 0
                    clip_keyframe_new[dict] = combine_deal(clip_keyframe_new, dict, keyname_action, action_segment, frames_num) # 合并处理
                score_data[key] = clip_keyframe_new

            # 完成后
            logger.info(f"正在记录f{key}视频")
            with open(record_file_path, 'w') as f:
                json.dump(score_data, f, indent=4, ensure_ascii=False)
            # break

            current_time_save_json = time.time()

            logger.debug("finish time, {}".format(current_time_save_json - in_time))
            # 将视频截取
            #save_clip_video(yaml_params, cap_top, cap_side, clip_keyframe_new, video_id)

            current_time_save_clip_video = time.time()

            release_camera(cap_side, cap_top)
            logger.debug("finish time save_clip_video, {}".format(current_time_save_clip_video - in_time))
            return score_data

        else:
            frame_counter += 1


            if frame_counter % 5 == 0 and future_detector is None:
                future_detector = detector.inference(frame_top, frame_side)

            # get obj result
            if future_detector is not None:
                detector_result = future_detector
                future_detector = None

            current_time = time.time()
            current_frame = frame_counter
            if (current_time - old_time > interval_second):
                total_frame_processed_in_interval = current_frame - interval_start_frame
                fps = total_frame_processed_in_interval / (current_time - old_time)
                interval_start_frame = current_frame
                old_time = current_time

            ''' The score evaluation module need to merge the results of the two modules and generate the scores '''
            if detector_result is not None:
                top_det_results, side_det_results = detector_result[0], detector_result[1]
                keyframe_dict, state, top_det_results, side_det_results, frame_top, frame_side, display_frame_counter, keyname_action, action_segment, frames_num = evaluator.inference(
                    top_det_results=top_det_results,
                    side_det_results=side_det_results,
                    clstor_res_dict=clstor_res_dict,
                    frame_top=frame_top,
                    frame_side=frame_side,
                    frame_counter=frame_counter,
                    frames_num=frames_num,
                    yaml_params=yaml_params,
                    args=args,
                    clstor=clstor,
                    top_exp_angle=top_exp_angle,
                    side_exp_angle=side_exp_angle
                )
                buffer_display = top_det_results, side_det_results, frame_top, frame_side, display_frame_counter, keyframe_dict, keyname_action, action_segment, frames_num
                # logger.debug("fps,{}".format(fps))

        if cv2.waitKey(1) in {ord('q'), ord('Q'), 27}:  # Esc
            break


def release_camera(cap_side, cap_top):
    cap_top.release()
    cap_side.release()
    cv2.destroyAllWindows()


def combine_deal(clip_keyframe_new, dict, keyname_action, action_segment, frames_num):
    s = clip_keyframe_new[dict]
    s = [list(item) for item in s]
    # 合并预测相邻区间
    last_end = -1
    new_s = []
    for k_idx, k in enumerate(s):
        if last_end == -1:
            last_end = k[1]
            new_s.append(copy.deepcopy(k))
        else:
            if k[0] - last_end < 200:
                new_s[-1][-1] = k[1]
            else:
                new_s.append(copy.deepcopy(k))
            last_end = k[1]
    # 选最长的一段
    if len(new_s) > 0:
        longest_range = -1
        longest_idx = -1
        for k_idx, k in enumerate(new_s):
            r = k[1] - k[0]
            if r < 0:
                raise DataFormatException("combine deal error")
            if r > longest_range:
                longest_idx = k_idx
                longest_range = r
        new_s = [new_s[longest_idx]]
    else:
        # 段内都是0
        action_name = keyname_action[dict].split("-")[0] if "-" in keyname_action[dict] else keyname_action[dict]
        result = [action_segment[action_name] if action_name in action_segment else [0, frames_num]]
        new_s = result
    return new_s


# 截取视频
def save_clip_video(yaml_params, cap_top, cap_side, clip_keyframe_new, video_id):
    save_video_param = yaml_params['save_video_param']
    video_clip_path = config.public_config["public_path"] + config.public_config["video_clip_path"]
    video_clip_trans_path = config.public_config["public_path"] + config.public_config["video_clip_trans_path"]
    video_code = save_video_param['video_clip_code']
    fps = save_video_param['fps']
    width = save_video_param['width']
    height = save_video_param['height']
    max_size = save_video_param['max_size']
    fourcc = cv2.VideoWriter_fourcc(*video_code)

    for keyframe_item in clip_keyframe_new:

        if len(clip_keyframe_new[keyframe_item]) == 0:
            logger.info("keyframe size is empty")
            continue
        if len(clip_keyframe_new[keyframe_item][0]) <= 1:
            logger.info("keyframe not end or start")
            continue

        start_keyframe_frame = clip_keyframe_new[keyframe_item][0][0]
        end_keyframe_frame = clip_keyframe_new[keyframe_item][0][1]
        cap_top.set(cv2.CAP_PROP_POS_FRAMES, start_keyframe_frame - 1)
        cap_side.set(cv2.CAP_PROP_POS_FRAMES, start_keyframe_frame - 1)
        sub_frame_counter = start_keyframe_frame
        frame_list_top = []
        frame_list_side = []


        save_path_side, save_path_top = get_save_path(keyframe_item, save_video_param, video_id, video_clip_path, "video_clip_suffix")
        trans_side, trans_top = get_save_path(keyframe_item, save_video_param, video_id, video_clip_trans_path, "video_clip_trans")

        out_top = cv2.VideoWriter(save_path_top, fourcc, fps, (width, height))
        out_side = cv2.VideoWriter(save_path_side, fourcc, fps, (width, height))

        logger.debug("save_path_top, {}  and trans_top, {}".format(save_path_top, trans_top))
        logger.debug("save_path_side, {} and trans_side, {}".format(save_path_side, trans_side))
        while cap_top.isOpened() and cap_side.isOpened():
            ret_top, frame_top = cap_top.read()
            ret_side, frame_side = cap_side.read()
            if sub_frame_counter >= start_keyframe_frame and sub_frame_counter <= end_keyframe_frame:
                buffer_write(frame_list_side, frame_list_top, frame_side, frame_top, height, max_size, out_side,
                             out_top, width)

            else:
                write_and_trans(frame_list_side, frame_list_top, height, out_side, out_top, save_path_side,
                                save_path_top, trans_side, trans_top, width)

                break

            sub_frame_counter += 1


def write_and_trans(frame_list_side, frame_list_top, height, out_side, out_top, save_path_side, save_path_top,
                    trans_side, trans_top, width):
    write(width, height, frame_list_top, out_top)
    out_top.release()
    write(width, height, frame_list_side, out_side)
    out_side.release()
    if os.path.exists(save_path_top):
        os.system(f'ffmpeg -y -i "{save_path_top}" -vcodec h264 "{trans_top}"')
    if os.path.exists(save_path_side):
        os.system(f'ffmpeg -y -i "{save_path_side}" -vcodec h264 {trans_side}')


def buffer_write(frame_list_side, frame_list_top, frame_side, frame_top, height, max_size, out_side, out_top, width):
    frame_list_top.append(frame_top)
    frame_list_side.append(frame_side)
    if len(frame_list_top) >= max_size:
        write(width, height, frame_list_top, out_top)
        frame_list_top.clear()
    if len(frame_list_side) >= max_size:
        write(width, height, frame_list_side, out_side)
        frame_list_side.clear()


def get_save_path(keyframe_item, save_video_param, video_id, video_clip_path, video_clip_suffix):
    dir_save_path_top = video_clip_path.replace("{{video_id}}", video_id).replace("{{angle}}", "top")
    dir_save_path_side = video_clip_path.replace("{{video_id}}", video_id).replace("{{angle}}", "side")
    create_dir_not_exist(dir_save_path_top)
    create_dir_not_exist(dir_save_path_side)
    save_path_top = dir_save_path_top + "/" + keyframe_item + save_video_param[video_clip_suffix]
    save_path_side = dir_save_path_side + "/" + keyframe_item + save_video_param[video_clip_suffix]
    return save_path_side, save_path_top


def write(width, height, imgs, out):
    if len(imgs) > 0:
        for m in imgs:
            m = cv2.resize(m, (width, height))
            out.write(m)


def build_argparser(expId):

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    yaml_path = config.public_config["public_path"] + config.public_config["yaml_path"] + "/" + expId

    args.add_argument("--data_param", type=str, default= yaml_path + config.public_config["yam_param_suffix"], help="param.yaml path")
    args.add_argument("--data_def", type=str, default=yaml_path + config.public_config["yaml_def_suffix"], help="def.yaml path")
    args.add_argument("--data_rule", type=str, default=yaml_path + config.public_config["yaml_rule_suffix"], help="rule.yaml path")
    return parser

def yaml_load(args, type):
    if type == "data_param":
        input_data = args.data_param
    elif type == "data_def":
        input_data = args.data_def
    else:
        input_data = args.data_rule
    with open(input_data, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def extract_keyframe_service(expId, video_list, json_name):
    keyframe = {}
    for video_id in video_list:
        keyframe = single_video(expId, video_id, json_name)

    return keyframe

def single_video(expId, video_id, json_name):
    args = build_argparser(expId).parse_args(args=[])
    yaml = {}
    yaml['yaml_params'] = yaml_load(args, "data_param")
    yaml['yaml_def'] = yaml_load(args, "data_def")
    yaml['yaml_rule'] = yaml_load(args, "data_rule")

    yaml_params = yaml['yaml_params']
    args.mode_cls = yaml_params['mode_cls']
    args.device = yaml_params['device']

    input_video_path = config.public_config['public_path'] + config.public_config["input_video_path"].replace("{{video_id}}", video_id)
    top_video = input_video_path.replace("{{angle}}", "top")
    side_video = input_video_path.replace("{{angle}}", "side")
    logger.debug("top_video_path, {} and side_video_path, {}".format(top_video, side_video))

    args.topview = top_video + yaml_params['video_suffix']
    args.sideview = side_video + yaml_params['video_suffix']

    core = Core()

    "检测对象"
    detector = Detector(
        core,
        args.device,
        yaml['yaml_def']['exp_object']['cls_top'],
        yaml['yaml_def']['exp_object']['cls_side'],
        yaml_params)

    if args.mode_cls == "multi":
        mae_models_dict = yaml_params['class_model']['multi']['mae']
        mobilev2_models_dict = yaml_params['class_model']['multi']['mobilenet']
    else:
        mae_models_dict = {}
        mobilev2_models_dict = yaml_params['class_model']['one']['mobilenet']

    clstor = Clstor(
        core=core,
        device=args.device,
        mae_models=mae_models_dict,
        mobilev2_models=mobilev2_models_dict)

    "打分对象"
    evaluator = Evaluator(yaml)

    """
    摄像头的代码 
    """
    cap_top = cv2.VideoCapture(args.topview)
    if not cap_top.isOpened():
        raise ValueError(f"Can't read an video or frame from {args.topview}")
    cap_side = cv2.VideoCapture(args.sideview)

    frames_num_top = int(cap_top.get(7))
    frames_num_side = int(cap_side.get(7))
    frames_num = min(frames_num_top, frames_num_side)

    if not cap_side.isOpened():
        raise ValueError(f"Can't read an video or frame from {args.sideview}")

    logger.debug("top fps: {}".format(cap_top.get(cv2.CAP_PROP_FPS)))
    logger.debug("side fps: {}".format(cap_side.get(cv2.CAP_PROP_FPS)))

    keyframe = video_loop(
        args, yaml, cap_top, cap_side, detector, clstor, frames_num, evaluator, json_name, video_id)

    return keyframe

