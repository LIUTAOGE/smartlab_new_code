import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from dataset.mysql_util import DBUtil
from extract_keyframe_service import extract_keyframe_service
from sys_config import config
from utils.json_util import status_N1, status_1, status_0, extract_succ, extract_failed, STATUS_SUCC, STATUS_ING, \
    STATUS_ERROR, STATUS_PART_ERROR, status_part_succ, status_data_set
from utils.logger_config import get_logger

logger = get_logger(config.LOG_PATH)
app = FastAPI(debug=True)


dbUtil = DBUtil()
keyframe_exp_dic = {}
uuid_exp_dic = {}

keyframe_exp_access = {}

pools = ThreadPoolExecutor(config.MAX_WORK)

class extractKeyframe(BaseModel):
    expId: str
    video_list: list

class extractStatus(BaseModel):
    execTaskId: str

@app.post("/smartlab/keyframe/extractKeyframe")
def extractKeyframe(info: extractKeyframe):
    logger.debug(f"*******************前台传参：{info} *******************")
    video_list = info.video_list
    expId = info.expId

    logger.debug("keyframe_exp_dic keys, {}, keyframe_exp_access keys, {}, uuid_exp_dic keys, {}".
                 format(keyframe_exp_dic.keys(), keyframe_exp_access.keys(), uuid_exp_dic.keys()))
    keyframe_k = (str(video_list) + "#" + expId).replace("'", "")
    if keyframe_k not in keyframe_exp_access:
        logger.debug(f"keyframe_exp_access has not")
        try:
            uuid = str(uuid4())
            json_name = uuid + "#" + expId + ".json"
            task = pools.submit(extract_keyframe_service, expId, video_list, json_name)

            keyframe_exp_dic[uuid] = task
            keyframe_exp_access[keyframe_k] = task
            uuid_exp_dic[keyframe_k] = uuid

            dbUtil.save_data(expId, uuid)
        except Exception as e:
            logger.error('message:{}'.format(traceback.format_exc()))
            return extract_failed(data=str(e))

        return extract_succ(data=uuid)
    else:
        logger.debug(f"keyframe_exp_access already has")
        return extract_succ(data=uuid_exp_dic[keyframe_k])


@app.post("/smartlab/keyframe/extractkeyframeStatus")
def extractkeyframeStatus(info: extractStatus):
    logger.debug(f"*******************前台传参：{info}*******************")
    execTaskId = info.execTaskId
    json_dir = config.public_config["public_path"] + config.public_config["json_path"]
    logger.debug("json_dir,{}".format(json_dir))
    uuid_exp_dic_reverse = {v: k for k, v in uuid_exp_dic.items()} # value uuid唯一

    # 字典为空，查询历史记录
    if not keyframe_exp_dic or execTaskId not in keyframe_exp_dic:
        try:
            res = dbUtil.query_last(execTaskId)
            if res and len(res) > 2 and (res[3] == STATUS_SUCC or res[3] == STATUS_PART_ERROR):
                res_json = json.load(open(json_dir + "/" + execTaskId + "#" + res[2] + ".json", 'r', encoding='utf-8'))
                return status_data_set(sta=str(res[3]), data=res_json)
            else:
                return status_N1(data="Please restart extract")
        except Exception as e:
            logger.error('message:{}'.format(traceback.format_exc()))
            return status_N1(data=str(e))

    # 字典不为空
    keyframe_k = uuid_exp_dic_reverse[execTaskId]
    expId = keyframe_k.split("#")[len(keyframe_k.split("#"))-1]
    if keyframe_exp_dic[execTaskId].done():
        try:
            res_json = keyframe_exp_dic[execTaskId].result()
            keyframe_exp_access.pop(keyframe_k)
            keyframe_exp_dic.pop(execTaskId)
            uuid_exp_dic.pop(keyframe_k)
            dbUtil.update_succ(execTaskId, STATUS_SUCC)
        except Exception as e:
            logger.error('message:{}'.format(traceback.format_exc()))
            return status_error_handle(e, execTaskId, expId, keyframe_k, json_dir)

        return status_1(data=res_json)
    else:
        return status_0()

"""
部分成功，返回部分json
"""
def status_error_handle(e, execTaskId, expId, keyframe_k, json_dir):
    try:
        if keyframe_k in keyframe_exp_access:
            keyframe_exp_access.pop(keyframe_k)
        file_path = json_dir + "/" + execTaskId + "#" + expId + ".json"
        if os.path.isfile(file_path):
            res_json = json.load(open(file_path, 'r', encoding='utf-8'))

            dbUtil.update_succ(execTaskId, STATUS_PART_ERROR)

            return status_part_succ(sta=STATUS_PART_ERROR, data=res_json)
        else:
            dbUtil.update_succ(execTaskId, STATUS_ERROR)
            return status_N1(data=str(e))
    except Exception as e:
        logger.error('message:{}'.format(traceback.format_exc()))
        return status_N1(data="status_error_handle error")



# if __name__ == '__main__':
#     uvicorn.run(app)