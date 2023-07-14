import os
# 项目根目录
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据库配置
mysql_config = {
    "host": "127.0.0.1",
    "port": 3306,
    "userName": "root",
    "password": "root",
    "dbName": "smartlab",
    "charsets": "utf8"
}

# yaml文件地址配置
public_config = {
    # "public_path": "/datanfs/develop/smartlab_sys_data/", #configs/
    "public_path": "configs/",
    "yaml_path": "definition",
    "json_path": "keyframe_records", #公共存储json路径地址
    "input_video_path": "video/{{video_id}}/{{angle}}/raw_video/{{angle}}",
    "video_clip_path": "video/{{video_id}}/{{angle}}/video_clip",
    "video_clip_trans_path": "video/{{video_id}}/{{angle}}/trans_clip",
    "yam_param_suffix": "/exp_param.yml",
    "yaml_def_suffix": "/exp_def.yml",
    "yaml_rule_suffix": "/exp_rule.yml"
}

# 线程配置
MAX_WORK = 5

# log地址
LOG_PATH = os.path.join(project_dir, "run.log")