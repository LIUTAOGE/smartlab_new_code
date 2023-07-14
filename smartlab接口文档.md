# 1. 视频抽帧接口
#### 接口： Post http://172.16.30.3:30005/smartlab/video/extractFrames
#### 请求数据：
```json
{
	"videoId": "A1120003992009080028-2021-12-07-07-37-30",   //学生实验唯一id
    "view": "top"
}
```
#### 返回数据：
```json
{
    "code": 200,                                                        //200：正常状态码；999999：异常状态码
    "result": {
        "execTaskId": "A1120003992009080028-2021-12-07-07-37-30_top"    //为视频id与视角的拼接
        
    },
    "message": "success"
}
```

# 2. 抽帧进度查询
#### 接口： Post http://172.16.30.3:30005/smartlab/video/extractStatus
#### 请求数据：
```json
{
	"execTaskId": "A1120003992009080028-2021-12-07-07-37-30_top"
}
```
#### 返回数据：
```json
{
	"code": 200,        //200：正常状态码；999999：异常状态码
	"result": {
		"status": 1     //抽取状态。1：完成， 0：未完成, -1：抽取异常。
	},
	"message": "success"
}
```

# 3. 模型训练接口
#### 接口： Post http://172.16.30.3:30005/smartlab/modelPlatform/modelTrain
#### 请求数据：
```json
{
    "annTaskId": "anno_task_id_such_as_increment_id_object",         //标注任务id
    "taskType": "objDec",     //任务类型。objDec:目标检测，objCls:分类。
    "version": "v1"              //训练版本
}
```
#### 返回数据：
```json
{
    "code": 200,                                                                  //200：正常状态码；999999：异常状态码
    "result": {
        "execTaskId": "anno_task_id_such_as_increment_id_object_objDec_v1"        //执行任务Id, 为标注任务id、任务类型、训练版本的拼接
    },
    "message": "success"
}
```
# 4. 训练进度查询
#### 接口： Post http://172.16.30.3:30005/smartlab/modelPlatform/trainStatus
#### 请求数据：
```json
{
	"execTaskId": "anno_task_id_such_as_increment_id_object_objDec_v1"
}
```
#### 返回数据：
```json
{
    "code": 200,
    "result": {
        "status": 1,         //训练状态。1：完成， 0：未完成, -1：训练异常。
        "modelMetrics": {"AP50":  95.0, "AP50:95": 88.7}    //注：该字段仅在训练完成时会返回
	},
    "message": "success",   // 正常情况返回success, 异常情况返回异常提示
    "exceptionId": ""       //注：该字段仅在训练异常时会返回
}
```

# 5. 视频关键帧抽取接口
#### 接口： Post http://172.16.30.3:30005/smartlab/keyframe/extractKeyframe
#### 请求数据：
```json
{   "expId": "chemistry", // 实验id
    "video_list": [    // list id
        "2023-03-06-14-23-41",
        "2023-03-06-09-23-01",
        "2023-03-06-11-24-24"
    ]
}
```
#### 返回数据：
```json
{
    "code": 200,
    "result": {
        "execTaskId": "333"
    },
    "message": "success"
}
```

# 6. 视频关键帧进度查询
#### 接口： Post http://172.16.30.3:30005/smartlab/keyframe/extractkeyframeStatus
#### 请求数据：
```json
{
        "execTaskId": "333"
}
```
#### 返回数据：
```json
{
    "code": 200,
    "message": "success",
    "result": {
        "status": "1", //抽取状态。1：完成， 0：未完成, -1：抽取异常。  2: 批量数据，有部分视频正常抽取
        "keyframe": {
            "2023-03-06-14-43-15": {
                "initial": [
                    [
                        44,
                        406
                    ]
                ],
                "measuring_first_tray": [
                    [
                        360,
                        722
                    ]
                ],
                "measuring_first_balance": [
                    [
                        565,
                        927
                    ]
                ],
                "measuring_first_tidy": [
                    [
                        1003,
                        1123
                    ]
                ],
                "measuring_two_cylinder": [
                    [
                        1004,
                        1222
                    ]
                ],
                "measuring_two_tray": [
                    [
                        1157,
                        1519
                    ]
                ],
                "measuring_two_balance": [
                    [
                        1256,
                        1497
                    ]
                ],
                "end": [
                    [
                        1663,
                        1663
                    ]
                ]
            }
        }
    }
}
```

# 7.主动学习

(1) 主动学习训练进度查询:
#### 接口： Post http://172.16.30.3:10005/smartlab/activelearningStatus

#### 请求数据：
```json
{
	"execTaskId": "444"
}
```
#### 返回数据：
```json
{
    "code": 200,
    "result": {
        "status": 1,         //训练状态。1：完成， 0：未完成, -1：训练异常。
        "curRoundSelectedImgs": ["XXX.jpg", "XXX.jpg", "XXX.jpg"，...], //注：该字段仅在训练完成时会返回。当前此轮主动学习挑选的数据的ID
        "curRoundSelectedJson": "当前此轮主动学习挑选的数据保存json文件的路径" //注：该字段仅在训练完成时会返回
	},
    "message": "success",
    "exceptionId": ""       //注：该字段仅在训练异常时会返回
}
```

(2) 主动学习训练
#### 接口： Post http://172.16.30.3:10005/smartlab/activeLearningTrain
#### 请求数据：
```json
{
    "annTaskId": "1",         //标注任务id
    "taskType": "objDec",     //任务类型。objDec:目标检测，objCls:分类。
    "curRound":, 2，//当前主动学习挑选数据的轮数
    "labelJson": "已标注图片的json文件路径",
    "unlabelJson":"未标注图片的json文件路径"
}
```
#### 返回数据：
```json
{
    "code": 200,
    "result": {
        "execTaskId": "444"
	},
    "message": "success"
}
```

# 8.主动学习预标注接口
(1)预标注进度查询:
#### 接口： Post http://172.16.30.3:10005/smartlab/preAnnotationStatus
#### 请求数据：
```json
{
	"execTaskId": "555"
}
```
#### 返回数据：
```json
{
    "code": 200,
    "result": {
        "status": 1,         //训练状态。1：完成， 0：未完成, -1：训练异常。
        "inferenceResultJson": "待标注数据pre_annotation_json的标注结果输出路径" //
	},
    "message": "success",
    "exceptionId": ""       //注：该字段仅在训练异常时会返回
}
```


(2)批量标注接口：
#### 接口： Post http://172.16.30.3:10005/smartlab/preAnnotationBatch
#### 请求数据：
```json
{
    "annTaskId": "1",         //标注任务id
    "taskType": "objDec",     //任务类型。objDec:目标检测，objCls:分类。
    "curRound": 2, //当前主动学习挑选数据的轮数
    "preAnnotationJson": "待标注图片的json文件路径",
    "inferenceResultJson":"待标注数据pre_annotation_json的标注结果输出路径"
}
```
#### 返回数据：
```json
{
	"code": 200,
	"result": {
        "execTaskId": "555"
	},
	"message": "success"
}
```
（3）单个图片标注接口：
#### 接口： Post http://172.16.30.3:10005/smartlab/preAnnotationSingle
#### 请求数据：
```json
{
	"annTaskId": "1",         //标注任务id
    "taskType": "objDec",     //任务类型。objDec:目标检测，objCls:分类。
    "curRound": 2,//当前主动学习挑选数据的轮数
    "imagePath": "标注图片的文件路径"
}
```
#### 返回数据：
```json
{
	"code": 200,
	"result": {
        "inferenceResult": [[cls_id,x_top,y_top,x_bottom,y_bottom,score],...]， //可选，返回推理图片的若干个bbox预测结果
	},
	"message": "success"
}
```