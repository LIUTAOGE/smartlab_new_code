from fastapi import status
from fastapi.responses import JSONResponse, Response  # , ORJSONResponse


STATUS_SUCC = "1"
STATUS_ERROR = "-1"
STATUS_ING = "0"
STATUS_PART_ERROR = "2"
# 注意有个 * 号 不是笔误， 意思是调用的时候要指定参数 e.g.resp_200（data=xxxx)
def status_N1(*, data: str) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': data,
            'result': {
                "status": "-1"
            }
        }
    )

def status_1(*, data: dict) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "success",
            'result': {
                "status": "1",
                "keyframe": data
            }
        }
    )

def status_data_set(*, sta:str, data: dict) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "success",
            'result': {
                "status": sta,
                "keyframe": data
            }
        }
    )

def status_part_succ(*, sta:str, data: dict) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "part success",
            'result': {
                "status": sta,
                "keyframe": data
            }
        }
    )

def status_0() -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "success",
            'result': {
                "status": "0",
            }
        }
    )

def extract_succ(*, data: str) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "success",
            'result': {
                "execTaskId": data,
            }
        }
    )

def extract_failed(*, data: str) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': data,
            'result': {
                "execTaskId": "error",
            }
        }
    )
