#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys

import pymysql

import six

from sys_config import config
from utils.json_util import STATUS_SUCC, STATUS_ING
from utils.logger_config import get_logger

logger = get_logger(config.LOG_PATH)

class DBUtil:
    """mysql util"""
    db = None
    cursor = None

    def __init__(self):
        self.host = config.mysql_config['host']
        self.port = config.mysql_config['port']
        self.userName = config.mysql_config['userName']
        self.password = config.mysql_config['password']
        self.dbName = config.mysql_config['dbName']
        self.charsets = config.mysql_config['charsets']

    # 链接数据库
    def get_con(self):
        """ 获取conn """
        self.db = pymysql.Connect(
            host=self.host,
            port=self.port,
            user=self.userName,
            passwd=self.password,
            db=self.dbName,
            charset=self.charsets
        )
        self.cursor = self.db.cursor()

    # 关闭链接
    def close(self):
        self.cursor.close()
        self.db.close()

    # 主键查询数据
    def get_one(self, sql):
        res = None
        try:
            self.get_con()
            self.cursor.execute(sql)
            res = self.cursor.fetchone()
            self.close()
        except Exception as e:
            logger.error("query failed！" + str(e))
            value = sys.exc_info()
            six.reraise(*value)
        return res

    # 查询列表数据
    def get_all(self, sql):
        res = None
        try:
            self.get_con()
            self.cursor.execute(sql)
            res = self.cursor.fetchall()
            self.close()
        except Exception as e:
            logger.error("query failed！" + str(e))
            value = sys.exc_info()
            six.reraise(*value)
        return res

    # 插入数据
    def __insert(self, sql):
        count = 0
        try:
            self.get_con()
            count = self.cursor.execute(sql)
            self.db.commit()
            self.close()
        except Exception as e:
            logger.error("insert failed！" + str(e))
            self.db.rollback()
            value = sys.exc_info()
            six.reraise(*value)
        return count

    # 保存数据
    def save(self, sql):
        return self.__insert(sql)

    # 更新数据
    def update(self, sql):
        return self.__insert(sql)

    # 删除数据
    def delete(self, sql):
        return self.__insert(sql)

    # 删除除了最新的其他数据
    def delete_not_succ(self, key, execTaskId):
        sql = f"delete from keyframe_status where exp_key = '{key}' and exp_status = {STATUS_ING} and id not in (select id from (select id from keyframe_status where exp_uuid = '{execTaskId}'  order by id desc limit 1) tt)"
        logger.debug("delete_not_succ sql is {}".format(sql))
        self.delete(sql)

    # 成功后修改状态
    def update_succ(self, execTaskId, status):
        sql = f"update keyframe_status set exp_status = {status} where exp_uuid = '{execTaskId}' order by id desc limit 1"
        logger.debug("update_succ sql is {}".format(sql))
        res = self.update(sql)
        return res

    # 查询最新一次记录
    def query_last(self, execTaskId):
        sql = f"select * from keyframe_status where exp_uuid = '{execTaskId}'  order by id desc limit 1"
        logger.debug("query_last sql is {}".format(sql))
        res = self.get_one(sql)
        return res

    # 向数据库中存入数据
    def save_data(self, expId, uuid):
        sql = f"INSERT INTO keyframe_status(`exp_uuid`, `exp_key`, `exp_status`) VALUES ('{uuid}', '{expId}', {STATUS_ING})"
        logger.debug("save_data sql is {}".format(sql))
        return self.save(sql)
