1. 配置reuqire.txt环境
2. 安装数据库
CREATE TABLE IF NOT EXISTS `keyframe_status`(
   `id` INT UNSIGNED AUTO_INCREMENT,
   `exp_uuid`  VARCHAR(100) NOT NULL,
   `exp_key` VARCHAR(10000) NOT NULL,
   `exp_status` VARCHAR(40) NOT NULL,
   `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
   `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
   PRIMARY KEY ( `id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

3. 配置sys_config下面config配置【其中有yaml地址和数据库配置】