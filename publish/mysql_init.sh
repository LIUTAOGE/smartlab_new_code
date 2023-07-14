#!/bin/bash
  
user="root"  
pass="root"
pp="mysql.smartlab"
DB="smartlab"  
TB="keyframe_status" 

mysql -h$pp  -u$user -p$pass <<EOF  
use $DB;            


CREATE TABLE IF NOT EXISTS $TB(
id INT UNSIGNED AUTO_INCREMENT,
exp_uuid  VARCHAR(100) NOT NULL,
exp_key VARCHAR(10000) NOT NULL,
exp_status VARCHAR(40) NOT NULL,
create_time datetime DEFAULT CURRENT_TIMESTAMP,
update_time datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
PRIMARY KEY ( id )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;


EOF
