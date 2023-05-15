import os

STAGE = os.environ.get('STAGE')
CONFIG_DIRECTORY = os.environ.get('CONFIG_DIR', "/opt/ml/input/config")

print("stage: "+STAGE)

DATABASE_CONFIG = {}
"""Configure database params given the stage"""

if STAGE == 'dev':
    DATABASE_CONFIG["PEM_FILE"] = CONFIG_DIRECTORY + "/" + "dataBaseKey.pem"
    DATABASE_CONFIG["HOSTNAME"] = "koombea20stg.ssh.wpengine.net"
    DATABASE_CONFIG["USERNAME"] = "koombea20stg"
    DATABASE_CONFIG['PASSWORD'] = 'opypHiPy2GiuCyApXQpZ'
    DATABASE_CONFIG["SSH_PORT"] = 22

    DATABASE_CONFIG['MYSQL_HOSTNAME'] = '127.0.0.1'
    DATABASE_CONFIG['MYSQL_PORT'] = 3306
    DATABASE_CONFIG['MYSQL_DBNAME'] = 'wp_koombea20stg'

    DATABASE_CONFIG["DEV"] = True

    DATABASE_CONFIG["PROD_HOSTNAME"] = "koombea20.ssh.wpengine.net"
    DATABASE_CONFIG["PROD_MYSQL_DBNAME"] = "wp_koombea20"
    DATABASE_CONFIG["PROD_USERNAME"] = "koombea20"
    DATABASE_CONFIG["PROD_PASSWORD"] = "-WFgRvi2dcg9HDx28JpA"

elif STAGE == "prod":
    DATABASE_CONFIG["PEM_FILE"] = CONFIG_DIRECTORY + "/" + "dataBaseKey.pem"
    DATABASE_CONFIG["HOSTNAME"] = "koombea20.ssh.wpengine.net"
    DATABASE_CONFIG["USERNAME"] = "koombea20"
    DATABASE_CONFIG["PASSWORD"] = "-WFgRvi2dcg9HDx28JpA"
    DATABASE_CONFIG["SSH_PORT"] = 22

    DATABASE_CONFIG['MYSQL_HOSTNAME'] = '127.0.0.1'
    DATABASE_CONFIG['MYSQL_PORT'] = 3306
    DATABASE_CONFIG["MYSQL_DBNAME"] = "wp_koombea20"