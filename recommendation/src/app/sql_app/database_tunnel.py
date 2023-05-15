from app.sql_app.ssh_utilities import from_private_key
from sshtunnel import SSHTunnelForwarder
from os.path import expanduser
from app.config.config_variables import DATABASE_CONFIG

pkeyfilepath = DATABASE_CONFIG['PEM_FILE']
pemFile = open(pkeyfilepath, 'r')

privateKey = from_private_key( pemFile, password = None)

def get_tunnel():
    tunnel = SSHTunnelForwarder(
        (DATABASE_CONFIG['HOSTNAME'], DATABASE_CONFIG['SSH_PORT']),
        ssh_username = DATABASE_CONFIG['USERNAME'],
        ssh_pkey = privateKey,
        remote_bind_address = (DATABASE_CONFIG['MYSQL_HOSTNAME'], DATABASE_CONFIG['MYSQL_PORT']),
        set_keepalive = 2.0
    )

    tunnel_prod = None

    if DATABASE_CONFIG.get("DEV", False):
        tunnel_prod = SSHTunnelForwarder(
            (DATABASE_CONFIG['PROD_HOSTNAME'], DATABASE_CONFIG['SSH_PORT']),
            ssh_username = DATABASE_CONFIG['PROD_USERNAME'],
            ssh_pkey = privateKey,
            remote_bind_address = (DATABASE_CONFIG['MYSQL_HOSTNAME'], DATABASE_CONFIG['MYSQL_PORT']),
            set_keepalive = 2.0
        )
    return (tunnel, tunnel_prod)