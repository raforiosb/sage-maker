from koombea_serve.connection.ssh_utilities import from_private_key
from koombea_serve.config.settings import settings
from sshtunnel import SSHTunnelForwarder

pkeyfilepath = settings.PEM_FILE
pemFile = open(pkeyfilepath, 'r')

privateKey = from_private_key( pemFile, password = None)

def get_tunnel():
    tunnel = SSHTunnelForwarder(
        (settings.HOSTNAME, int(settings.SSH_PORT)),
        ssh_username = settings.USERNAME,
        ssh_pkey = privateKey,
        remote_bind_address = (settings.MYSQL_HOSTNAME, int(settings.MYSQL_PORT)),
        set_keepalive = 2.0
    )
    return tunnel