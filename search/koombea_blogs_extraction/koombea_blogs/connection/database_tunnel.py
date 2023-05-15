from koombea_blogs.connection.ssh_utilities import from_private_key
from sshtunnel import SSHTunnelForwarder

def get_private_key(settings):
    pkeyfilepath = settings.PEM_FILE
    pemFile = open(pkeyfilepath, 'r')
    privateKey = from_private_key( pemFile, password = None)
    return privateKey

def get_tunnel(settings):
    privateKey = get_private_key(settings)
    tunnel = SSHTunnelForwarder(
        (settings.HOSTNAME, int(settings.SSH_PORT)),
        ssh_username = settings.USERNAME,
        ssh_pkey = privateKey,
        remote_bind_address = (settings.MYSQL_HOSTNAME, int(settings.MYSQL_PORT)),
        set_keepalive = 2.0
    )
    return tunnel