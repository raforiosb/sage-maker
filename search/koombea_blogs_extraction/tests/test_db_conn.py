from sqlalchemy.engine import Connection
from koombea_blogs.data_module.data_extraction import DBInformation


def test_connection(conn: Connection):
    needed_tables = DBInformation.necessary_table_names
    tables_names = [
        table_info[0]
        for table_info in conn.execute("show tables").all()
        if table_info[0] in needed_tables
    ]
    assert len(tables_names) == len(
        needed_tables
    ), "Integration test with koombea website db failed!"
