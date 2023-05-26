# This file is provided by Brøndby IF and includes the scripts used to retrieve data their SQL database

from warnings import warn
import urllib
from xmlrpc.client import Boolean
import pyodbc
import pandas as pd
import sqlalchemy

# Loads query with arguments from db to pandas df
def load_db_to_pd(sql_query: str, arguments: tuple = (), db_name: str = "Scouting_Raw"):
    db_call = f'Driver={{SQL Server}};Server=BIF-SQL02\SQLEXPRESS02;Database={db_name};Trusted_connection=yes'
    con = pyodbc.connect(db_call)
    sql_query_with_args = (sql_query % arguments)
    return pd.read_sql_query(sql_query_with_args, con)

# Uploads the data to the database
def upload_data_to_db(data: pd.DataFrame, table_name: str, db_name: str = 'Development',
                      exists: str = 'append', fast_executemany: bool = False, chunksize: int = None):
    params = urllib.parse.quote_plus(f"Driver={{SQL Server}};Server=BIF-SQL02\SQLEXPRESS02;Database={db_name}")
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, fast_executemany=fast_executemany)
    data.to_sql(table_name, con=engine, if_exists=exists, index=False, chunksize=chunksize)
    return