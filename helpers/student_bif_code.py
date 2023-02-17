from warnings import warn
import urllib
from xmlrpc.client import Boolean
import pyodbc
import pandas as pd
import sqlalchemy

def load_db_to_pd(sql_query: str, arguments: tuple = (), db_name: str = "Scouting_Raw"):
    """
    Loads query with arguments from db to pandas df

        Parameters:
            :param sql_query: query as string (args denoted as %s in query)
            :param arguments: arguments of varying types in tuple (tuple_length = #args in query)
                                  If an arg is a tuple, convert the tuple to string - ex: (str((123, 124)), #2nd arg)
            :param db_name  : database name (often not needed if database is raw, BI, staging etc)

        Returns:
            Pandas df

        Author:
            NCH
    """
    db_call = f'Driver={{SQL Server}};Server=BIF-SQL02\SQLEXPRESS02;Database={db_name};Trusted_connection=yes'
    con = pyodbc.connect(db_call)
    sql_query_with_args = (sql_query % arguments)
    return pd.read_sql_query(sql_query_with_args, con)

def upload_data_to_db(data: pd.DataFrame, table_name: str, db_name: str = 'Development',
                      exists: str = 'append', fast_executemany: bool = False, chunksize: int = None):
    """
    Uploads the data to the database

        Parameters:
            :param data       : pandas df to be uploaded
            :param table_name : table name
            :param db_name    : database name
            :param exists     : 'append' or 'replace' to append to the data or replace the data already in the db
            :param fast_executemany : False as default, set to True to fast execute
            :param chunksize  : Specify the number of rows in each batch to be written at a time. By default None

        Returns:
            Nothing (data is added to the database)

        Author:
            NCH
    """
    params = urllib.parse.quote_plus(f"Driver={{SQL Server}};Server=BIF-SQL02\SQLEXPRESS02;Database={db_name}")
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, fast_executemany=fast_executemany)
    data.to_sql(table_name, con=engine, if_exists=exists, index=False, chunksize=chunksize)
    return