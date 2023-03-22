# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *

df_events_related_ids = load_db_to_pd(sql_query="select * from sd_table_re", db_name='Development')
df_joi = generate_joi(df_events_related_ids)
