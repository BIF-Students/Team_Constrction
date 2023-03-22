# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *

def getJoi(df):
 df_joi = generate_joi(df)
 return df_joi
