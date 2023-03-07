# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
import pandas as pd
from collections import defaultdict

# Load data from a SQL database table into a pandas DataFrame
df = load_db_to_pd(sql_query="select * from sd_table", db_name='Development')

# Define a class called 'EventStats' with five attributes
class EventStats:
    def __init__(self, combinedVaep, eId1, eId2, t1, t2):
        self.combinedVaep = combinedVaep
        self.eId1 = eId1
        self.eId2 = eId2
        self.t1 = t1
        self.t2 = t2

# Define a function called 'makeKey' that takes two integer parameters and returns a string
def makeKey(id1, id2):
    return str(id1) + str(id2) if id1 < id2 else str(id2) + str(id1)

# Create a default dictionary to store EventStats objects for each match and key combination
matches = defaultdict(dict)

# Iterate over the DataFrame, creating EventStats objects for each row and the following row
for i, row in df[:-1].iterrows():
    mId = row['matchId']  # get the match ID for the current row
    key = makeKey(row['playerId'], df.loc[i + 1, 'playerId'])  # create a key based on player IDs
    t1 = int(row['teamId'])  # get the team ID for the current row
    t2 = int(df.loc[i + 1, 'teamId'])  # get the team ID for the following row
    e1 = int(row['eventId'])  # get the event ID for the current row
    e2 = int(df.loc[i + 1, 'eventId'])  # get the event ID for the following row
    p1 = int(row['playerId'])  # get the player ID for the current row
    p2 = int(df.loc[i + 1, 'playerId'])  # get the player ID for the following row

    vaepSum = row['sumVaep'] + df.loc[i + 1, 'sumVaep']  # calculate the sum of the VAEP values for the two events
    # If the two events are on the same team and involve different players, add them to the matches dictionary
    if t1 == t2 and p1 != p2:
        if key in matches[mId]:
            matches[mId][key].append(EventStats(vaepSum, e1, e2, t1, t2))
        else:
            matches[mId][key] = [EventStats(vaepSum, e1, e2, t1, t2)]