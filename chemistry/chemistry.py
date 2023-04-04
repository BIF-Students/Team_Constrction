# Import necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from chemistry.distance import *
from chemistry.jdi import *
from chemistry.joi import *
from chemistry.netoi import getOi
from chemistry.responsibility_share import *
from chemistry.smallTest import test_players_in_a_match
from chemistry.sql_statements import *
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *


# Load data from a SQL database table into a pandas DataFrame
sd_table, df_matches_all, df_sqaud, df_keepers, df_events_related_ids, df_players_teams, df_events_goals, df_pos = load_data(competitionId=364)

#Extract keeper id's
keepers = (df_keepers.playerId.values).tolist()

#REmove keepers from all used dataframes
df_sqaud = df_sqaud.query("playerId not in @keepers")
df_events_related_ids = df_events_related_ids.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

#Remove players not in the starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

#Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
df_pos['sumVaep'] = df_pos['sumVaep'].fillna(0)

#Remove player zero
sd_table = sd_table[sd_table['playerId'] != 0]

#Remove player 0
df_events_related_ids = df_events_related_ids[df_events_related_ids['playerId'] != 0]


'''
This extract the playing time between pair of players 
for a whole season and normalized per 90 minutes
'''

df_pairwise_playing_time = compute_pairwise_playing_time(df_sqaud)

df_process = sd_table.copy()
df_process_20_21 = df_process[df_process['seasonId'] == min(df_process.seasonId)]
df_process_21_22 = df_process[df_process['seasonId'] == max(df_process.seasonId)]


#Extract net offensive impact per game per team
df_net_oi = getOi(df_process.copy())

#Extract distance measures
df_ec = getDistance(df_process_21_22.copy())

#Extract players shares
df_player_share = getResponsibilityShare((df_process_21_22.copy()))

#Extract jdi
df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all)

#Extract joi
df_joi = getJoi(df_pos)

#Computes joi90 and jdi90 for each pair of players
df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

'''
Extract minute and second for each event. This is important to 
determine team vaep in a game in the minutes a particular player
was on the pitch
'''
stamps = get_timestamps(max(df_process.seasonId))

df_chemistry = compute_chemistry(df_process_21_22, df_sqaud, stamps, df_joi90_and_jdi90)
df_overview = get_overview_frame(df_chemistry, df_players_teams)

chem_ability = generate_chemistry_ability(df_overview)
chem_ability_v2 = generate_chemistry_ability_v2(df_overview)
















