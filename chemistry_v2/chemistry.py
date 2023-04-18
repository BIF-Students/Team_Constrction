# Import necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from chemistry_v2.distance import *
from chemistry_v2.jdi import *
from chemistry_v2.joi import *
from chemistry_v2.netoi import getOi
from chemistry_v2.responsibility_share import *
from chemistry_v2.smallTest import test_players_in_a_match
from chemistry_v2.sql_statements import *
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry_v2.chemistry_helpers import *


# Load data from a SQL database table into a pandas DataFrame
sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionId=364)

#Extract keeper id's
keepers = (df_keepers.playerId.values).tolist()

#REmove keepers from all used dataframes
df_sqaud = df_sqaud.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

#Remove players not in the starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

#Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
df_pos['sumVaep'] = df_pos['sumVaep'].fillna(0)

#Remove player zero
sd_table = sd_table[sd_table['playerId'] != 0]


'''
This extract the playing time between pair of players 
for a whole season and normalized per 90 minutes
'''
seasons = "186243, 187374, 186267, 187526, 186353, 187528, 186212, 187483, 186810, 187475, 186265,187511"
sd_tab = load_sd_table(seasons)
squad = get_squad(seasons)
df_pairwise_playing_time = compute_pairwise_playing_time(squad)

sd_tab = sd_tab.query("playerId not in @keepers")
sd_tab['sumVaep'] = sd_tab['sumVaep'].fillna(0)
sd_tab = sd_tab[sd_tab['playerId'] != 0]
df_process = sd_tab.copy()
df_process_20_21 = df_process[df_process['seasonId'] == min(df_process.seasonId)]
df_process_21_22 = df_process[df_process['seasonId'] == max(df_process.seasonId)]


#Extract net offensive impact per game per team
df_net_oi = getOi(df_process.copy())

#Extract distance measures
df_ec = get_distance(df_process.copy())
print('hej')

#Extract players shares
df_player_share = get_responsibility_share(df_process.copy())
print('hej')

#Extract jdi
df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all)

df_pos_2 = load_pos_data(seasons)
#Extract joi
df_joi = getJoi(df_pos_2)

#Computes joi90 and jdi90 for each pair of players
df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

'''
Extract minute and second for each event. This is important to 
determine team vaep in a game in the minutes a particular player
was on the pitch
'''
stamps = get_timestamps(seasons)

df_chemistry = compute_chemistry(df_process, df_sqaud, stamps, df_joi90_and_jdi90)
df_overview = get_overview_frame(df_chemistry, df_players_teams)

chem_ability = generate_chemistry_ability(df_overview)
chem_ability_v2 = generate_chemistry_ability_v2(df_overview)
















