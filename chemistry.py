# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
import pandas as pd
from collections import defaultdict
from helpers.helperFunctions import *
import math
import numpy as np
import itertools
from helpers.chemistry_helpers import *


# Load data from a SQL database table into a pandas DataFrame
df = load_db_to_pd(sql_query="select * from sd_table", db_name='Development')
df_events_related_ids = load_db_to_pd(sql_query="select * from sd_table_re", db_name='Development')
matches_all = load_db_to_pd(sql_query="select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] WHERE matchId IN (SELECT matchId from Scouting.dbo.Wyscout_Matches where Scouting.dbo.Wyscout_Matches.seasonId in (187530))", db_name='Development')
df_sqaud = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Match_Squad] WHERE matchId IN (SELECT matchId FROM Scouting.dbo.Wyscout_Matches WHERE Scouting.dbo.Wyscout_Matches.seasonId in (187530));", db_name='Scouting_Raw')

#Remove players not in starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]


'''
This extract the playing time between pair of players 
for a whole season and normalized per 90 minutes
'''
pairwise_playing_time = compute_pairwise_playing_time(df_sqaud)



#Compute joi values
df_joi = generate_joi(df_events_related_ids)

df['sumVaep'] = df['sumVaep'].fillna(0)
df = df[df['playerId'] != 0]

df_def_actions_player = df
df_vaep_zone_match = df
df_pos_player = df

df_def_actions_player.insert(57, 'nonPosAction', df_def_actions_player.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True) # Mark if action is defensive
df_def_actions_player = df_def_actions_player[df_def_actions_player['nonPosAction'] == 1] # Filter dataframe for defensive actions
df_vaep_zone_match['zone'] = df_vaep_zone_match.apply(lambda row: find_zone_chemistry(row), axis = 1)
df_def_actions_player['zone'] = df_def_actions_player.apply(lambda row: find_zone_chemistry(row), axis = 1) # Filter dataframe for defensive actions

matches_positions = {}
df_pos_player.apply(lambda row: allocate_position(row, matches_positions), axis = 1)

def_zones_vaep = find_zones_and_vaep(df_vaep_zone_match)
df_def_actions_vaep_player = find_zones_and_vaep(df_def_actions_player)
def_zones_vaep = def_zones_vaep.groupby(['matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                       'zone_6': 'sum',
                                                                        'zone_7': 'sum',
                                                                        'zone_8': 'sum',
                                                           'zone_9': 'sum'})
df_running_vaep_avg = team_vaep_game(def_zones_vaep)

def_zones_vaep_player = df_def_actions_vaep_player.groupby(['playerId', 'matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                        'zone_6': 'sum',
                                                                        'zone_7': 'sum',
                                                                        'zone_8': 'sum',
                                                                        'zone_9': 'sum'})



avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions, {}), orient='index', columns=['matchId', 'teamId', 'playerId', 'avg_x', 'avg_y']).reset_index(drop=True)
df_matches_and_teams = (df[['matchId', 'teamId']].drop_duplicates()).reset_index(drop=True)

ec_dict = compute_distances(df_matches_and_teams, avg_position_match_df)
ec_df = pd.DataFrame.from_dict(ec_dict, orient='index', columns=['matchId', 'teamId', 'player1', 'player2', 'distance']).reset_index(drop=True)


'''
First, the net oi in between expected vaep and actual vaep i computed per game per team
Second, a range of steps, explained in the methods are made to prepare the data to the final jdi computation
Third: the actual jdi values are computed by fits using the distances between players as a way 
to determine chemistry values and next the share per player per sone with regards to defensive actions
Finally, a sum of the jdi values per szone is comuted and poses the final jdi computation per a pair of players per game
'''

df_net_oi = compute_net_oi_game(df_running_vaep_avg)
df_swoi_netoi_dist = process_for_jdi(df_def_actions_player, df_net_oi, matches_all, ec_df)
df_jdi = compute_jdi(df_swoi_netoi_dist)
df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, pairwise_playing_time)


df_checker = df_jdi[(df_jdi.matchId == 5252460)]