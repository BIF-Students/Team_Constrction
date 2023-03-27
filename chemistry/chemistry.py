# Import necessary modules
from chemistry.distance import getDistance
from chemistry.jdi import getJdi
from chemistry.joi import getJoi
from chemistry.netoi import getOi
from chemistry.responsibility_share import getResponsibilityShare
from chemistry.smallTest import test_players_in_a_match
from chemistry.sql_statements import *
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *
from datetime import datetime


# Load data from a SQL database table into a pandas DataFrame
sd_table = load_table_comp(852)
df_events_related_ids = load_db_to_pd(sql_query="select * from sd_table_re", db_name='Development')
df_matches_all = load_db_to_pd(sql_query="select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] WHERE matchId IN (SELECT matchId from Scouting.dbo.Wyscout_Matches where Scouting.dbo.Wyscout_Matches.seasonId in (187530))", db_name='Development')
df_sqaud = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Match_Squad] WHERE matchId IN (SELECT matchId FROM Scouting.dbo.Wyscout_Matches WHERE Scouting.dbo.Wyscout_Matches.seasonId in (187530));", db_name='Scouting_Raw')
df_keepers = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players] where role_code2 = 'GK'", db_name="Scouting_Raw")


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
df_net_oi = getOi(df_process)

#Extract distance measures
df_ec = getDistance(df_process_21_22)

#Extract players shares
df_player_share = getResponsibilityShare(df_process_21_22)

#Extract jdi
df_jdi = getJdi(df_net_oi, df_matches_all, df_ec, df_player_share)

#Extract joi
df_joi = getJoi(df_events_related_ids)

#df_checker = df_ec[(df_ec.teamId == 4487)]
#df_tester = test_players_in_a_match(df_player_share=df_player_share, df_net_oi= df_net_oi, df_matches_all=df_matches_all, df_ec=df_ec, df_jdi=df_jdi, match=5252420, p1=170237, p2=450371, t1= 4487 )

#Extract normalized values
df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

stamps = get_timestamps(187530)


vaep, subs,tvp = get_TVP(df_process_21_22, df_sqaud, stamps)
