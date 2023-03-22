# Import necessary modules
from chemistry.distance import getDistance
from chemistry.jdi import getJdi
from chemistry.netoi import getOi
from chemistry.responsibility_share import getResponsibilityShare
from helpers.student_bif_code import load_db_to_pd  # custom module
from helpers.helperFunctions import *
from chemistry.chemistry_helpers import *


# Load data from a SQL database table into a pandas DataFrame
df = load_db_to_pd(sql_query="select * from sd_table", db_name='Development')
df_events_related_ids = load_db_to_pd(sql_query="select * from sd_table_re", db_name='Development')
df_matches_all = load_db_to_pd(sql_query="select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] WHERE matchId IN (SELECT matchId from Scouting.dbo.Wyscout_Matches where Scouting.dbo.Wyscout_Matches.seasonId in (187530))", db_name='Development')
df_sqaud = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Match_Squad] WHERE matchId IN (SELECT matchId FROM Scouting.dbo.Wyscout_Matches WHERE Scouting.dbo.Wyscout_Matches.seasonId in (187530));", db_name='Scouting_Raw')

#Remove players not in starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

df['sumVaep'] = df['sumVaep'].fillna(0)
df = df[df['playerId'] != 0]

'''
This extract the playing time between pair of players 
for a whole season and normalized per 90 minutes
'''
df_pairwise_playing_time = compute_pairwise_playing_time(df_sqaud)

df_process = df.copy()

df_net_oi = getOi(df_process)
df_ec = getDistance(df_process)

b = df_net_oi[df_net_oi.matchId == 5252305]

df_tester2 = df_ec[(df_ec.player1 == 26000) & (df_ec.matchId == 5252305) | (df_ec.player2 == 26000) & (df_ec.matchId == 5252305) ]

df_player_share = getResponsibilityShare(df_process)

df_p1_dist = df_ec[(df_ec.player1 == 170237) & (df_ec.matchId == 5252420)]
df_p2_dist = df_ec[(df_ec.player1 == 450371) & (df_ec.matchId == 5252420)]

net_oi_g = df_net_oi[(df_net_oi.matchId ==5252420) & (df_net_oi.teamId == 4487)]

dist_p1_p2 = 67.94878

df_p1_imp = df_player_share[(df_player_share.playerId == 170237) & (df_player_share.matchId == 5252420)]
df_p1_imp = pd.merge(df_p1_imp, net_oi_g, on=(['matchId', 'teamId']))
df_p1_imp['dist'] = [67.94878]
df_p1_imp['jd1'] = df_p1_imp.zone_1_imp*df_p1_imp.zone_1_net_oi * df_p1_imp.dist
df_p1_imp['jd2'] = df_p1_imp.zone_2_imp*df_p1_imp.zone_2_net_oi * df_p1_imp.dist
df_p1_imp['jd3'] = df_p1_imp.zone_3_imp*df_p1_imp.zone_3_net_oi * df_p1_imp.dist
df_p1_imp['jd4'] = df_p1_imp.zone_4_imp*df_p1_imp.zone_4_net_oi * df_p1_imp.dist
df_p1_imp['jd5'] = df_p1_imp.zone_5_imp*df_p1_imp.zone_5_net_oi * df_p1_imp.dist
df_p1_imp['jd6'] = df_p1_imp.zone_6_imp*df_p1_imp.zone_6_net_oi * df_p1_imp.dist
df_p1_imp['jdi'] = df_p1_imp.jd1 + df_p1_imp.jd2 + df_p1_imp.jd3 + df_p1_imp.jd4 + df_p1_imp.jd5 + df_p1_imp.jd6
df_p2_imp = df_player_share[(df_player_share.playerId == 450371) & (df_player_share.matchId == 5252420)]
df_p2_imp = pd.merge(df_p2_imp, net_oi_g, on=(['matchId', 'teamId']))
df_p2_imp['dist'] = [67.94878]
df_p2_imp['jd1'] = df_p2_imp.zone_1_imp*df_p1_imp.zone_1_net_oi * df_p2_imp.dist
df_p2_imp['jd2'] = df_p2_imp.zone_2_imp*df_p2_imp.zone_2_net_oi * df_p2_imp.dist
df_p2_imp['jd3'] = df_p2_imp.zone_3_imp*df_p2_imp.zone_3_net_oi * df_p2_imp.dist
df_p2_imp['jd4'] = df_p2_imp.zone_4_imp*df_p2_imp.zone_4_net_oi * df_p2_imp.dist
df_p2_imp['jd5'] = df_p2_imp.zone_5_imp*df_p2_imp.zone_5_net_oi * df_p2_imp.dist
df_p2_imp['jd6'] = df_p2_imp.zone_6_imp*df_p2_imp.zone_6_net_oi * df_p2_imp.dist
df_p2_imp['jdi'] = df_p2_imp.jd1 + df_p2_imp.jd2 + df_p2_imp.jd3 + df_p2_imp.jd4 + df_p2_imp.jd5 + df_p2_imp.jd6

df_team


df_jdi = getJdi(df_net_oi, df_matches_all, df_ec, df_player_share)

h = dist_snoi_doi[(dist_snoi_doi.player2 == 170188) & (dist_snoi_doi.matchId == 5252305) ]
g = df_player_share[(df_player_share.playerId == 170188) & (df_player_share.matchId == 5252305)]
l = df_player_share[(df_player_share.playerId == 26000) & (df_player_share.matchId == 5252305)]

fg = 0.06250 * 32 * 38.91348
gf = 0.31250 * 32 * 38.91348

tester3 =df_jdi[(df_jdi.player1 == 170188) & (df_jdi.matchId == 5252305) & (df_jdi.player1 != 0) & (df_jdi.player2 == 26000) | (df_jdi.player1 == 26000) & (df_jdi.matchId == 5252305) & (df_jdi.player1 != 0) & (df_jdi.player2 == 170188)]

fg + gf

df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)


