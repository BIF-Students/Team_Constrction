# Import necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from chemistry.distance import *
from chemistry.jdi import *
from chemistry.joi import *
from chemistry.netoi import *
from chemistry.responsibility_share import *
from chemistry.smallTest import test_players_in_a_match
from chemistry.sql_statements import *
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *

print('hej')

'''def get_normalized_vales(compId):
    
    return df_joi90_and_jdi90, stamps, df_sqaud, df_process_21_22, df_players_teams
'''

sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionIds="795, 524, 364")

print("hej")
'''
sd_table = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/sd_table.csv")
df_events_goals = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_events_goals.csv")
df_keepers = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_keepers.csv")
df_players_teams = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_players_teams.csv")
df_pos = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_pos.csv")
df_sqaud = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_sqaud.csv")
'''
def get_chemistry_leagues(sd_table, df_matches_all, df_sqaud, df_keepers, df_pos):
    # Load data from a SQL database table into a pandas DataFrame
    #sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionId=competitionIds)

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

    compIds = sd_table['competitionId'].unique()
    factor_list = []
    jdi_joi_list = []
    oi_list = []
    jdi_list = []
    share_list = []
    dist_list = []
    share_dist_list = []
    time_list = []
    squad_list = []
    jdi_oi_list = []
    for comId in compIds:
        league_df = sd_table[sd_table['competitionId'] == comId ]
        df_process = league_df.copy()
        s_21_22 = max(df_process.seasonId)
        df_process_20_21 = df_process[df_process['seasonId'] == min(df_process.seasonId)]
        df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
        matches_filtered = df_matches_all.merge(df_process_21_22, on=['matchId'], how = 'inner')
        df_sqaud_filtered = df_sqaud[df_sqaud['seasonId'] == s_21_22]
        pos_filtered = df_pos[df_pos['seasonId'] == s_21_22]
        squad_list.append(df_sqaud)
        pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
        df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

        # Extract net offensive impact per game per team
        df_net_oi = getOi(df_process.copy())
        # Extract distance measures
        df_ec = getDistance(df_process_21_22.copy())
        # Extract players shares
        df_player_share = getResponsibilityShare((df_process_21_22.copy()))

        # Extract jdi
        df_jdi, share_dist = get_jdi(df_player_share, df_ec, df_net_oi, matches_filtered)
        # Extract joi
        df_joi = get_joi(pos_sorted)
        # Computes joi90 and jdi90 for each pair of players
        df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

        stamps = get_timestamps(max(df_process.seasonId))
        ready_for_scaling = prepare_for_scaling(df_process_21_22, df_sqaud, stamps)
        ready_for_scaling['seasonId'] = s_21_22
        df_joi90_and_jdi90['seasonId'] = s_21_22
        factor_list.append(ready_for_scaling)
        jdi_joi_list.append(df_joi90_and_jdi90)
        oi_list.append(df_jdi)
        share_dist_list.append(share_dist)
        #jdi_oi_list.append(jdi_with_oi)
        #jdi_list.append(df_jdi)
        #share_list.append(df_player_share)
        #dist_list.append(df_ec)
        #time_list.append(df_pairwise_playing_time)
    #factor_frame = pd.concat(factor_list)
    #joi_jdi_frame = pd.concat(jdi_joi_list)
    oi_frame = pd.concat(oi_list)
    share_dist_frame = pd.concat(share_dist_list)
    #jid_oi_frame = pd.concat(jdi_oi_list)
    #jdi_frame = pd.concat(jdi_list)
    #share_frame = pd.concat(share_list)
    #dist_frame = pd.concat(dist_list)
    #time_frame = pd.concat(time_list)
    #df_chemistry = get_chemistry(factor_frame, joi_jdi_frame )
    return oi_frame, share_dist_frame
    #return df_chemistry, dist_frame, share_frame, jdi_frame, oi_frame, joi_jdi_frame, factor_frame, time_frame

#df_chemistry, dist_frame, share_frame, jdi_frame, oi_frame, joi_jdi_frame, factor_frame, time_frame = get_chemistry_leagues(sd_table.copy(), df_matches_all.copy(), df_sqaud.copy(), df_keepers.copy(), df_pos.copy())
oi_frame, share_dist_frame = get_chemistry_leagues(sd_table.copy(), df_matches_all.copy(), df_sqaud.copy(), df_keepers.copy(), df_pos.copy())



matches_filtered = df_matches_all.merge(df_process_21_22, on=['matchId'], how = 'inner')
df_sqaud_filtered = df_sqaud[df_sqaud['seasonId'] == s_21_22]
pos_filtered = df_pos[df_pos['seasonId'] == s_21_22]
pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

# Extract net offensive impact per game per team
df_net_oi = getOi(df_process.copy())
# Extract distance measures
df_ec = getDistance(df_process_21_22.copy())
# Extract players shares
df_player_share = getResponsibilityShare((df_process_21_22.copy()))

# Extract jdi
df_jdi, share_dist = get_jdi(df_player_share, df_ec, df_net_oi, matches_filtered)

df_overview = get_overview_frame(df_chemistry, df_players_teams)
df_spain = df_overview[df_overview['seasonId'] == 187526]
df_eng = df_overview[df_overview['seasonId'] == 187475]

chem_ability = generate_chemistry_ability(df_overview)
chem_ability_v2 = generate_chemistry_ability_v2(df_overview)




def get_jdi_leagues(sd_table, df_matches_all, df_sqaud, df_keepers, df_pos):
    # Load data from a SQL database table into a pandas DataFrame
    #sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionId=competitionIds)



    #Remove player zero
    sd_table = sd_table[sd_table['playerId'] != 0]

    compIds = sd_table['competitionId'].unique()
    jdi_list = []
    squad_list = []
    oi_list = []
    share_list = []
    for comId in compIds:
        league_df = sd_table[sd_table['competitionId'] == comId ]
        df_process = league_df.copy()
        s_21_22 = max(df_process.seasonId)
        df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
        df_sqaud_filtered = df_sqaud[df_sqaud['seasonId'] == s_21_22]
        pos_filtered = df_pos[df_pos['seasonId'] == s_21_22]
        squad_list.append(df_sqaud)
        # Extract net offensive impact per game per team
        df_net_oi = getOi(df_process.copy())
        # Extract distance measures
        df_ec = getDistance(df_process_21_22.copy())
        # Extract players shares
        df_player_share = getResponsibilityShare((df_process_21_22.copy()))
        oi_list.append(df_net_oi)
        # Extract jdi
        df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all)
        df_jdi['seasonId'] = s_21_22
        jdi_list.append(df_jdi)
        share_list.append(df_player_share)
    jdi_frame = pd.concat(jdi_list)
    oi_frame = pd.concat(oi_list)
    share_frame = pd.concat(share_list)
    return jdi_frame, oi_frame, share_frame




















def get_jdi_leagues_v2(sd_table, df_matches_all, df_sqaud, df_keepers, df_pos):
    # Load data from a SQL database table into a pandas DataFrame
    #sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionId=competitionIds)

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

    compIds = sd_table['competitionId'].unique()
    jdi_list = []
    squad_list = []
    oi_list = []
    share_list = []
    for comId in compIds:
        league_df = sd_table[sd_table['competitionId'] == comId ]
        df_process = league_df.copy()
        s_21_22 = max(df_process.seasonId)
        df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
        df_sqaud_filtered = df_sqaud[df_sqaud['seasonId'] == s_21_22]
        pos_filtered = df_pos[df_pos['seasonId'] == s_21_22]
        squad_list.append(df_sqaud)
        # Extract net offensive impact per game per team
        df_net_oi = getOi_v2(df_process.copy())
        # Extract distance measures
        df_ec = getDistance(df_process_21_22.copy())
        # Extract players shares
        df_player_share = getResponsibilityShare_v4((df_process_21_22.copy()))
        oi_list.append(df_net_oi)
        # Extract jdi
        df_jdi = jdi_found_v2(df_player_share, df_ec, df_net_oi, df_matches_all)
        df_jdi['seasonId'] = s_21_22
        jdi_list.append(df_jdi)
        share_list.append(df_player_share)
    jdi_frame = pd.concat(jdi_list)
    oi_frame = pd.concat(oi_list)
    share_frame = pd.concat(share_list)
    return jdi_frame, oi_frame, share_frame

jdi, oi, share = get_jdi_leagues_v2(sd_table.copy(), df_matches_all.copy(), df_sqaud.copy(), df_keepers.copy(), df_pos.copy())
df_jdi_season = jdi.groupby(['p1', 'p2', 'teamId', 'seasonId'], as_index=False)['jdi'].sum()

tester = share[share['playerId'] ==237034]
tester2 = share[share['playerId'] ==395636]

df_overview_v2 = get_overview_frame_jdi(df_jdi_season, df_players_teams)
df_spain_v2 = df_overview_v2[df_overview_v2['seasonId'] == 187526]
df_eng_v2 = df_overview_v2[df_overview_v2['seasonId'] == 187475]

merged = merged[['matchId', 'teamId', 'playerId1', 'playerId2', 'pairwise_involvement',
       'jdi_zone_1', 'jdi_zone_2', 'jdi_zone_3', 'jdi_zone_4', 'jdi_zone_5',
       'jdi_zone_6', 'jdi', 'p1', 'p2', 'zone_1', 'zone_2', 'zone_3', 'zone_4',
       'zone_5', 'zone_6', 'zone_1_prior_avg', 'zone_2_prior_avg', 'zone_3_prior_avg',
       'zone_4_prior_avg', 'zone_5_prior_avg', 'zone_6_prior_avg',
       'zone_1_net_oi', 'zone_2_net_oi', 'zone_3_net_oi', 'zone_4_net_oi',
       'zone_5_net_oi', 'zone_6_net_oi']]


tester = jdi_frame_t.merge(share_frame_v2, on=['matchId', 'p1', 'p2'])
tester = tester.drop_dupliactes(subset = ['matchId', 'p1', 'p2'], keep = 'first').reset_index(drop = True)
tester.columns

df_overview_v2 = get_overview_frame_jdi_oi(tester, df_players_teams)
df_overview_v2 = df_overview_v2.merge(share_frame_v2, on=['p1', 'p2'])
