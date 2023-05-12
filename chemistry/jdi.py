import numpy as np

from chemistry.chemistry_helpers import *




def get_jdi(player_shares, player_distances, df_net_oi, matches_all, def_suc):
    df_player_share = player_shares.merge(player_shares, on=['matchId', 'teamId'], suffixes=(1, 2))
    df_player_share = df_player_share[df_player_share['playerId1'] != df_player_share['playerId2'] ]
    df_player_share['id'] = df_player_share.apply( lambda row: tuple(sorted([row['matchId'], row['playerId1'], row['playerId2']])), axis=1)
    df_player_share = df_player_share.drop_duplicates(subset=['id'], keep='first')
    df_player_share = df_player_share.drop(['id'], axis=1)
    df_player_share_dist = (df_player_share.merge(player_distances, on=['matchId', 'teamId', 'playerId1', 'playerId2']))
    df_player_share_dist = (df_player_share_dist[['playerId1', 'matchId', 'teamId','zone_1_pl1', 'zone_2_pl1',
                                                  'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1',
                                                  'zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2',
                                                  'zone_5_pl2', 'zone_6_pl2','zone_1_t1','zone_2_t1', 'zone_3_t1',
                                                  'zone_4_t1', 'zone_5_t1','zone_6_t1','zone_1_imp1', 'zone_2_imp1',
                                                  'zone_3_imp1','zone_4_imp1','zone_5_imp1', 'zone_6_imp1', 'playerId2',
                                                'zone_1_imp2', 'zone_2_imp2', 'zone_3_imp2', 'zone_4_imp2',
                                                'zone_5_imp2', 'zone_6_imp2', 'distance']]).rename(columns={ 'zone_1_t1': 'zone_1_team','zone_2_t1': 'zone_2_team', 'zone_3_t1': 'zone_3_team','zone_4_t1': 'zone_4_team', 'zone_5_t1': 'zone_5_team', 'zone_6_t1':'zone_6_team'})
    matches_noi_opp = df_net_oi.merge(matches_all, on='matchId')
    matches_noi_opp['opposingTeam'] = np.where(matches_noi_opp.teamId == matches_noi_opp.home_teamId, matches_noi_opp.away_teamId,matches_noi_opp.home_teamId)
    matches_noi_opp_m = matches_noi_opp[['matchId','teamId', 'opposingTeam']]
    matches_noi_opp = matches_noi_opp[['matchId', 'teamId', 'zone_1_expected_vaep', 'zone_2_expected_vaep', 'zone_3_expected_vaep',
                     'zone_4_expected_vaep', 'zone_5_expected_vaep', 'zone_6_expected_vaep', 'zone_1_net_oi',
                     'zone_2_net_oi', 'zone_3_net_oi', 'zone_4_net_oi',
                     'zone_5_net_oi', 'zone_6_net_oi']]
    mno = matches_noi_opp_m[['matchId', 'opposingTeam']].merge(matches_noi_opp, left_on=['matchId', 'opposingTeam'], right_on=['matchId', 'teamId'])
    mno = mno.drop(['teamId'], axis=1)
    mno = mno.merge(matches_noi_opp_m, on=['matchId', 'opposingTeam'])
    df_jdi = df_player_share_dist.merge(mno, on=['matchId', 'teamId'])
    df_jdi_v2 = jdi_compute(df_jdi, def_suc)
    df_jdi_v2 = df_jdi_v2[['matchId', 'teamId', 'playerId1', 'playerId2', 'pairwise_involvement', 'jdi_zone_1', 'jdi_zone_2','jdi_zone_3', 'jdi_zone_4', 'jdi_zone_5', 'jdi_zone_6', 'jdi']]
    df_jdi_v2['p1'] = np.where(df_jdi_v2.playerId1 < df_jdi_v2.playerId2, df_jdi_v2.playerId1, df_jdi_v2.playerId2)
    df_jdi_v2['p2'] = np.where(df_jdi_v2.playerId2 > df_jdi_v2.playerId1, df_jdi_v2.playerId2, df_jdi_v2.playerId1)
    return df_jdi_v2




def jdi_found_v2(player_shares, player_distances, df_net_oi, matches_all):
    df_player_share = player_shares.merge(player_shares, on=['matchId', 'teamId'], suffixes=(1, 2))
    df_player_share = df_player_share[df_player_share['playerId1'] != df_player_share['playerId2'] ]
    df_player_share['id'] = df_player_share.apply( lambda row: tuple(sorted([row['matchId'], row['playerId1'], row['playerId2']])), axis=1)
    df_player_share = df_player_share.drop_duplicates(subset=['id'], keep='first')
    df_player_share = df_player_share.drop(['id'], axis=1)
    df_player_share_dist = (df_player_share.merge(player_distances, on=['matchId', 'teamId', 'playerId1', 'playerId2']))
    df_player_share_dist = (df_player_share_dist[['playerId1', 'matchId', 'teamId','zone_1_pl1', 'zone_2_pl1',
                                                  'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1',
                                                  'zone_7_pl1', 'zone_8_pl1', 'zone_9_pl1','zone_1_pl2',
                                                  'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2','zone_5_pl2',
                                                  'zone_6_pl2', 'zone_7_pl2', 'zone_8_pl2','zone_9_pl2',
                                                  'zone_1_t1','zone_2_t1', 'zone_3_t1','zone_4_t1',
                                                  'zone_5_t1','zone_6_t1', 'zone_7_t1','zone_8_t1',
                                                  'zone_9_t1', 'zone_1_imp1', 'zone_2_imp1','zone_3_imp1',
                                                  'zone_4_imp1','zone_5_imp1', 'zone_6_imp1','zone_7_imp1',
                                                  'zone_8_imp1', 'zone_9_imp1', 'playerId2','zone_1_imp2',
                                                  'zone_2_imp2', 'zone_3_imp2', 'zone_4_imp2', 'zone_5_imp2',
                                                  'zone_6_imp2', 'zone_7_imp2', 'zone_8_imp2','zone_9_imp2', 'distance']]).rename(columns={ 'zone_1_t1': 'zone_1_team','zone_2_t1': 'zone_2_team', 'zone_3_t1': 'zone_3_team','zone_4_t1': 'zone_4_team', 'zone_5_t1': 'zone_5_team', 'zone_6_t1':'zone_6_team' , 'zone_7_t1':'zone_7_team', 'zone_8_t1':'zone_8_team', 'zone_9_t1':'zone_9_team'})
    matches_noi_opp = (((((df_net_oi.merge(matches_all, left_on=['matchId', 'teamId'], right_on=['matchId', 'home_teamId']))[['matchId', 'teamId', 'away_teamId']]).rename(columns={'away_teamId': 'opposing_team'})).merge(df_net_oi,left_on=['matchId','opposing_team'],right_on=['matchId','teamId']))[['matchId', 'teamId_x', 'opposing_team', 'zone_1_net_oi', 'zone_2_net_oi', 'zone_3_net_oi', 'zone_4_net_oi','zone_5_net_oi', 'zone_6_net_oi','zone_7_net_oi', 'zone_8_net_oi', 'zone_9_net_oi']]).rename(columns={'teamId_x': 'teamId'})
    df_jdi = df_player_share_dist.merge(matches_noi_opp, on=['matchId', 'teamId'])
    df_jdi_v2 = jdi_computed_v2(df_jdi)
    df_jdi_v2 = df_jdi_v2[['matchId', 'teamId', 'playerId1', 'playerId2', 'pairwise_involvement', 'jdi_zone_1', 'jdi_zone_2', 'jdi_zone_3', 'jdi_zone_4', 'jdi_zone_5', 'jdi_zone_6', 'jdi_zone_7', 'jdi_zone_8', 'jdi_zone_9', 'jdi']]
    df_jdi_v2['p1'] = np.where(df_jdi_v2.playerId1 < df_jdi_v2.playerId2, df_jdi_v2.playerId1, df_jdi_v2.playerId2)
    df_jdi_v2['p2'] = np.where(df_jdi_v2.playerId2 > df_jdi_v2.playerId1, df_jdi_v2.playerId2, df_jdi_v2.playerId1)
    return df_jdi_v2



