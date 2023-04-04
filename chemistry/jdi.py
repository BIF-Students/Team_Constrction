import numpy as np

from chemistry.chemistry_helpers import *




def get_jdi(player_shares, player_distances, df_net_oi, matches_all):
    df_player_share = player_shares.merge(player_shares, on=['matchId', 'teamId'], suffixes=(1, 2))
    df_player_share_dist = (df_player_share.merge(player_distances, on=['matchId', 'teamId', 'playerId1', 'playerId2']))
    df_player_share_dist = (df_player_share_dist[['playerId1', 'matchId', 'teamId','zone_1_pl1', 'zone_2_pl1',
                                                  'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1',
                                                  'zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2',
                                                  'zone_5_pl2', 'zone_6_pl2','zone_1_t1','zone_2_t1', 'zone_3_t1',
                                                  'zone_4_t1', 'zone_5_t1','zone_6_t1','zone_1_imp1', 'zone_2_imp1',
                                                  'zone_3_imp1','zone_4_imp1','zone_5_imp1', 'zone_6_imp1', 'playerId2',
                                                'zone_1_imp2', 'zone_2_imp2', 'zone_3_imp2', 'zone_4_imp2',
                                                'zone_5_imp2', 'zone_6_imp2', 'distance']]).rename(columns={ 'zone_1_t1': 'zone_1_team','zone_2_t1': 'zone_2_team', 'zone_3_t1': 'zone_3_team','zone_4_t1': 'zone_4_team', 'zone_5_t1': 'zone_5_team', 'zone_6_t1':'zone_6_team'})
    matches_noi_opp = (((((df_net_oi.merge(matches_all, left_on=['matchId', 'teamId'], right_on=['matchId', 'home_teamId']))[['matchId', 'teamId', 'away_teamId']]).rename(columns={'away_teamId': 'opposing_team'})).merge(df_net_oi,left_on=['matchId','opposing_team'],right_on=['matchId','teamId']))[['matchId', 'teamId_x', 'opposing_team', 'zone_1_net_oi', 'zone_2_net_oi', 'zone_3_net_oi', 'zone_4_net_oi','zone_5_net_oi', 'zone_6_net_oi']]).rename(columns={'teamId_x': 'teamId'})
    df_jdi = df_player_share_dist.merge(matches_noi_opp, on=['matchId', 'teamId'])
    df_jdi_v2 = compute_jdi_v2(df_jdi)
    df_jdi_v2 = df_jdi_v2[['matchId', 'teamId', 'playerId1', 'playerId2', 'jdi']]
    df_jdi_v2['p1'] = np.where(df_jdi_v2.playerId1 < df_jdi_v2.playerId2, df_jdi_v2.playerId1, df_jdi_v2.playerId2)
    df_jdi_v2['p2'] = np.where(df_jdi_v2.playerId2 > df_jdi_v2.playerId1, df_jdi_v2.playerId2, df_jdi_v2.playerId1)
    return df_jdi_v2

