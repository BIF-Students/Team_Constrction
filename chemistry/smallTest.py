import numpy as np
import pandas as pd


def test_players_in_a_match(df_player_share,df_net_oi, df_matches_all, df_ec, df_jdi, match, p1, p2, t1):
    net_oi_g = df_net_oi[(df_net_oi.matchId == match) & (df_net_oi.teamId == t1)]
    df_p1_imp = df_player_share[(df_player_share.playerId == p1) & (df_player_share.matchId == match)]
    df_p1_imp = pd.merge(df_p1_imp, net_oi_g, on=(['matchId', 'teamId']))
    df_p1_dist = df_ec[(df_ec.player1 == p1) & (df_ec.player2 == p2) & (df_ec.matchId == match) | (
                df_ec.player1 == p2) & (df_ec.player2 == p1) & (df_ec.matchId == match)]
    dist = df_p1_dist.distance
    dist = dist.values[0]
    df_p1_imp['dist'] = [dist]
    df_p1_imp = df_p1_imp.merge(df_matches_all, on=['matchId'])
    # Determine the opposing team in order for us to check whether opposing team over or underperforms in an area
    df_p1_imp['opposing_team'] = np.where(df_p1_imp.teamId == df_p1_imp.home_teamId, df_p1_imp.away_teamId,
                                          df_p1_imp.home_teamId)
    df_p1_imp = df_p1_imp.drop(['home_teamId', 'away_teamId'], axis=1)
    df_p1_imp = df_p1_imp[['playerId', 'matchId', 'teamId', 'zone_1_pl', 'zone_2_pl', 'zone_3_pl',
                           'zone_4_pl', 'zone_5_pl', 'zone_6_pl', 'zone_1_t', 'zone_2_t',
                           'zone_3_t', 'zone_4_t', 'zone_5_t', 'zone_6_t', 'zone_1_imp',
                           'zone_2_imp', 'zone_3_imp', 'zone_4_imp', 'zone_5_imp', 'zone_6_imp',
                           'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'opposing_team', 'dist']]
    df_p1_imp = pd.merge(df_p1_imp, df_net_oi, left_on=['matchId', 'opposing_team'], right_on=['matchId', 'teamId'])
    df_p1_imp['jdi_zone_1'] = df_p1_imp.zone_1_imp * df_p1_imp.zone_1_net_oi * df_p1_imp.dist
    df_p1_imp['jdi_zone_2'] = df_p1_imp.zone_2_imp * df_p1_imp.zone_2_net_oi * df_p1_imp.dist
    df_p1_imp['jdi_zone_3'] = df_p1_imp.zone_3_imp * df_p1_imp.zone_3_net_oi * df_p1_imp.dist
    df_p1_imp['jdi_zone_4'] = df_p1_imp.zone_4_imp * df_p1_imp.zone_4_net_oi * df_p1_imp.dist
    df_p1_imp['jdi_zone_5'] = df_p1_imp.zone_5_imp * df_p1_imp.zone_5_net_oi * df_p1_imp.dist
    df_p1_imp['jdi_zone_6'] = df_p1_imp.zone_6_imp * df_p1_imp.zone_6_net_oi * df_p1_imp.dist
    df_p1_imp[
        'jdi'] = df_p1_imp.jdi_zone_1 + df_p1_imp.jdi_zone_2 + df_p1_imp.jdi_zone_3 + df_p1_imp.jdi_zone_4 + df_p1_imp.jdi_zone_5 + df_p1_imp.jdi_zone_6

    df_p2_imp = df_player_share[(df_player_share.playerId == p2) & (df_player_share.matchId == match)]
    df_p2_imp = pd.merge(df_p2_imp, net_oi_g, on=(['matchId', 'teamId']))
    df_p2_imp['dist'] = [67.94878]
    df_p2_imp = df_p2_imp.merge(df_matches_all, on=['matchId'])
    # Determine the opposing team in order for us to check whether opposing team over or underperforms in an area
    df_p2_imp['opposing_team'] = np.where(df_p2_imp.teamId == df_p2_imp.home_teamId, df_p2_imp.away_teamId,
                                          df_p2_imp.home_teamId)
    df_p2_imp = df_p2_imp.drop(['home_teamId', 'away_teamId'], axis=1)
    df_p2_imp = df_p2_imp[['playerId', 'matchId', 'teamId', 'zone_1_pl', 'zone_2_pl', 'zone_3_pl',
                           'zone_4_pl', 'zone_5_pl', 'zone_6_pl', 'zone_1_t', 'zone_2_t',
                           'zone_3_t', 'zone_4_t', 'zone_5_t', 'zone_6_t', 'zone_1_imp',
                           'zone_2_imp', 'zone_3_imp', 'zone_4_imp', 'zone_5_imp', 'zone_6_imp',
                           'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'opposing_team', 'dist']]
    df_p2_imp = pd.merge(df_p2_imp, df_net_oi, left_on=['matchId', 'opposing_team'], right_on=['matchId', 'teamId'])
    df_p2_imp['jdi_zone_1'] = df_p2_imp.zone_1_imp * df_p2_imp.zone_1_net_oi * df_p2_imp.dist
    df_p2_imp['jdi_zone_2'] = df_p2_imp.zone_2_imp * df_p2_imp.zone_2_net_oi * df_p2_imp.dist
    df_p2_imp['jdi_zone_3'] = df_p2_imp.zone_3_imp * df_p2_imp.zone_3_net_oi * df_p2_imp.dist
    df_p2_imp['jdi_zone_4'] = df_p2_imp.zone_4_imp * df_p2_imp.zone_4_net_oi * df_p2_imp.dist
    df_p2_imp['jdi_zone_5'] = df_p2_imp.zone_5_imp * df_p2_imp.zone_5_net_oi * df_p2_imp.dist
    df_p2_imp['jdi_zone_6'] = df_p2_imp.zone_6_imp * df_p2_imp.zone_6_net_oi * df_p2_imp.dist
    df_p2_imp['jdi'] = df_p2_imp.jdi_zone_1 + df_p2_imp.jdi_zone_2 + df_p2_imp.jdi_zone_3 + df_p2_imp.jdi_zone_4 + df_p2_imp.jdi_zone_5 + df_p2_imp.jdi_zone_6

    tester = df_jdi[(df_jdi.p1 == p1) & (df_jdi.p2 == p2) & (df_jdi.matchId == match) | (df_jdi.p1 == p2) & (df_jdi.p2 == p1) & (df_jdi.matchId == match) ]

    df_p1_imp = df_p1_imp.drop(
        ['teamId_y', 'zone_1_x', 'zone_2_x', 'zone_3_x', 'zone_4_x', 'zone_5_x', 'zone_6_x', 'zone_1_y', 'zone_2_y',
         'zone_3_y', 'zone_4_y', 'zone_5_y', 'zone_6_y', 'dist', 'zone_1_cumsum', 'zone_2_cumsum', 'zone_3_cumsum',
         'zone_4_cumsum', 'zone_5_cumsum', 'zone_6_cumsum', 'games_played', 'zone_1_expected_vaep',
         'zone_2_expected_vaep', 'zone_3_expected_vaep', 'zone_4_expected_vaep', 'zone_5_expected_vaep',
         'zone_6_expected_vaep'], axis=1)
    df_p2_imp = df_p2_imp.drop(
        ['teamId_y', 'zone_1_x', 'zone_2_x', 'zone_3_x', 'zone_4_x', 'zone_5_x', 'zone_6_x', 'zone_1_y', 'zone_2_y',
         'zone_3_y', 'zone_4_y', 'zone_5_y', 'zone_6_y', 'dist', 'zone_1_cumsum', 'zone_2_cumsum', 'zone_3_cumsum',
         'zone_4_cumsum', 'zone_5_cumsum', 'zone_6_cumsum', 'games_played', 'zone_1_expected_vaep',
         'zone_2_expected_vaep', 'zone_3_expected_vaep', 'zone_4_expected_vaep', 'zone_5_expected_vaep',
         'zone_6_expected_vaep'], axis=1)
    df_p1_imp = df_p1_imp.rename(columns={'teamId_x': 'teamId'})
    df_p2_imp = df_p2_imp.rename(columns={'teamId_x': 'teamId'})
    check = tester[df_p1_imp.columns]
    concatted = pd.concat([check, df_p1_imp, df_p2_imp])
    return concatted
