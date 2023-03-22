import numpy as np

from chemistry.chemistry_helpers import process_for_jdi, compute_jdi


def getJdi(df_net_oi, matches_all, ec_df, df_player_share):
    df_snoi_netoi_dist = process_for_jdi(df_net_oi, matches_all, ec_df, df_player_share)
    df_snoi_netoi_dist['p1'] = np.where(df_snoi_netoi_dist['player1'] > df_snoi_netoi_dist['player2'], df_snoi_netoi_dist['player2'], df_snoi_netoi_dist['player1'])
    df_snoi_netoi_dist['p2'] = np.where(df_snoi_netoi_dist['player1'] == df_snoi_netoi_dist['p1'], df_snoi_netoi_dist['player2'], df_snoi_netoi_dist['player1'])
    df_jdi = compute_jdi(df_snoi_netoi_dist)
    return df_jdi