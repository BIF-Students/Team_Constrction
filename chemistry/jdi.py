import numpy as np


'''
This method is responsible for computing JDI between two players in a game
The method takes two dataframes as arguments:
    1. A Dataframe containing all values related to each player with regard to JDI needed components
    2. A Dataframe containing all successfull defensive actions performed by a player in match
The method returns a dataframe with a column describing the pairwaise JDI scores
'''
def jdi_compute(df, def_suc):
    # Compute the total number of team actions in each zone
    df['zones_total'] = np.sum(df[['zone_1_team', 'zone_2_team', 'zone_3_team', 'zone_4_team', 'zone_5_team', 'zone_6_team']], axis=1)

    # Compute the sum of player 1 involvement in each zone
    df['p1_sum'] = np.sum(df[['zone_1_pl1', 'zone_2_pl1', 'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1']],axis=1)

    # Compute the sum of player 2 involvement in each zone
    df['p2_sum'] = np.sum(df[['zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2', 'zone_5_pl2', 'zone_6_pl2']],axis=1)

    # Compute the pairwise involvement of the player pair
    df['pairwise_involvement'] = ((df.p1_sum + df.p2_sum) / df.zones_total)

    # Compute the JDI for zone 1
    df['jdi_zone_1'] = np.where(df.zone_1_team > 15, (((((((df.zone_6_net_oi * df.zone_1_imp1)) + ((df.zone_6_net_oi * df.zone_1_imp2))))))),0)

    # Compute the JDI for zone 2
    df['jdi_zone_2'] = np.where(df.zone_2_team > 15, ((((((df.zone_5_net_oi * df.zone_2_imp1)) + ((df.zone_5_net_oi * df.zone_2_imp2)))))), 0)

    # Compute the JDI for zone 3
    df['jdi_zone_3'] = np.where(df.zone_3_team > 15, ((((((df.zone_4_net_oi * df.zone_3_imp1)) + ((df.zone_4_net_oi * df.zone_3_imp2)))))), 0)

    # Compute the JDI for zone 4
    df['jdi_zone_4'] = np.where(df.zone_4_team > 15, ((((((df.zone_3_net_oi * df.zone_4_imp1)) + ((df.zone_3_net_oi * df.zone_4_imp2)))))),  0)

    # Compute the JDI for zone 5
    df['jdi_zone_5'] = np.where(df.zone_5_team > 15, ((((((df.zone_2_net_oi * df.zone_5_imp1)) + ((df.zone_2_net_oi * df.zone_5_imp2)))))),  0)

    # Compute the JDI for zone 6
    df['jdi_zone_6'] = np.where(df.zone_6_team > 15,  ((((((df.zone_1_net_oi * df.zone_6_imp1)) + ((df.zone_1_net_oi * df.zone_6_imp2)))))),  0)

    # Merge the defensive success DataFrame for player 1
    df = df.merge(def_suc, left_on=['playerId1', 'matchId'], right_on=['playerId', 'matchId'], how='left')

    # Merge the defensive success DataFrame for player 2
    df = df.merge(def_suc, left_on=['playerId2', 'matchId'], right_on=['playerId', 'matchId'], how='left')

    # Fill missing values in defensive actions with 0
    df['def_action_x'] = df['def_action_x'].fillna(0)
    df['def_action_y'] = df['def_action_y'].fillna(0)

    # Compute the final JDI considering defensive actions and distance
    df['jdi'] = (((df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6) * ( df.def_action_x + df.def_action_y)) / df.distance)
    return df


'''
This method is responsible for conducting the JDI computation
It takes 5 dataframes as arguments in the signature:
    1. A dataframe containing observations with the responsibility share of each player in a match in a zone
    2. A dataframe containing the average distance between pairs of players in all games across a season.
    3. A dataframe containing the net offensive impact of all teams across all matches in the season 21/22
    4. A dataframe containing information of all matches provided in the database of Brondby
    5. A dataframe contaning alle successful defensive action performed by players in differet matches
The methods returns a dataframe with observatoins describing the JDI produced by pairs of players across all matches in a season
'''
def get_jdi(player_shares, player_distances, df_net_oi, matches_all, def_suc):
    # Merge player shares on matchId and teamId
    df_player_share = player_shares.merge(player_shares, on=['matchId', 'teamId'], suffixes=(1, 2))

    # Filter out rows where playerId1 is equal to playerId2
    df_player_share = df_player_share[df_player_share['playerId1'] != df_player_share['playerId2']]

    # Create a unique identifier for each player pair using matchId, playerId1, and playerId2
    df_player_share['id'] = df_player_share.apply( lambda row: tuple(sorted([row['matchId'], row['playerId1'], row['playerId2']])), axis=1)

    # Remove duplicates based on the unique identifier
    df_player_share = df_player_share.drop_duplicates(subset=['id'], keep='first')

    # Drop the 'id' column
    df_player_share = df_player_share.drop(['id'], axis=1)

    # Merge player shares and distances on matchId, teamId, playerId1, and playerId2
    df_player_share_dist = df_player_share.merge(player_distances, on=['matchId', 'teamId', 'playerId1', 'playerId2'])

    # Select the relevant columns for further analysis
    df_player_share_dist = df_player_share_dist[
        ['playerId1', 'matchId', 'teamId', 'zone_1_pl1', 'zone_2_pl1', 'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1',
         'zone_6_pl1', 'zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2', 'zone_5_pl2', 'zone_6_pl2', 'zone_1_t1',
         'zone_2_t1', 'zone_3_t1', 'zone_4_t1', 'zone_5_t1', 'zone_6_t1', 'zone_1_imp1', 'zone_2_imp1', 'zone_3_imp1',
         'zone_4_imp1', 'zone_5_imp1', 'zone_6_imp1', 'playerId2', 'zone_1_imp2', 'zone_2_imp2', 'zone_3_imp2',
         'zone_4_imp2', 'zone_5_imp2', 'zone_6_imp2', 'distance']]

    # Rename the 'zone_1_t1', 'zone_2_t1', 'zone_3_t1', 'zone_4_t1', 'zone_5_t1', 'zone_6_t1' columns to 'zone_1_team', 'zone_2_team', 'zone_3_team', 'zone_4_team', 'zone_5_team', 'zone_6_team', respectively
    df_player_share_dist = df_player_share_dist.rename(
        columns={'zone_1_t1': 'zone_1_team', 'zone_2_t1': 'zone_2_team', 'zone_3_t1': 'zone_3_team',
                 'zone_4_t1': 'zone_4_team', 'zone_5_t1': 'zone_5_team', 'zone_6_t1': 'zone_6_team'})

    # Merge df_net_oi and matches_all dataframes on matchId
    matches_noi_opp = df_net_oi.merge(matches_all, on='matchId')

    # Create a new column 'opposingTeam' based on the condition
    matches_noi_opp['opposingTeam'] = np.where(matches_noi_opp.teamId == matches_noi_opp.home_teamId,
                                               matches_noi_opp.away_teamId, matches_noi_opp.home_teamId)

    # Select the relevant columns from matches_noi_opp
    matches_noi_opp_m = matches_noi_opp[['matchId', 'teamId', 'opposingTeam']]

    # Select the relevant columns from matches_noi_opp
    matches_noi_opp = matches_noi_opp[
        ['matchId', 'teamId', 'zone_1_expected_vaep', 'zone_2_expected_vaep', 'zone_3_expected_vaep',
         'zone_4_expected_vaep', 'zone_5_expected_vaep', 'zone_6_expected_vaep', 'zone_1_net_oi', 'zone_2_net_oi',
         'zone_3_net_oi', 'zone_4_net_oi', 'zone_5_net_oi', 'zone_6_net_oi']]

    # Merge matches_noi_opp_m and matches_noi_opp on matchId and opposingTeam
    mno = matches_noi_opp_m.merge(matches_noi_opp, left_on=['matchId', 'opposingTeam'], right_on=['matchId', 'teamId'])
    mno = mno.drop(['teamId_y'], axis=1)
    mno = mno.rename(columns = {'teamId_x': 'teamId'})

    # Merge mno and matches_noi_opp_m on matchId and opposingTeam
    mno = mno.merge(matches_noi_opp_m, on=['matchId', 'opposingTeam'])
    mno = mno.drop(['teamId_y'], axis=1)
    mno = mno.rename(columns = {'teamId_x': 'teamId'})
    # Merge df_player_share_dist and mno on matchId and teamId
    df_jdi = df_player_share_dist.merge(mno, on=['matchId', 'teamId'])

    # Compute JDI using jdi_compute() function and store the result in df_jdi_v2
    df_jdi_v2 = jdi_compute(df_jdi, def_suc)

    # Select the relevant columns for the final result
    df_jdi_v2 = df_jdi_v2[
        ['matchId', 'teamId', 'playerId1', 'playerId2', 'pairwise_involvement', 'jdi_zone_1', 'jdi_zone_2',
         'jdi_zone_3', 'jdi_zone_4', 'jdi_zone_5', 'jdi_zone_6', 'jdi']]

    # Create 'p1' column with minimum player ID
    df_jdi_v2['p1'] = np.where(df_jdi_v2.playerId1 < df_jdi_v2.playerId2, df_jdi_v2.playerId1, df_jdi_v2.playerId2)

    # Create 'p2' column with maximum player ID
    df_jdi_v2['p2'] = np.where(df_jdi_v2.playerId2 > df_jdi_v2.playerId1, df_jdi_v2.playerId2, df_jdi_v2.playerId1)

    # Return the final dataframe df_jdi_v2
    return df_jdi_v2



