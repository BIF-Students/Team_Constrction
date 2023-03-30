from chemistry.chemistry_helpers import find_zone_chemistry, find_zones_and_vaep, team_vaep_game, compute_net_oi_game

def getOi(df):
    #Copy inital df to dataframe of interest
    df_vaep_zone = df

    df_vaep_zone_match = df

    # Add column with zones related to each action
    df_vaep_zone_match['zone'] = df_vaep_zone_match.apply(lambda row: find_zone_chemistry(row), axis=1)

    df_vaep_zone_match = df_vaep_zone_match[(~df_vaep_zone_match.typePrimary.isin(
        ['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))]


    # Method used to make zone column to columns marking each zone and computing vaep per action in a particular zone
    df_zones_vaep = find_zones_and_vaep(df_vaep_zone_match)

    df_20_21 = df_zones_vaep[df_zones_vaep['seasonId'] == min(df_zones_vaep.seasonId)]
    df_21_22 = df_zones_vaep[df_zones_vaep['seasonId'] == max(df_zones_vaep.seasonId)]

    #Sum values per game per zone
    def_zones_vaep_team_season = df_20_21.groupby(['teamId', 'seasonId'], as_index=False).agg({
                                                                        'zone_1':'mean',
                                                                        'zone_2':'mean',
                                                                        'zone_3': 'mean',
                                                                        'zone_4': 'mean',
                                                                        'zone_5': 'mean',
                                                                       'zone_6': 'mean'})
    def_zones_vaep_team_season = def_zones_vaep_team_season.rename(columns = {
                                                                   'zone_1': 'zone_1_prior_avg' ,
                                                                   'zone_2': 'zone_2_prior_avg' ,
                                                                   'zone_3': 'zone_3_prior_avg' ,
                                                                   'zone_4': 'zone_4_prior_avg' ,
                                                                   'zone_5': 'zone_5_prior_avg' ,
                                                                   'zone_6': 'zone_6_prior_avg'
                                                                              })
    def_zones_vaep_team_season = def_zones_vaep_team_season[['teamId',
                                                             'zone_1_prior_avg',
                                                             'zone_2_prior_avg',
                                                             'zone_3_prior_avg',
                                                             'zone_4_prior_avg',
                                                             'zone_5_prior_avg',
                                                             'zone_6_prior_avg'
                                                             ]]

    #Sum values per game per zone
    df_zones_vaep = df_21_22.groupby(['matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                       'zone_6': 'sum'})


    #Here cumulative sums per game are added as columns
    #Further, columns with expected vaep values per zone are added, as an average of all
    # games preceding a particular game
    df_running_vaep_avg = team_vaep_game(df_zones_vaep)
    df_merged = df_running_vaep_avg.merge(def_zones_vaep_team_season, on ='teamId')

    #The netoi per game per zone is added by subtracting the expected vaep in a zone in a game with the actual
    #vaep produced in a game in a zone
    df_net_oi = compute_net_oi_game(df_merged)
    return df_net_oi
