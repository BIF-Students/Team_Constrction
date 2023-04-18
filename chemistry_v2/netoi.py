
from chemistry_v2.chemistry_helpers import *


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

    min_seasons = df_zones_vaep.groupby('competitionId')['seasonId'].min()
    max_seasons = df_zones_vaep.groupby('competitionId')['seasonId'].max()
    prev_season = df_zones_vaep[df_zones_vaep.apply(lambda x: x['seasonId'] in min_seasons.values, axis=1)]
    season_of_interest = df_zones_vaep[df_zones_vaep.apply(lambda x: x['seasonId'] in max_seasons.values, axis=1)]


    #Previous season average zone_vaep per team
    def_zones_vaep_team_season = prev_season.groupby(['teamId', 'seasonId', 'competitionId'], as_index=False).agg({
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
    def_zones_vaep_team_season = (def_zones_vaep_team_season[['teamId', 'seasonId', 'competitionId',
                                                            'zone_1_prior_avg',
                                                             'zone_2_prior_avg',
                                                             'zone_3_prior_avg',
                                                            'zone_4_prior_avg',
                                                             'zone_5_prior_avg',
                                                             'zone_6_prior_avg'
                                                             ]]).rename(columns = {'seasonId': 'previous_season'})

    #Sum values per game per zone
    df_zones_vaep = season_of_interest.groupby(['matchId', 'teamId', 'competitionId', 'seasonId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                        'zone_6': 'sum'})


    df_running_vaep_avg = get_running_tvg_v2(df_zones_vaep)

    #Here cumulative sums per game are added as columns
    #Further, columns with expected vaep values per zone are added, as an average of all
    # games preceding a particular game
    df_merged = df_running_vaep_avg.merge(def_zones_vaep_team_season, on =['teamId', 'competitionId'])

    #The netoi per game per zone is added by subtracting the expected vaep in a zone in a game with the actual
    #vaep produced in a game in a zone
    df_net_oi = compute_net_oi_game(df_merged)
    return df_net_oi


