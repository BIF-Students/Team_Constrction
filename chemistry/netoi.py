from chemistry.chemistry_helpers import find_zone_chemistry, find_zones_and_vaep, team_vaep_game, compute_net_oi_game

def getOi(df):
    #Copy inital df to dataframe of interest
    df_vaep_zone_match = df

    #Add column with zones related to each action
    df_vaep_zone_match['zone'] = df_vaep_zone_match.apply(lambda row: find_zone_chemistry(row), axis = 1)

    #Method used to make zone column to columns marking each zone and computing vaep per action in a particular zone
    def_zones_vaep = find_zones_and_vaep(df_vaep_zone_match)

    #Sum values per game per zone
    def_zones_vaep = def_zones_vaep.groupby(['matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                       'zone_6': 'sum'})
    #Here cumulative sums per game are added as columns
    #Further, columns with expected vaep values per zone are added, as an average of all
    # games preceding a particular game
    df_running_vaep_avg = team_vaep_game(def_zones_vaep)

    #The netoi per game per zone is added by subtracting the expected vaep in a zone in a game with athe actual
    # vaep produced in a game in a zne
    df_net_oi = compute_net_oi_game(df_running_vaep_avg)
    return df_net_oi

