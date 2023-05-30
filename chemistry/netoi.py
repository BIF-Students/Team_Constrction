import numpy as np

'''
This method computes the net offensive impact for each zone in a DataFrame based on the number of games played.
It takes one argument:
    1: A dataFrame containing the input data
The method returns an updated DataFrame with the net offensive impact for each zone
'''
def compute_net_oi_game(df):
    # make a copy of the input dataframe to avoid modifying it directly
    copy = df

    # compute net offensive impact for each zone based on number of games played
    copy['zone_1_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_1_expected_vaep - copy.zone_1_prior_avg),
                                     (copy.zone_1_expected_vaep - copy.zone_1))
    copy['zone_2_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_2_expected_vaep - copy.zone_2_prior_avg),
                                     (copy.zone_2_expected_vaep - copy.zone_2))
    copy['zone_3_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_3_expected_vaep - copy.zone_3_prior_avg),
                                     (copy.zone_3_expected_vaep - copy.zone_3))
    copy['zone_4_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_4_expected_vaep - copy.zone_4_prior_avg),
                                     (copy.zone_4_expected_vaep - copy.zone_4))
    copy['zone_5_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_5_expected_vaep - copy.zone_5_prior_avg),
                                     (copy.zone_5_expected_vaep - copy.zone_5))
    copy['zone_6_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_6_expected_vaep - copy.zone_6_prior_avg),
                                     (copy.zone_6_expected_vaep - copy.zone_6))

    # return the updated dataframe with net offensive impact for each zone
    return copy




#Method for computing running aver vaep using observatoins as input and with a label input
def compute_running_avg_team_vaep (row, label):
    #If it is the first game of the season, we use the current cumulative sum
    if row['games_played'] == 0:
        return row[label]
    else: # Else we divide the current sum with the amout of games played
        return row[label] / (row['games_played'] +1)



'''
This method calculates various metrics related to team VAEP (Value Added) for each zone.
It takes one argument:
    1. A DataFrame containing team data
The method return an updated DataFrame with various team VAEP metrics for each zone
'''
def team_vaep_game_oi(df):
    # Sort the data by teamId and matchId
    df = df.sort_values(by=['teamId', 'matchId'])

    # Compute the cumulative sum of each zone for each team
    df[['zone_1_cumsum', 'zone_2_cumsum',
        'zone_3_cumsum', 'zone_4_cumsum',
        'zone_5_cumsum', 'zone_6_cumsum'
        ]] = df.groupby('teamId')[['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6']].cumsum()

    # Compute the number of games played by each team
    df['games_played'] = df.groupby(['teamId']).cumcount()

    # Compute the running average of expected VAEP for each zone
    df['zone_1_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_1_cumsum'), axis=1)
    df['zone_2_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_2_cumsum'), axis=1)
    df['zone_3_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_3_cumsum'), axis=1)
    df['zone_4_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_4_cumsum'), axis=1)
    df['zone_5_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_5_cumsum'), axis=1)
    df['zone_6_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_6_cumsum'), axis=1)

    # Return the updated dataframe
    return df


'''
This method is responsible for computing the net offensive impact in a zone across a match in a particular season
The method takes one argument
    1. A dataframe containing an event stream across two seasons 20/21 and 21/22
    The method returns a dataframe with observations with information on the neto offensive impact of a team in a match in all zones
'''

def find_zones_and_vaep_oi(df):
    # create dummy variables for the zone column
    df['zone_1'] = np.where(df.zone == 1, df.sumVaep, 0)
    df['zone_2'] = np.where(df.zone == 2, df.sumVaep, 0)
    df['zone_3'] = np.where(df.zone == 3, df.sumVaep, 0)
    df['zone_4'] = np.where(df.zone == 4, df.sumVaep, 0)
    df['zone_5'] = np.where(df.zone == 5, df.sumVaep, 0)
    df['zone_6'] = np.where(df.zone == 6, df.sumVaep, 0)

    return df



'''
This method is used to determine in which zone an action has happened - this is used in a lambda expression
The method takes one argument 
    1. A dataframe with one row
The name of the zone is returned as an integer
'''
def find_zone_chemistry_oi(frame):
    s = ""  # Initialize variable s to an empty string
    x = frame['x']  # Extract the value of x from the input row
    y = frame['y']  # Extract the value of y from the input row

    # Check which zone the (x, y) pair belongs to
    if x >= 0 and x < 50 and y >= 0 and y < 33:
        s = 1  # Assign s to 1 if the pair belongs to zone 1
    elif x >= 50 and x <= 100 and y >= 0 and y < 33:
        s = 4  # Assign s to 4 if the pair belongs to zone 4
    elif x >= 0 and x < 50 and y >= 33 and y < 66:
        s = 2  # Assign s to 2 if the pair belongs to zone 2
    elif x >= 50 and x <= 100 and y >= 33 and y < 66:
        s = 5  # Assign s to 5 if the pair belongs to zone 5
    elif x >= 0 and x < 50 and y >= 66 and y <= 100:
        s = 3  # Assign s to 3 if the pair belongs to zone 3
    elif x >= 50 and x <= 100 and y >= 66 and y <= 100:
        s = 6  # Assign s to 6 if the pair belongs to zone 6
    else:
        s = 0  # Assign s to 0 if the (x, y) pair doesn't belong to any zone

    return s  # Return the zone number that the (x, y) pair belongs to


def getOi(df):
    # Copy inital df to dataframe of interest
    df_vaep_zone_match = df

    # Add column with zones related to each action
    df_vaep_zone_match['zone'] = df_vaep_zone_match.apply(lambda row: find_zone_chemistry_oi(row), axis=1)
    df_vaep_zone_match = df_vaep_zone_match[(~df_vaep_zone_match.typePrimary.isin(
        ['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))]

    print("ramt")
    # Method used to make zone column to columns marking each zone and computing vaep per action in a particular zone
    df_zones_vaep = find_zones_and_vaep_oi(df_vaep_zone_match)
    print("ramt")

    # Extract data from season 20/21
    df_20_21 = df_zones_vaep[df_zones_vaep['seasonId'] == min(df_zones_vaep.seasonId)]

    # Extract data from season 21/22
    df_21_22 = df_zones_vaep[df_zones_vaep['seasonId'] == max(df_zones_vaep.seasonId)]

    #Compute mean offensive vvalue for all matches in all zones in the 20/21 season
    def_zones_vaep_team_season = df_20_21.groupby(['teamId', 'seasonId'], as_index=False).agg({
        'zone_1': 'mean',
        'zone_2': 'mean',
        'zone_3': 'mean',
        'zone_4': 'mean',
        'zone_5': 'mean',
        'zone_6': 'mean'})
    #rename columns to more sensible names
    def_zones_vaep_team_season = def_zones_vaep_team_season.rename(columns={
        'zone_1': 'zone_1_prior_avg',
        'zone_2': 'zone_2_prior_avg',
        'zone_3': 'zone_3_prior_avg',
        'zone_4': 'zone_4_prior_avg',
        'zone_5': 'zone_5_prior_avg',
        'zone_6': 'zone_6_prior_avg'
    })
    #Zoom in on specific columns based on relevance
    def_zones_vaep_team_season = def_zones_vaep_team_season[['teamId',
                                                             'zone_1_prior_avg',
                                                             'zone_2_prior_avg',
                                                             'zone_3_prior_avg',
                                                             'zone_4_prior_avg',
                                                             'zone_5_prior_avg',
                                                             'zone_6_prior_avg'
                                                             ]]

    # Sum values per game per zone
    df_zones_vaep = df_21_22.groupby(['matchId', 'teamId'], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum'})

    # Here cumulative sums per game are added as columns
    # Further, columns with expected vaep values per zone are added, as an average of all
    # games preceding a particular game
    df_running_vaep_avg = team_vaep_game_oi(df_zones_vaep)
    df_merged = df_running_vaep_avg.merge(def_zones_vaep_team_season, on='teamId')

    # The netoi per game per zone is added by subtracting the expected vaep in a zone in a game with the actual
    # vaep produced in a game in a zone
    df_net_oi = compute_net_oi_game(df_merged)
    return df_net_oi
