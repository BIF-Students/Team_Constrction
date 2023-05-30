#Import of necessary libraries
from matplotlib import pyplot as plt
from plotly.offline import offline
from sklearn.preprocessing import MinMaxScaler
from chemistry.jdi import get_jdi
from chemistry.joi import get_joi
from chemistry.responsibility_share import *
from chemistry.sql_statements import *
from chemistry.distance import *
from chemistry.netoi import *
from datetime import date, datetime
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px


'''
This method is used to determine in which zone an action has happened - this is used in a lambda expression
The method takes one argument 
    1. A dataframe with one row
The name of the zone is returned as an integer'''
def find_zone_chemistry_pred(frame):
    s = ""  # Initialize variable s to an empty string
    x = (frame['spatial_pos_y'])[0]  # Extract the value of x from the input row
    y = (frame['spatial_pos_y'])[1]  # Extract the value of y from the input row

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



'''
This method is used to create a DataFrame containing defensive actions based on successful duels.
It takes two arguments:
    1. ground_duels: DataFrame containing ground duels data
    2. air_duels: DataFrame containing air duels data
The resulting DataFrame contains the event ID, match ID, and defensive action information for successful duels.
The method returns the created DataFrame.
'''
def create_def_succes_frame(ground_duels, air_duels):
    # Select relevant columns from ground duels DataFrame for possession recovery
    ground_duels_suc_rec = ground_duels[['eventId', 'matchId', 'groundDuel_recoveredPossession']]

    # Select relevant columns from ground duels DataFrame for stopping progress
    ground_duels_suc_stop = ground_duels[['eventId', 'matchId', 'groundDuel_stoppedProgress']]

    # Filter ground duels DataFrame for possession recovery and rename column
    ground_duels_suc_rec_1 = ground_duels_suc_rec[
        ground_duels_suc_rec['groundDuel_recoveredPossession'] == True].rename(
        columns={'groundDuel_recoveredPossession': 'def_action'})

    # Filter ground duels DataFrame for stopping progress and rename column
    ground_duels_suc_stop_1 = ground_duels_suc_stop[ground_duels_suc_stop['groundDuel_stoppedProgress'] == True].rename(
        columns={'groundDuel_stoppedProgress': 'def_action'})

    # Filter air duels DataFrame for first touch and rename column
    air_duels_suc = air_duels[air_duels['aerialDuel_firstTouch'] == True].rename(
        columns={'aerialDuel_firstTouch': 'def_action'})

    # Concatenate all the filtered DataFrames into a single DataFrame
    df_succes_tot = pd.concat(
        [ground_duels_suc_rec_1, ground_duels_suc_stop_1, air_duels_suc[['eventId', 'matchId', 'def_action']]])

    # Replace 'True' values in the 'def_action' column with 1
    df_succes_tot['def_action'] = df_succes_tot['def_action'].replace(True, 1)

    return df_succes_tot




'''
This method generates a DataFrame containing player chemistry ability based on input DataFrame.
It takes one argument:
    1. A DataFrame containing player chemistry information
The method returns the DataFrame containing player chemistry ability for each player.
'''
def generate_chemistry_ability(df):
    df1 = df[['p1','shortName_x', 'role_name_x','areaName_x', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p1':'playerId', 'shortName_x': 'shortName', 'role_name_x': 'role_name' ,'areaName_x': 'areaName'})
    df2 = df[['p2', 'shortName_y', 'role_name_y','areaName_y', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p2':'playerId','shortName_y': 'shortName', 'role_name_y': 'role_name' ,'areaName_y': 'areaName'})
    players_and_chemistry = pd.concat([df1, df2])
    players_and_chemistry_season = players_and_chemistry.groupby(['playerId', 'shortName',  'role_name' , 'areaName'], as_index=False)['chemistry'].sum().reset_index(drop=True)

    return players_and_chemistry_season



'''
This method processes data to calculate net offensive impact and perform various merges and transformations.
It takes four arguments:
    1. A dataframe containing net offensive impact data
    2. A dataFrame containing match data
    3. A DataFrame containing distance data
    4. A DataFrame containing player share data

The method return a final DataFrame after processing and merging with data on the responsibility shared of paris of players in a game
'''
def process_for_jdi (df_net_oi, matches_all, distances_df, df_player_share):
    #MAke a copy
    df_net_oi_v2 = df_net_oi[['matchId', 'teamId',
                              'zone_1_net_oi', 'zone_2_net_oi',
                              'zone_3_net_oi', 'zone_4_net_oi',
                              'zone_5_net_oi', 'zone_6_net_oi'
                              ]]
    #Merge with mataches to discover home and away teams
    df_net_oi_v2 = pd.merge(df_net_oi_v2, matches_all, on='matchId')

    #Determine the opposing team in order for us to check whether opposing team over or underperforms in an area
    df_net_oi_v2['opposing_team'] = np.where(df_net_oi_v2.teamId == df_net_oi_v2.home_teamId, df_net_oi_v2.away_teamId,df_net_oi_v2.home_teamId)

    #Remove irrelevant columns
    df_net_oi_v2 = df_net_oi_v2.drop(['home_teamId', 'away_teamId'], axis=1)

    #Copy of df
    duplicate_for_merge = df_net_oi_v2.copy()

    #merge the two dataframes to get who is the opposing team for each game and what are their net oi
    merged_opposing_vaep_values = pd.merge(df_net_oi_v2, duplicate_for_merge, left_on=(['matchId', 'teamId']),right_on=(['matchId', 'opposing_team']))


    #Remove unusable columns
    merged_opposing_vaep_values = merged_opposing_vaep_values.drop(['teamId_y', 'zone_1_net_oi_x', 'zone_2_net_oi_x',
                                                                    'zone_3_net_oi_x', 'zone_4_net_oi_x',
                                                                    'zone_5_net_oi_x', 'zone_6_net_oi_x',
                                                                    'opposing_team_y'
                                                                    ], axis=1)

    #Assign appropriate feature names
    merged_opposing_vaep_values = merged_opposing_vaep_values.rename(
        columns={'teamId_x': 'teamId', 'opposing_team_x': 'opposing_team',
                 'zone_1_net_oi_y': 'zone_1_net_oi', 'zone_2_net_oi_y': 'zone_2_net_oi',
                 'zone_3_net_oi_y': 'zone_3_net_oi', 'zone_4_net_oi_y': 'zone_4_net_oi',
                 'zone_5_net_oi_y': 'zone_5_net_oi', 'zone_6_net_oi_y': 'zone_6_net_oi'})

    #Merge to obtain net oi in each dataframe and share for each player
    df_snoi_netoi = pd.merge(merged_opposing_vaep_values, df_player_share, on=(['matchId', 'teamId']), how='inner')

    #merge with distances
    df_share_dist = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'playerId1']), how='inner')
    df_share_dist_2 = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'playerId2']), how='inner')
    df_share_dist_final = pd.concat([df_share_dist, df_share_dist_2])
    return df_share_dist_final


'''
This method performs pairwise calculations on playing time data.
It takes one argument:
    1. A dataFrame containing data about playing time in games across a season
The method returns a dataframe with info on pairwise playing time of two individuals
'''
def pairwise_playing_time(df):
    # Merge the DataFrame with itself based on the matchId and teamId columns
    paired = df.merge(df, on=(['matchId', 'teamId']), how='inner', suffixes=('1', '2'))
    # Remove any rows where playerId1 is equal to playerId2
    paired = paired[paired.playerId1 != paired.playerId2]
    # Create new columns 'p1' and 'p2' to keep the IDs of the two paired players
    paired['id'] = paired.apply(lambda row: tuple(sorted([row['matchId'], row['playerId1'], row['playerId2']])), axis=1)
    paired = paired.drop_duplicates(subset=['id'], keep='first')
    paired = paired.drop(['id'], axis=1)
    paired['p1'] = np.where(paired.playerId1 < paired.playerId2, paired.playerId1, paired.playerId2)
    paired['p2'] = np.where(paired.playerId2 > paired.playerId1, paired.playerId2, paired.playerId1)

    # Set the 'minutes' column to the smaller of minutes1 and minutes2
    paired['minutes'] = np.where(paired.minutes1 < paired.minutes2, paired.minutes1, paired.minutes2)
    # Keep only the relevant columns in the resulting DataFrame
    paired = paired[['teamId', 'matchId','p1', 'p2', 'minutes']]
    # Remove any duplicate rows in the DataFrame
    paired = paired.drop_duplicates()
    # Group the DataFrame by teamId, p1, and p2, and sum the minutes column
    paired = paired.groupby(['teamId', 'p1','p2'], as_index=False )['minutes'].sum()
    return paired


'''
This method computes normalized values based on the given DataFrames.
It takes three arguments:
    1. A dataFrame containing 'joi' values for player pairs in each team for each game
    2. A dataFrame containing 'jdi' values for player pairs in each team for each game
    3. A dataFrame containing pairwise playing time data

The method returns a dataframe with joi and jdi as per-90-minutes-played while filtering for players with less than 8 games of pairwise playing time
'''
def compute_normalized_values(df_joi_game, df_jdi_game, df_pairwise_time):
    # Aggregate sum of 'joi' for each player pair in each team for the season
    df_joi_season = df_joi_game.groupby(['p1', 'p2', 'teamId'], as_index = False).agg({
                                                                                     'joi': 'sum',
                                                                                     'goals': 'sum',
                                                                                     'assists': 'sum',
                                                                                     'second_assists': 'sum'
                                                                                     })
    # Aggregate sum of 'jdi' for each player pair in each team for the season
    df_jdi_season = df_jdi_game.groupby(['p1', 'p2', 'teamId'], as_index = False)['jdi'].sum()
    # Merge the two dataframes based on the player pairs
    df_merged = pd.merge(df_joi_season, df_jdi_season, on=['p1', 'p2'])
    df_merged = df_merged.merge(df_pairwise_time, on=['p1', 'p2'])

    # Compute the normalized value of 'joi' and 'jdi' by dividing them by the norm90 value
    df_merged['winners90'] =  (df_merged.assists + df_merged.goals) * 90/df_merged.minutes
    df_merged['df_joi90'] = (df_merged.joi * 90/df_merged.minutes) #* df_merged.winners90
    df_merged['df_jdi90'] = df_merged.jdi * 90/df_merged.minutes
    df_merged = df_merged[['p1', 'p2', 'teamId', 'joi', 'jdi', 'minutes', 'df_jdi90', 'df_joi90', 'winners90']]

    # Filter out player pairs who have played less than 300 minutes together
    df_merged = df_merged.query('minutes >= 720')
    return df_merged


'''
This function calculates the TVP (Team Value Per 90) for each player in a squad based on the given DataFrames.
It takes three arguments:
     1. A dataFrame containing VAEP values for events in matches
     2. A dataFrame containing information about players in the starting eleven and substitutions during a game
     3. A dataFrame containing with timestamps related to match events
The method returns A dataFrame with calculated TVP values for each player in the squad
'''
def get_TVP(df_vaep, squad, stamps):
    # Select only necessary columns from 'stamps' dataframe
    stamps = stamps[['eventId', 'matchPeriod', 'minute', 'second', 'matchTimestamp']]
    # Merge 'df_vaep' and 'stamps' dataframes based on 'eventId'
    df_vaep = df_vaep.merge(stamps, on=('eventId'))
    # Convert 'minutes' column to integer type
    squad['minutes_passed'] = squad['minutes'].astype(int)
    # Calculate 'seconds_passed_within_minute' column based on 'minutes_passed'
    squad['seconds_passed_within_minute'] = ((squad['minutes'] - squad['minutes_passed']) * 60).astype(float)

    player_TVG = []
    # Loop through each row of 'squad' dataframe
    for i, row in squad.iterrows():
        # Select rows from 'df_vaep' dataframe based on match and team ID
        g_vaep = df_vaep.loc[(df_vaep['matchId'] == row['matchId']) & (df_vaep['teamId'] == row['teamId'])]
        # Select only rows where the event happened before or within the current minute
        g_vaep = g_vaep.loc[(g_vaep['minute'] < (row['minutes_passed'] +1)) & ((g_vaep['minute'] < row['minutes_passed']) | (g_vaep['minute'] == row['minutes_passed']) & (g_vaep['second'] <= row['seconds_passed_within_minute']))]
        # Append a list of values to 'player_TVG' list
        player_TVG.append([row.playerId, row.matchId, row.teamId, row.minutes_passed, row.seconds_passed_within_minute, sum(g_vaep['sumVaep'])])
    # Create a new dataframe 'frame' with columns and data from 'player_TVG'
    frame = pd.DataFrame(data=player_TVG, columns=['playerId', 'matchId', 'teamId', 'minutes_played', 'seconds_in_minute', 'in_game_team_vaep'])
    # Group 'frame' dataframe by 'teamId' and 'playerId' and calculate mean of 'in_game_team_vaep' for each group
    team_vaep_p_in_game = frame.groupby(['teamId', 'playerId'], as_index=False).agg({'minutes_played': 'sum', 'seconds_in_minute': 'sum', 'in_game_team_vaep': 'sum'})
    team_vaep_p_in_game['seconds_to_minutes'] = team_vaep_p_in_game['seconds_in_minute'] / 60
    team_vaep_p_in_game['total_minutes'] = team_vaep_p_in_game['minutes_played'] + team_vaep_p_in_game['seconds_to_minutes']
    team_vaep_p_in_game['in_game_team_vape_per_90'] = team_vaep_p_in_game['in_game_team_vaep'] / (team_vaep_p_in_game['total_minutes'] / 90)
    return team_vaep_p_in_game


'''
This function prepares the data for scaling by calculating TVP (Team Value Per 90) and league average VAEP values.
It takes four arguments:
    1. DataFrame containing VAEP values for events in matches
    2. squad: DataFrame containing information about players in a squad
    3. stamps: DataFrame containing information about match events
    .4 match_duration: Duration of a match in minutes


Parameters:
- df (DataFrame): DataFrame containing VAEP values for events in matches
- squad (DataFrame): DataFrame containing information about players in a squad
- stamps (DataFrame): DataFrame containing information about match events
- match_duration (int): Duration of a match in minutes

Returns:
- merged (DataFrame): Prepared DataFrame for scaling with player pairings and factor values
'''


def prepare_for_scaling(df, squad, stamps, match_duration):
    tvp = get_TVP(df, squad, stamps)
    league_avg_vaep_per_90 = get_league_vaep(df, match_duration)
    tvp['league_vaep_per_90'] = league_avg_vaep_per_90  # Place league vaep as column
    tvp['factor'] = tvp['in_game_team_vape_per_90'] / tvp['league_vaep_per_90']  # Compute global factor values

    merged = tvp.merge(tvp, on='teamId', suffixes=('1', '2'))  # Pair players
    merged = merged[['teamId', 'playerId1', 'playerId2', 'factor1', 'factor2']]
    # Filter columns for pairings of players with themselves and establish an order
    # by having the player with the smallest id in the p1 column
    merged = merged[(merged.playerId1 != merged.playerId2) & (merged.playerId1 < merged.playerId2)]
    merged = merged.rename(columns={'playerId1': 'p1', 'playerId2': 'p2'})  # Rename player columns
    merged = merged.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    merged['combined_factor'] = merged['factor1'] + merged['factor2']
    return merged



'''
This method is responsible for attaching the role of a player to the players in the chemistry dataset
The method takes two arguments:
    1. A dataframe with the chemistry values and player ids 
    2. A dataframe with data on each player recorded in Wyscout
The method attaches the roles and returns the dataframw 
'''
def add_pos(df, df_players):
    df_player_1_added = pd.merge(df, df_players[['playerId', 'role_name']], left_on='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players[['playerId', 'role_name']], left_on='p2', right_on="playerId")
    cols = df.columns
    columns = np.concatenate((['role_name_x', 'role_name_y'], cols))
    df_filtered = df_player_2_added[columns]
    return df_filtered


'''
This function calculates the weighted chemistry values based on the given DataFrame.
It takes one argument:
    1. A dataFrame containing chemistry values for player pairings
The method returns a dataFrame with calculated weighted chemistry values
'''
def get_weighted_chemistry(df_chem):
    df_chem['chemistry'] = np.where((df_chem.role_name_x == 'Forward') & (df_chem.role_name_y == 'Forward'),
                                    ((df_chem.df_joi90 * 0.75) + (df_chem.df_jdi90 * 0.25)) * (df_chem.combined_factor),
                                    np.where((df_chem.role_name_x == 'Defender') & (df_chem.role_name_y == 'Defender'),
                                             ((df_chem.df_joi90 * 0.25) + (df_chem.df_jdi90 * 0.75)) * (df_chem.combined_factor),
                                             np.where(((df_chem.role_name_x == 'Midfielder') & (df_chem.role_name_y == 'Forward'))
                                             | ((df_chem.role_name_x == 'Forward') & ( df_chem.role_name_y == 'Midfielder')),
                                                      ((df_chem.df_joi90 * 0.60) + (df_chem.df_jdi90 * 0.40)) * (df_chem.combined_factor),
                                                      np.where(((df_chem.role_name_x == 'Midfielder') & (df_chem.role_name_y == 'Defender')) |
                                                               ((df_chem.role_name_x == 'Defender') & (df_chem.role_name_y == 'Midfielder')),   ((df_chem.df_joi90 * 0.40) + (df_chem.df_jdi90 * 0.60)) * (df_chem.combined_factor),  ((df_chem.df_joi90 * 0.50) + ( df_chem.df_jdi90 * 0.50)) * (df_chem.combined_factor)
                                                               )
                                                      )
                                             )
                                    )
    return df_chem


def get_chemistry(df_factor_values, df_joi90_jdi90, df_players_teams):
    dfm_id = df_factor_values[['p1', 'p2', 'teamId', 'seasonId']]  # Extract columns that should not be scaled
    mask_m = ~df_factor_values.columns.isin(['p1', 'p2', 'teamId', 'seasonId'])  # Extract column names for scaling
    df_scale_m = df_factor_values.loc[:, mask_m]
    scale1 = MinMaxScaler(feature_range=(0.8, 1.2))
    df_scale_m[df_scale_m.columns] = scale1.fit_transform(df_scale_m[df_scale_m.columns])  # Perform min/max scaling
    df_factors_scaled = pd.concat([dfm_id.reset_index(drop=True), df_scale_m.reset_index(drop=True)], axis=1)

    # Merge with joi and jdi dataframes
    dfc = pd.merge(df_joi90_jdi90, df_factors_scaled, on=['teamId', 'p1', 'p2', 'seasonId'], how='inner')
    dfc = dfc.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    df_id = dfc[['p1', 'p2', 'teamId','seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor']]  # Extract columns that should not be scaled
    mask = ~dfc.columns.isin( ['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor'])  # Extract column names for scaling
    df_scale = dfc.loc[:, mask]  # Extract columns for scaling

    scale2 = MinMaxScaler()  # Initiate sclaing instance
    df_scale[df_scale.columns] = scale2.fit_transform(df_scale[df_scale.columns])  # Perform min/max scaling
    # Re-establosh dataframe with id's
    df_chemistry = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)

    df_chemistry_pos = add_pos(df_chemistry, df_players_teams)
    # Compute chemistry columns based on formula
    df_chem_final = get_weighted_chemistry(df_chemistry_pos)

    return df_chem_final


'''
This method is responsible for attaching names and team of pairs of players from which chemistry has been determined
The method takes two arguments:
    1. A dataframe with the chemistry values and player ids 
    2. A dataframe with data on each player recorded in Wyscout
The method attaches the meta data and returns the new dataframw
'''
def get_overview_frame(df_chem, df_players):
    df_player_1_added = pd.merge(df_chem, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    df_filtered = df_player_2_added[['p1','p2', 'seasonId', 'shortName_x', 'shortName_y', 'minutes', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'df_jdi90', 'df_joi90','chemistry']]
    return df_filtered



'''
This method is responsible for converting the birtdate to an age -  the method is used in a lambda expression
The method takes one argument
    1. A Dataframe with one row about the the birtdate of a player  - the method is used
The emthod returns an age value as an integer
'''
def get_age(birthdate):
    today = date.today()
    birthdate_formatted = datetime.strptime(birthdate, '%Y-%m-%d').date()
    age = today.year - birthdate_formatted.year - ((today.month, today.day) < (birthdate_formatted.month, birthdate_formatted.day))
    return age





'''
This method is responsible for computing the Per-90-minutes-played league vaep
The mathod takes two arguments:
    1. A dataframe with event data in a league across as season 
    2. A dataframe with data on the duration of each match in the league 
The method returns a value being the per-90-minutes played league vaep
'''
def get_league_vaep(df, match_duration):
    vaep = df['sumVaep'].sum()
    total_minutes = match_duration['minutes'].sum()
    league_vaep_per_90 = vaep/(total_minutes/90)
    return league_vaep_per_90



'''
This method is responsible for extracting the chemistry ability for each player
The method accepts one dataframe as argument 
    1. A dataframe containing pairs of players with their chemistry values
The method returns a dataframe with each pair and their corresponding chemistry ability values
'''
def compute_player_chemistry_ability(chemistries):
    # Initialize an empty list to store the computed data
    list_with_data = []

    # Iterate over each row in the chemistries DataFrame
    for i, row in chemistries.iterrows():
        # Get the values of p1 and p2 from the current row
        p1 = row['p1']
        p2 = row['p2']

        # Filter the chemistries DataFrame to find rows where p1 or p2 matches and the other player is different
        adaptability_frame_p1 = chemistries[
            (chemistries['p1'] == p1) & (chemistries['p2'] != p2) | (chemistries['p2'] == p1) & (
                        p2 != chemistries['p1'])]

        # Filter the chemistries DataFrame to find rows where p2 or p1 matches and the other player is different
        adaptability_frame_p2 = chemistries[
            (chemistries['p1'] == p2) & (chemistries['p2'] != p1) | (chemistries['p2'] == p2) & (
                        p1 != chemistries['p1'])]

        # Compute the average chemistry value for p1 by taking the mean of the 'chemistry' column in adaptability_frame_p1
        p1_ability = adaptability_frame_p1['chemistry'].mean()

        # Compute the average chemistry value for p2 by taking the mean of the 'chemistry' column in adaptability_frame_p2
        p2_ability = adaptability_frame_p2['chemistry'].mean()

        # Get the values of teamId and seasonId from the current row
        tid = row['teamId']
        sid = row['seasonId']

        # Append the computed values to the list_with_data as a list
        list_with_data.append([p1, p2, tid, sid, p1_ability, p2_ability, row['chemistry']])

    # Convert the list of lists to a DataFrame with specified column names
    df_result = pd.DataFrame(list_with_data,
                             columns=['p1', 'p2', 'teamId', 'seasonId', 'chem_coef_x', 'chem_coef_y', 'chemistry'])

    # Return the resulting DataFrame
    return df_result


'''
This method i responsible for identifying and extracting all new arrivals to a team for the season 21/22
The method takes two arguments:
    1. A dataframe with transfer history
    2. A dataframe with chemistry ability values related to each pair of players
the metohd returns a dataframe containing observations consisting of pairs of players with where at least one was a new arrival to a clu 
including their chemistry ability figures 
'''
def get_new_arrivals(df_transfers, df_chem):
    # Filter out transfers with invalid start dates
    df_transfers = df_transfers[df_transfers['startDate'] != '0000-00-00']

    # Filter out transfers of type 'Back from Loan'
    df_transfers = df_transfers[df_transfers['type'] != 'Back from Loan']

    # Filter out transfers of type 'Free Agent'
    df_transfers = df_transfers[df_transfers['type'] != 'Free Agent']

    # Extract the year from the start date and create a new 'date' column
    df_transfers['date'] = df_transfers.apply(lambda row: (datetime.strptime(row['startDate'], '%Y-%m-%d').date()).year, axis=1)

    # Filter transfers that occurred between 2020 and 2021 (exclusive)
    df_transfers = df_transfers[(df_transfers['date'] > 2020) & (df_transfers['date'] < 2022)]

    # Merge df_transfers with df_chem based on the playerId and p1 columns, and rename playerId to 'arrival'
    df_arrivals_1 = (df_transfers.merge(df_chem, left_on='playerId', right_on='p1')).rename(columns={'playerId': 'arrival'})

    # Merge df_transfers with df_chem based on the playerId and p2 columns, and rename playerId to 'arrival'
    df_arrivals_2 = df_transfers.merge(df_chem, left_on='playerId', right_on='p2').rename(columns={'playerId': 'arrival'})

    # Concatenate the two arrival DataFrames into a single DataFrame
    df_arrivals = pd.concat([df_arrivals_1, df_arrivals_2])

    # Return the resulting DataFrame containing new arrivals
    return df_arrivals


def count_teams(transfer, players):
    from_team = transfer[['playerId', 'fromTeamId']].rename(columns={'fromTeamId': 'num_transfer'})
    to_team = transfer[['playerId', 'toTeamId']].rename(columns={'toTeamId': 'num_transfer'})
    concatted = pd.concat([from_team, to_team])
    players_and_transfer = concatted.groupby('playerId')['num_transfer'].nunique().reset_index()
    players_and_transfer_2 = (players.merge(players_and_transfer, on='playerId', how='left'))[['playerId', 'num_transfer']]
    players_and_transfer_2 = players_and_transfer_2.fillna(1)
    return players_and_transfer_2


def get_data():
    players = get_all_players()
    players_teams = get_players_and_teams()
    keepers = players[players['role_code2'] == 'GK']
    matches = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_matches_all.csv")
    squad = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_squad.csv", decimal=",",sep=(';'))
    squad = squad.merge(matches[['matchId', 'seasonId']], on='matchId')
    def_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/def_success.csv", decimal=",")
    air_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/air_suc.csv", decimal=",")
    def_suc_t = create_def_succes_frame(def_suc, air_suc)
    return keepers, players_teams, matches, squad, def_suc_t


def merge_performance_stats(pos, min_match_pl, zones_actions  ):
    pos_tot = pos.groupby(['playerId'], as_index = False).agg({'avg_x': 'mean', 'avg_y': 'mean'})
    min_match_pl = min_match_pl[min_match_pl['minutes'] > 30]
    min_match_pl_tot = min_match_pl.groupby(['playerId'], as_index = False).agg({'minutes': 'sum', 'matchId': 'count'})
    zones_actions_tot = zones_actions.groupby(['playerId'], as_index = False).agg({'zone_1_pl': 'sum', 'zone_2_pl': 'sum', 'zone_3_pl': 'sum', 'zone_4_pl': 'sum', 'zone_5_pl': 'sum', 'zone_6_pl': 'sum'})
    pos_tot = pos_tot.merge(min_match_pl_tot, on='playerId')
    merged = pos_tot.merge(zones_actions_tot, on='playerId')
    return merged






"""
Computes the expected wins
The argument receives these argument:
    1.he input dataframe containing match data.
    The method returns: A dataframe containing the team IDs and expected wins   
"""
def compute_expected_outputs(df):
    # Calculate the total expected goals (xG) per team per match
    xg_game = df.groupby(['teamId', 'matchId'], as_index=False)['xG'].sum()

    # Merge the xg_game dataframe with itself on the matchId column
    xg_game_v2 = xg_game.merge(xg_game, on='matchId')

    # Filter out rows where the teamIds are the same (teams playing against themselves)
    xg_game_v2 = xg_game_v2[xg_game_v2['teamId_x'] != xg_game_v2['teamId_y']]

    # Drop duplicate matchId entries to get one entry per match
    xg_game_v3 = xg_game_v2.drop_duplicates(subset='matchId')

    # Determine the winning team based on xG comparison
    xg_game_v3['teamId'] = np.where(xg_game_v3.xG_x > xg_game_v3.xG_y, xg_game_v3.teamId_x,
                                    np.where(xg_game_v3.xG_y > xg_game_v3.xG_x, xg_game_v3.teamId_y, -1))

    # Count the number of wins for each team
    team_count_wins = xg_game_v3.groupby(['teamId']).size().reset_index(name='Count')

    # Drop duplicate matchId entries to get one entry per match
    xg_game_v4 = xg_game_v2.drop_duplicates(subset='matchId')

    # Determine the losing team based on xG comparison
    xg_game_v4['teamId'] = np.where(xg_game_v4.xG_x < xg_game_v4.xG_y, xg_game_v4.teamId_x, np.where(xg_game_v4.xG_y < xg_game_v4.xG_x, xg_game_v4.teamId_y, -1))

    # Count the number of losses for each team
    team_count_losses = xg_game_v4.groupby(['teamId']).size().reset_index(name='Count')

    # Rename the 'Count' column to 'expected_wins' in the team_count_wins dataframe
    team_count_wins = team_count_wins.rename(columns={'Count': 'expected_wins'})

    # Rename the 'Count' column to 'expected_losses' in the team_count_losses dataframe
    team_count_losses = team_count_losses.rename(columns={'Count': 'expected_losses'})

    # Merge the team_count_wins and team_count_losses dataframes based on the teamId column
    team_count_total = team_count_wins.merge(team_count_losses, on='teamId')

    return team_count_total



#Count the amount of actual wins - Not used for anything during this project
#SAFE TO DELETE
def compute_äctual_outputs(df, matches_all):
    matches = matches_all[matches_all['seasonId'] == (df.iloc[0].seasonId)]
    team_count_wins = matches.groupby(['winner']).size().reset_index(name='Count')
    team_count_wins = team_count_wins.rename(columns={'winner': 'teamId', 'Count': 'actual_wins'})
    return team_count_wins



#Data exploratoin plot related to investigating the target distribution
def check_dis(df, feature):
    plt.hist(df[feature], bins=10)
    plt.title("Histogram of " + feature +  " variable")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


#Here we compute the chemistry produced by a team in a season.
#NOTE: JOI90 and JDI90 are kept for validation and discussion purposes.
def get_team_chemistry(df):
    df_team_chem = df.groupby(['teamId', 'seasonId'], as_index = False).agg({'chemistry': 'sum', 'df_joi90':'sum', 'df_jdi90': 'sum'})
    return df_team_chem



#This is the method responsible for the initial data load related to all chemistry predictoin purposes
#NOTE: Brondby needs to change all data loads to api calls. The process was simply to heavy during this project.
def data_for_prediction():
    # Read the 'chemistry' data from a CSV file
    areas = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/area_db.csv")
    chemistry = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/pairwise_chemistry.csv")

    # Read the 'players_roles' data from a CSV file
    players_roles = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_clusters.csv")

    # Read the 'transfer' data from a CSV file
    transfer = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_transfer.csv")

    # Read the 'players' data from a CSV file
    players = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_players.csv")

    # Read the 'positions_and_formations' data from a CSV file
    positions_and_formations = pd.read_csv(
        "C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/player_positions.csv")

    # Read the 'performance_stats' data from a CSV file
    performance_stats = pd.read_csv(
        "C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/performance_stats_v2.csv")

    # Read the 'teams' data from a CSV file
    teams = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_teams.csv")

    # Return all the loaded data as separate DataFrames
    return chemistry, players_roles, performance_stats, transfer, positions_and_formations, players, teams, areas


def prepare_player_data_and_transfer_data(players, transfers, areas):
    # Drop rows with missing values in the 'birthDate' column from the 'players' DataFrame
    players = players.dropna(subset=['birthDate'])
    # Drop rows with missing values in the 'passportArea_name' column from the 'players' DataFrame
    players = players.dropna(subset=['passportArea_name'])

    # Filter the 'transfers' DataFrame to exclude rows with 'startDate' value of '0000-00-00'
    transfers = transfers[transfers['startDate'] != '0000-00-00']
    # Filter the 'transfers' DataFrame to exclude rows with missing values in the 'endDate' column
    transfers = transfers[transfers['endDate'].notna()]

    # Merge the 'players' DataFrame with the 'df_areas' DataFrame based on the 'passportArea_name' column
    players = players.merge(areas, left_on='passportArea_name', right_on='name')

    # Select specific columns from the 'players' DataFrame and create a new DataFrame called 'players_filtered'
    players_filtered = (players[
        ['playerId', 'shortName', 'birthDate', 'birthArea_name', 'passportArea_name', 'role_name',
         'currentTeamId']]).rename(columns={'role_name': 'position'})

    # Calculate the age for each player based on their 'birthDate' column using a lambda function
    players_filtered['age'] = players_filtered.apply(lambda row: get_age(row['birthDate']), axis=1)

    # Return the 'players_filtered' DataFrame and the 'transfer' DataFrame as the prepared player data and transfer data
    return players_filtered, transfers


'''
This function calculates the chemistry values for player pairings based on given DataFrames.
It takes three arguments:
    1. A dataFrame containing factor values for player pairings
    2. A dataFrame containing joi90 and jdi90 values for player pairings
    3. A dataFrame containing player-team information
The method return the  pairwise chemistry scores among two playres in a team across a season
'''

def generate_chemistry(squad, pos, df_matches_all, def_suc, players_teams, sd_table, comp_id):
    #Filter to obtains values from specific contest
    league_df = sd_table[sd_table['competitionId'] == comp_id]
    pos_league = pos[pos['competitionId'] == comp_id]

    # Remove players not in the starting 11
    squad = squad[(squad['bench'] == False)]

    # Copy the league dataframe for further processing
    df_process = league_df.copy()

    # Get the latest season ID
    s_21_22 = max(df_process['seasonId'])

    # Merge defensive success data with player and match information
    def_per_player_per_match = league_df[['eventId', 'playerId', 'seasonId']].merge(def_suc, on='eventId')
    def_per_player_per_match = def_per_player_per_match[def_per_player_per_match['seasonId'] == s_21_22]

    # Compute the defensive actions per player per match
    d_m_s = (def_per_player_per_match.groupby(['playerId', 'matchId', 'seasonId'], as_index=False)['def_action'].sum())[['playerId', 'matchId', 'def_action']]

    # Filter dataframes to only contain data from the latest season
    df_matches_all = df_matches_all[df_matches_all['seasonId'] == s_21_22]
    df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
    df_sqaud_filtered = squad[squad['seasonId'] == s_21_22]
    pos_filtered = pos_league[pos_league['seasonId'] == s_21_22]

    # Get the maximum duration of each match
    df_match_duration = df_sqaud_filtered.groupby('matchId', as_index=False)['minutes'].max()

    # Sort the filtered position data by possession session and event index
    pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)

    # Compute pairwise playing time for the squad
    df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

    # Extract net offensive impact per game per team
    df_net_oi = getOi(df_process.copy())

    # Extract distance measures
    df_ec, avg_pos = getDistance(df_process_21_22.copy())

    # Extract players shares
    df_player_share, zones_actions = getResponsibilityShare(df_process_21_22.copy())

    # Extract JDI
    df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all, d_m_s)

    #Extract JOI
    df_joi = get_joi(pos_sorted)

    #Normalize values based on per 90 minutes played metric
    df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

    #Extract timestamps related to each event
    stamps = get_timestamps(max(df_process['seasonId']))

    #Extract combinde factor values
    ready_for_scaling = prepare_for_scaling(df_process_21_22, df_sqaud_filtered, stamps, df_match_duration)
    #Filter dataframes
    ready_for_scaling['seasonId'] = s_21_22
    df_joi90_and_jdi90['seasonId'] = s_21_22

    #Get a focused dataframe containing only team data relevant for the chemistry data points to be used in the "get_chemistry method"
    df_players_teams_c = players_teams[players_teams['teamId'].isin(df_joi90_and_jdi90['teamId'].unique())]

    #Get chemistry valyes
    df_chemistry = get_chemistry(ready_for_scaling, df_joi90_and_jdi90, df_players_teams_c)[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor', 'joi', 'jdi', 'df_jdi90',  'df_joi90', 'winners90', 'chemistry']]
    return df_chemistry, df_process_21_22


#Single plot analysis purposed to check the relationship between two variables found in the dataframe, df
# This method takes three arguments,
# 1. a dataframe with at least two columns being expected wins and chemistry
# 2. A label in this case chemistry
# 3, A title as string ot be displayed in the plot
def plot_relationship_chemistry(df, label_y, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Select only the desired team names
    desired_teams = ['Real Madrid', 'Barcelona', 'Athletic Bilbao']
    filtered_df = df[df['name'].isin(desired_teams)]

    # Extract unique season IDs
    unique_season_ids = df['seasonId'].unique()

    # Set blood orange color for the dots
    dot_color = 'tab:red'

    # Create the scatter plot with blood orange dots
    for i, season_id in enumerate(unique_season_ids):
        season_df = df[df['seasonId'] == season_id]
        ax.scatter(season_df['df_joi90'], season_df[label_y], color=dot_color)

        # Add text annotations only for desired team names
        for j, name in enumerate(season_df['name']):
            if name in desired_teams:
                x_offset = 0.2
                y_offset = 0.2
                ax.text(season_df['df_joi90'].iloc[j] + x_offset, season_df[label_y].iloc[j] + y_offset, name,
                        ha='left',
                        va='bottom')

    ax.set_xlabel('df_joi90')
    ax.set_ylabel(label_y)
    ax.set_title('Relationship between ' + title + ' and Expected Wins', pad=40)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='lightgray', alpha=0.25, zorder=1)
    plt.show()

    # Single plot analysis purposed to check the relationship between two variables found in the dataframe, df
    # This method takes three arguments,
    # 1. a dataframe with at least two columns being expected wins and net worth
    # 2. A label in this case worth
    # 3, A title as string ot be displayed in the plot
def plot_relationship_netVal(df, label_y, title):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Select only the desired team names
        desired_teams = ['Real Madrid', 'Barcelona', 'Athletic Bilbao']
        filtered_df = df[df['name'].isin(desired_teams)]

        # Extract unique season IDs
        unique_season_ids = df['seasonId'].unique()

        # Set blood orange color for the dots
        color_map = plt.get_cmap('plasma')
        colors = color_map(np.linspace(0, 1, len(unique_season_ids)))

        # Create the scatter plot with blood orange dots
        for i, season_id in enumerate(unique_season_ids):
            season_df = df[df['seasonId'] == season_id]
            ax.scatter(season_df['Total Value'], season_df[label_y], color=colors[i])

            # Add text annotations only for desired team names
            for j, name in enumerate(season_df['name']):
                if name in desired_teams:
                    x_offset = 0.2
                    y_offset = 0.2
                    ax.text(season_df['Total Value'].iloc[j] + x_offset, season_df[label_y].iloc[j] + y_offset, name,
                            ha='left',
                            va='bottom')

        ax.set_xlabel('Total Value')
        ax.set_ylabel(label_y)
        ax.set_title('Relationship between Expected Wins and ' + title, pad=40)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(color='lightgray', alpha=0.25, zorder=1)
        plt.show()




#This function yields two horizontal side-by-side plots
# 1. The relationship between expected wins and chemistry
# 2. The relationship between expected wins and total net value of the teams
def plot_relationship_chem_val(df, label_y, title1, title2):
    # Select only the desired team names
    desired_teams = ['Real Madrid', 'Barcelona', 'Athletic Bilbao']
    filtered_df = df[df['name'].isin(desired_teams)]

    # Extract unique season IDs
    unique_season_ids = df['seasonId'].unique()

    # Set blood orange color for the dots
    dot_color = 'tab:red'

    # Create the scatter plot with blood orange dots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for Chemistry
    ax1.set_title('Expected Wins and ' + title1, pad=40, fontsize = 14)
    for i, season_id in enumerate(unique_season_ids):
        season_df = df[df['seasonId'] == season_id]
        ax1.scatter(season_df['chemistry'], season_df[label_y], color=dot_color)
        for j, name in enumerate(season_df['name']):
            if name in desired_teams:
                x_offset = 0.2
                y_offset = 0.2
                ax1.text(season_df['chemistry'].iloc[j] + x_offset, season_df[label_y].iloc[j] + y_offset, name,
                         ha='left', va='bottom', fontsize = 12)
    ax1.set_xlabel('Chemistry', fontsize = 12)
    ax1.set_ylabel(label_y, fontsize = 12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='lightgray', alpha=0.25, zorder=1)

    # Plot for Total Value
    ax2.set_title('Expected Wins and ' + title2, pad=40, fontsize = 14)
    color_map = plt.get_cmap('plasma')
    colors = color_map(np.linspace(0, 1, len(unique_season_ids)))
    for i, season_id in enumerate(unique_season_ids):
        season_df = df[df['seasonId'] == season_id]
        ax2.scatter(season_df['Total Value'], season_df[label_y], color=colors[i])
        for j, name in enumerate(season_df['name']):
            if name in desired_teams:
                x_offset = 0.2
                y_offset = 0.2
                ax2.text(season_df['Total Value'].iloc[j] + x_offset, season_df[label_y].iloc[j] + y_offset, name,
                         ha='left', va='bottom', fontsize = 12)
    ax2.set_xlabel('Total Value', fontsize = 12)
    ax2.set_ylabel(label_y, fontsize = 12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='lightgray', alpha=0.25, zorder=1)

    plt.tight_layout()
    plt.show()


'''
This method is used to determine the chemistry ability of a player.
The method takes one argument
    1. A dataframe containing the pairwise chemistry scores of players across a season
the method return a dataframe containing the average chemistry ability of a player in the league
'''
def get_avg_chem_league(df):
    # Create an empty DataFrame to store league abilities
    league_abilities = pd.DataFrame()

    # Get unique team IDs from the input DataFrame
    tid = df.teamId.unique()

    # Iterate over each team ID
    for i in tid:
        # Filter the DataFrame for the current team ID
        dfa = df[df['teamId'] == i]

        # Select player 1 data (playerId, teamId, chemistry) and rename the columns
        p_one = (dfa[['p1', 'teamId', 'chemistry']]).rename(columns={'p1': 'playerId'})

        # Select player 2 data (playerId, teamId, chemistry) and rename the columns
        p_two = (dfa[['p2', 'teamId', 'chemistry']]).rename(columns={'p2': 'playerId'})

        # Concatenate player 1 and player 2 data
        p = pd.concat([p_one, p_two])

        # Calculate the average chemistry for each player within the team
        abilities = p.groupby(['playerId', 'teamId'], as_index=False)['chemistry'].mean()

        # Concatenate the abilities of players from the current team with the league abilities DataFrame
        league_abilities = pd.concat([league_abilities, abilities])

    # Return the DataFrame containing the league abilities
    return league_abilities



'''
This method is responsible for producing the boxplot used to investigate the correlation for each league with the expected wins variable 
The method takes one argument
    1. A dataframe containing correlation figures for all leagues in scope
The method creates a boxplot using plotly
'''
def get_box(df):

    # Convert the 'Country' column to numeric codes
    df['Country'] = pd.Categorical(df['Country'])
    df['Country'] = df['Country'].cat.codes

    # Create the jitter plot using Plotly
    fig = go.Figure()

    # Add jitter points without legend
    fig.add_trace(go.Scatter(
        x=[0] * len(df),
        y=df['Correlation'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Country'],
            colorscale='plasma',
            opacity=0.7,
            symbol='circle'
        ),
        hovertext=df['Country'],
        showlegend=False,
        name='Jitter Plot'
    ))

    # Add box plot
    fig.add_trace(go.Box(
        y=df['Correlation'],
        marker=dict(
            color='lightgray',
            line=dict(color='black', width=1)
        ),
        showlegend=False,
        boxpoints=False
    ))

    # Add mean, min, and max annotations
    median_value = df['Correlation'].median()
    min_value = df['Correlation'].min()
    max_value = df['Correlation'].max()

    fig.add_annotation(x=-0.5, y=median_value, text=f'Median: {median_value:.2f}', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=-0.5, y=min_value, text=f'Min: {min_value:.2f}', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=-0.5, y=max_value, text=f'Max: {max_value:.2f}', showarrow=False, font=dict(size=14))

    # Create custom legend
    countries = df['Country'].unique()
    legend_items = []
    for country in countries:
        country_df = df[df['Country'] == country]
        legend_items.append(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(
                color=country_df['Country'].iloc[0],
                colorscale='plasma',
                opacity=0.7,
                symbol='circle',
                size=10
            ),
            hoverinfo='skip',
            showlegend=True,
            name=str(country)  # Convert country code to string
        ))

    # Add custom legend items
    for item in legend_items:
        fig.add_trace(item)

    # Update layout
    fig.update_layout(
        title='Countries & Correlation',
        xaxis_title='',
        yaxis_title='Correlation',
        width=800,
        height=800,
        plot_bgcolor='white',
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )
    offline.plot(fig, filename='box_plot.html')
    fig.show()


def plot_relationship(df, label):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['chemistry'],
        y=df['expected_wins'],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7
        ),
        name='Chemistry vs Expected Wins'
    ))

    fig.update_layout(
        title='Relationship between Chemistry and Expected Wins - ' + label,
        xaxis_title='Chemistry',
        yaxis_title='Expected Wins'
    )
    offline.plot(fig, filename='box_plot.html')

    fig.show()


'''
This method is responsible for generating the used binary indicator variables
The method takes on dataframe as argument 
    1. A dataframe consiting of data related to the prigin of a player, their current country and previous experience
The method returns a dataframe wit varaibles indicating if a pair of players share nationality, provenance or career experience
'''
def create_indicators(df):
    # Select specific columns from the input DataFrame 'df'
    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'countries_x', 'shortName_x', 'birthDate_x',
             'birthArea_name_x', 'passportArea_name_x', 'position_x', 'currentTeamId_x', 'age_x',
             'pos_group_x', 'ip_cluster_x','zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x',
             'zone_6_pl_x', 'minutes_played_season_x', 'match appearances_x', 'chem_coef_x', 'countries_y',
             'shortName_y', 'birthDate_y', 'birthArea_name_y', 'passportArea_name_y',
             'position_y', 'currentTeamId_y', 'age_y', 'pos_group_y', 'ip_cluster_y', 'minutes_played_season_y',
             'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y',
             'zone_6_pl_y', 'chem_coef_y']]

    # Create indicator columns based on conditions
    df['same_origin'] = np.where(df['birthArea_name_x'] == df['birthArea_name_y'], 1, 0)
    df['same_country'] = np.where(df['passportArea_name_x'] == df['passportArea_name_y'], 1, 0)
    df['played_in_same_country'] = df.apply(lambda row: 1 if len(check_country(row['countries_x'], row['countries_y'])) > 0 else 0, axis=1)

    # Select specific columns from the modified DataFrame 'df'
    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'same_origin', 'same_country', 'played_in_same_country',
             'position_x', 'age_x', 'pos_group_x', 'ip_cluster_x', 'minutes_played_season_x',
             'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x',
             'zone_6_pl_x', 'chem_coef_x', 'position_y', 'age_y', 'pos_group_y', 'ip_cluster_y',
             'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
             'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y']]

    return df


'''
This function finds players and their corresponding countries based on given DataFrames.
It takes three arguments:
    1. A dataFrame containing player information
    2. A dataFrame containing transfer information
    3- A dataFrame containing team information
The method returns a dataframe informatoin about players and the countries they previously played in
'''
def find_players_and_countries(df, transfer, teams):
    # Merge the 'transfer' DataFrame with the 'teams' DataFrame based on 'fromTeamId' and 'toTeamId' columns
    df_teams_and_transfers = pd.merge(transfer, teams, left_on='fromTeamId', right_on='teamId')
    df_teams_and_transfers = df_teams_and_transfers.merge(teams, left_on='toTeamId', right_on='teamId')

    # Convert the 'startDate' column to datetime objects and extract the date component
    df_teams_and_transfers['date'] = df_teams_and_transfers.apply(
        lambda row: datetime.strptime(row['startDate'], '%Y-%m-%d').date(), axis=1)

    # Create an empty dictionary to store player-country information
    p_dict = {}

    # Get unique player IDs from the input DataFrame 'df'
    players = df['playerId'].unique()

    # Iterate over each player ID
    for e in players:
        # Filter the merged DataFrame to get rows corresponding to the current player
        rows = df_teams_and_transfers[df_teams_and_transfers['playerId'] == e]

        # Sort the rows based on the 'date' column in ascending order
        rows = rows.sort_values(by=['date'], ascending=True)

        # Get the first row (earliest transfer) for the current player
        row = rows.iloc[0]

        # Get the starting country of the player from the 'areaName_x' column
        startCountry = row['areaName_x']

        # Filter the input DataFrame 'df' to get rows corresponding to the current player
        df_a = df[df['playerId'] == e]

        # Get unique countries from the filtered DataFrame
        countries = df_a['areaName_y'].unique().tolist()

        # If the starting country is not already in the list of countries, add it
        if startCountry not in countries:
            countries.append(startCountry)

        # Store the player ID and countries in the dictionary
        p_dict[e] = {'playerId': e, 'countries': countries}

    # Create a DataFrame from the dictionary and reset the index
    df_created = pd.DataFrame.from_dict(p_dict, orient='index').reset_index(drop=True)

    return df_created



'''
This method is responsible for generating the used binary indicator variables
The method takes on dataframe as argument 
    1. A dataframe consiting of data related to the prigin of a player, their current country and previous experience
The method returns a dataframe wit varaibles indicating if a pair of players share nationality, provenance or career experience
'''
def create_indicators(df):
    # Select specific columns from the input DataFrame 'df'
    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'countries_x', 'shortName_x', 'birthDate_x',
             'birthArea_name_x', 'passportArea_name_x', 'position_x', 'currentTeamId_x', 'age_x',
             'pos_group_x', 'ip_cluster_x','zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x',
             'zone_6_pl_x', 'minutes_played_season_x', 'match appearances_x', 'chem_coef_x', 'countries_y',
             'shortName_y', 'birthDate_y', 'birthArea_name_y', 'passportArea_name_y',
             'position_y', 'currentTeamId_y', 'age_y', 'pos_group_y', 'ip_cluster_y', 'minutes_played_season_y',
             'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y',
             'zone_6_pl_y', 'chem_coef_y']]

    # Create indicator columns based on conditions
    df['same_origin'] = np.where(df['birthArea_name_x'] == df['birthArea_name_y'], 1, 0)
    df['same_country'] = np.where(df['passportArea_name_x'] == df['passportArea_name_y'], 1, 0)
    df['played_in_same_country'] = df.apply(lambda row: 1 if len(check_country(row['countries_x'], row['countries_y'])) > 0 else 0, axis=1)

    # Select specific columns from the modified DataFrame 'df'
    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'same_origin', 'same_country', 'played_in_same_country',
             'position_x', 'age_x', 'pos_group_x', 'ip_cluster_x', 'minutes_played_season_x',
             'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x',
             'zone_6_pl_x', 'chem_coef_x', 'position_y', 'age_y', 'pos_group_y', 'ip_cluster_y',
             'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
             'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y']]

    return df



def elapsed_time(start_date_str, end_date_str):
    # Check if the end date is not '0000-00-00'
    if end_date_str != '0000-00-00':
        # Convert the start and end dates from strings to date objects
        start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        # Calculate the time difference between the start and end dates in days
        delta = end - start
        return delta.days
    else:
        # If the end date is '0000-00-00', return 0
        return 0



def handle_transfer_periods(df):
    # Apply the 'elapsed_time' function to each row of the DataFrame 'df' using the 'apply' function
    # and create a new column 'transfer_time' with the elapsed time in days
    df['transfer_time'] = df.apply(lambda row: elapsed_time(row['startDate'], row['endDate']), axis=1)
    return df


def handle_transfer_data(transfer, teams):
    # Merge the 'transfer' DataFrame with the 'teams' DataFrame based on 'fromTeamId' and 'toTeamId' columns
    df_teams_and_transfers = pd.merge(transfer, teams, left_on='fromTeamId', right_on='teamId')
    df_teams_and_transfers = df_teams_and_transfers.merge(teams, left_on='toTeamId', right_on='teamId')
    # Select specific columns from the merged DataFrame
    df_teams_and_transfers = df_teams_and_transfers[['toTeamId', 'fromTeamId', 'startDate', 'endDate', 'transfer_time', 'areaName_x', 'areaName_y', 'timestamp', 'playerId']]
    # Group the DataFrame by 'playerId' and 'areaName_y', and calculate the sum of 'transfer_time'
    transfers_handled_V2 = df_teams_and_transfers.groupby(['playerId', 'areaName_y'], as_index=False)['transfer_time'].sum()
    # Calculate the 'years_in_country' by dividing 'transfer_time' by 365.2425 (approximate number of days in a year)
    transfers_handled_V2['years_in_country'] = round((transfers_handled_V2['transfer_time'] / 365.2425), 2)
    # Filter the DataFrame to include only rows where 'years_in_country' is greater than or equal to 2
    transfers_handled_V2 = transfers_handled_V2[transfers_handled_V2['years_in_country'] >= 2]
    return transfers_handled_V2



def add_roles(df, roles):
    # Merge the 'df' DataFrame with the 'roles' DataFrame based on 'p1' and 'playerId' columns
    merged = df.merge(df, roles, left_on='p1', right_on='playerId')
    # Merge the 'df' DataFrame with the 'merged' DataFrame based on 'p2' and 'playerId' columns
    merged2 = df.merge(merged, roles, left_on='p2', right_on='playerId')
    return merged2


def duplicate_and_order_data(df):
    # Create a copy of the DataFrame 'df'
    df1 = df.copy()
    # Rename the columns in the copied DataFrame 'df1' to match the desired order and naming convention
    # by using the 'rename' function
    df1 = df1.rename(columns={'position_y': 'position_1', 'age_y': 'age_1', 'pos_group_y': 'pos_group_1', 'ip_cluster_y': 'ip_cluster_1', 'minutes_played_season_y': 'minutes_played_season_1', 'match appearances_y': 'match appearances_1', 'zone_1_pl_y': 'zone_1_pl_1', 'zone_2_pl_y': 'zone_2_pl_1', 'zone_3_pl_y': 'zone_3_pl_1', 'zone_4_pl_y': 'zone_4_pl_1', 'zone_5_pl_y': 'zone_5_pl_1', 'zone_6_pl_y': 'zone_6_pl_1', 'chem_ability_y': 'chem_ability_1', 'aerial_strength_y': 'aerial_strength_1', 'carry_strength_y': 'carry_strength_1', 'pressing_recovery_strength_y': 'pressing_recovery_strength_1', 'defensive_duel_strength_y': 'defensive_duel_strength_1', 'dribbles_strength_y': 'dribbles_strength_1', 'ground_duel_strength_y': 'ground_duel_strength_1', 'sliding_strength_y': 'sliding_strength_1', 'deep_crossing_strength_y': 'deep_crossing_strength_1'})
    df1 = df1.rename(columns={'position_x': 'position_y', 'age_x': 'age_y', 'pos_group_x': 'pos_group_y', 'ip_cluster_x': 'ip_cluster_y', 'minutes_played_season_x': 'minutes_played_season_y', 'match appearances_x': 'match appearances_y', 'zone_1_pl_x': 'zone_1_pl_y', 'zone_2_pl_x': 'zone_2_pl_y', 'zone_3_pl_x': 'zone_3_pl_y', 'zone_4_pl_x': 'zone_4_pl_y', 'zone_5_pl_x': 'zone_5_pl_y', 'zone_6_pl_x': 'zone_6_pl_y', 'chem_ability_x': 'chem_ability_y', 'aerial_strength_x': 'aerial_strength_y', 'carry_strength_x': 'carry_strength_y', 'pressing_recovery_strength_x': 'pressing_recovery_strength_y', 'defensive_duel_strength_x': 'defensive_duel_strength_y', 'dribbles_strength_x': 'dribbles_strength_y', 'ground_duel_strength_x': 'ground_duel_strength_y', 'sliding_strength_x': 'sliding_strength_y', 'deep_crossing_strength_x': 'deep_crossing_strength_y'})
    df1 = df1.rename(columns={'position_1': 'position_x', 'age_1': 'age_x', 'pos_group_1': 'pos_group_x', 'ip_cluster_1': 'ip_cluster_x', 'minutes_played_season_1': 'minutes_played_season_x', 'match appearances_1': 'match appearances_x', 'minutes_played_season_x': 'minutes_played_season_y', 'match appearances_x': 'match appearances_y', 'zone_1_pl_x': 'zone_1_pl_y', 'zone_1_pl_1': 'zone_1_pl_x', 'zone_2_pl_1': 'zone_2_pl_x', 'zone_3_pl_1': 'zone_3_pl_x', 'zone_4_pl_1': 'zone_4_pl_x', 'zone_5_pl_1': 'zone_5_pl_x', 'zone_6_pl_1': 'zone_6_pl_x', 'chem_ability_1': 'chem_ability_x', 'aerial_strength_1': 'aerial_strength_x', 'carry_strength_1': 'carry_strength_x', 'pressing_recovery_strength_1': 'pressing_recovery_strength_x', 'defensive_duel_strength_1': 'defensive_duel_strength_x', 'dribbles_strength_1': 'dribbles_strength_x', 'ground_duel_strength_1': 'ground_duel_strength_x', 'sliding_strength_1': 'sliding_strength_x', 'deep_crossing_strength_1': 'deep_crossing_strength_x'})
    return df1

'''
 This method finds the intersection of two sets of countries.
 The method takes two arguments
    1. A set of countries for one player - c1.
    2. A set of countries for a corresponding player - c2.
Returns a set containing the common countries between c1 and c2.
'''
def check_country(c1, c2):
    return set(c1).intersection(set(c2))

# The method checks if a value is NaN (float type) and returns a list containing the area if it is NaN, or the value itself otherwise.
def check_nan(val, area):
    if isinstance(val, float):
        return [area]
    else: return val


#This method creates and plots a density  plot for a specified label/column in a DataFrame.
def check_dis(df, label):
    sns.distplot(df[label], hist=False, kde=True,
                 kde_kws={'linewidth': 0.5}, )
    plt.show()

# Method to check target distribution
def show_target_distribution(df, label, sns=None):
    df = df[~df.index.duplicated(keep='first')]
    # Plot density plot of column 'petal_length'
    sns.kdeplot(data=df, x=label)
    plt.xlim(0, 0.06)
    plt.show()




'''
This method is responsible for determining the role of a player based on their roles from the clustering part of the proejct
The method takes one dataframe as argument:
    1. A dataframe with information on the cluster of a player
The method returns a dataframe where all players are put into their genereal role - the results of the first clustering
'''
def produce_overall_cluster(df):
    df['role_x'] = np.where((df.ip_cluster_x >= 0) & (df.ip_cluster_x <= 2), '1',
                                     np.where((df.ip_cluster_x >= 3) & (df.ip_cluster_x <= 5), '2',
                                              np.where((df.ip_cluster_x >= 6) & (df.ip_cluster_x <= 8), '3',
                                                  np.where((df.ip_cluster_x >= 9) & (df.ip_cluster_x <= 11), '4',
                                                           np.where((df.ip_cluster_x >= 12) & ( df.ip_cluster_x <= 14), '5',  '6')))))
    df['role_y'] = np.where((df.ip_cluster_y >= 0) & (df.ip_cluster_y <= 2), '1',
                                     np.where((df.ip_cluster_y >= 3) & (df.ip_cluster_y <= 5), '2',
                                              np.where( (df.ip_cluster_y >= 6) & (df.ip_cluster_y <= 8), '3',
                                                  np.where((df.ip_cluster_y >= 9) & ( df.ip_cluster_y <= 11), '4',
                                                           np.where((df.ip_cluster_y >= 12) & (df.ip_cluster_y <= 14), '5',  '6')))))

    return df




'''
This method is responsible for displaying feature importances
The method takes a prediction model and a dataframe as arguments
    1. The prediction models used for any prediction purposes[LightGBM, Xgboost or random forest]
    2. The input training dataset used to fit the model
The method goes on to plot the feature importances as a bar chart
'''
def show_feature_importances(model, input_train):
    # Generate and print feature scores
    feature_scores = pd.Series(model.feature_importances_, index=input_train.columns).sort_values(ascending=False)

    # Create a horizontal bar chart using Plotly Express
    fig = px.bar(feature_scores, orientation='h')

    # Customize the layout of the chart
    fig.update_layout(
        title='Feature Importances',
        showlegend=False,
    )

    # Display the chart
    fig.show()

    # Generate a standalone HTML file of the chart using Plotly's offline plotting library (optional)
    pyo.plot(fig)

'''
This method is responsible for displaying a confusion matrix using a heatmap - exploring feature correlations 
The method takes a dataframe as argument
    1. The input features are forwarded as a dataframe
The method computes the correlations in place and plota the heatmap
'''
def show_heat_map(features):
    # assume X is your feature matrix as a pandas DataFrame
    corr_matrix = features.corr()

    # plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    return corr_matrix


''' 
This method prepares the dataset for training a machine learning model by performing various data transformations and merging operations.
The method takes seven dataframes as arguments
    1. A dataframe containing info on transfers
    2. A dataframe containing info on the player roles
    3. A dataframe containing info on behavioral aspects
    4. A Dataframe containing info on the chemistry ability of the players
    5. A dataframe containing info player physological elements
    6. A taframe containing info on the positions and formations a player has been exposed to
The method return a prepared dataframe of features and a dataframe containing the target variable
'''
def prepare_set(transfer, teams, roles, performance_stats, ability_set, players_filtered, positions_formations):
    # Handle transfer periods in the transfer data
    df_transfer_periods = handle_transfer_periods(transfer)

    # Handle transfer data and merge with teams data
    tf_final = handle_transfer_data(df_transfer_periods, teams)

    # Find players and their countries
    players_and_cultures = find_players_and_countries(tf_final, df_transfer_periods, teams)

    # Merge filtered players data with players and cultures data
    df_v3 = players_filtered.merge(players_and_cultures, on='playerId', how='left')

    # Apply check_nan function to create a 'cultures' column
    df_v3['cultures'] = df_v3.apply(lambda row: check_nan(row['countries'], row['passportArea_name']), axis=1)

    # Select relevant columns and rename 'cultures' column to 'countries'
    df_v3 = (df_v3[
        ['playerId', 'shortName', 'birthDate', 'birthArea_name', 'passportArea_name', 'position',
         'currentTeamId', 'age', 'cultures']]).rename(columns={'cultures': 'countries'})

    # Merge with roles data
    df_v4 = df_v3.merge(roles, on='playerId')

    # Merge with performance statistics data
    df_v4 = df_v4.merge(performance_stats, on='playerId')

    # Rename 'minutes' column to 'minutes_played_season'
    df_v4 = df_v4.rename(columns={'minutes': 'minutes_played_season'})

    # Merge ability set data with df_v4 twice
    players_chemistry_1 = ability_set.merge(df_v4, left_on=['p1'], right_on=['playerId'])
    players_chemistry_t = players_chemistry_1.merge(df_v4, left_on=['p2'], right_on=['playerId'])

    # Filter out rows with invalid ip_cluster_x and ip_cluster_y values
    players_chemistry_t = players_chemistry_t[
        (players_chemistry_t['ip_cluster_x'] != -1) & (players_chemistry_t['ip_cluster_y'] != -1)]

    # Drop duplicates based on 'p1', 'p2', and 'teamId'
    set_1 = players_chemistry_t.drop_duplicates(subset=['p1', 'p2', 'teamId'])

    # Create indicators for chemistry
    with_indicators = create_indicators(set_1)

    # Duplicate and order data
    prep = duplicate_and_order_data(with_indicators)

    # Concatenate prep and with_indicators data
    prepped = pd.concat([prep, with_indicators])

    # Produce overall cluster
    prepped = produce_overall_cluster(prepped)

    # Count teams in transfer data for players_filtered
    num_transfer_df = count_teams(transfer, players_filtered)

    # Merge num_transfer_df twice with prepped data
    num_transfer_df_1 = prepped.merge(num_transfer_df, left_on=['p1'], right_on=['playerId'])
    num_transfer_df_2 = num_transfer_df_1.merge(num_transfer_df, left_on=['p2'], right_on=['playerId'])

    # Merge positions_formations data with num_transfer_df_2 twice
    positions_formations_1 = num_transfer_df_2.merge(positions_formations, left_on=['p1'], right_on=['playerId'])
    positions_formations_2 = positions_formations_1.merge(positions_formations, left_on=['p2'], right_on=['playerId'])

    # Drop duplicates based on 'p1', 'p2', and 'teamId'
    positions_formations_2 = positions_formations_2.drop_duplicates(subset=['p1', 'p2', 'teamId'])

    # Assign positions_formations_2 to prepped
    prepped = positions_formations_2

    # Drop unnecessary columns from prepped data
    pred_prep = prepped.drop(['p1', 'p2', 'teamId', 'seasonId', 'pos_group_x', 'ip_cluster_y', 'ip_cluster_x'], axis=1)

    # Create 'same_pos' column indicating if players have the same role
    pred_prep['same_pos'] = np.where(pred_prep['role_x'] == pred_prep['role_y'], 1, 0)

    # Create 'same_role' column indicating if players have the same position
    pred_prep['same_role'] = np.where(pred_prep['position_x'] == pred_prep['position_y'], 1, 0)

    # Drop 'role_x', 'role_y', 'position_x', 'position_y' columns
    pred_prep = pred_prep.drop(['role_x', 'role_y', 'position_x', 'position_y'], axis=1)

    # Fill missing values with 0
    pred_prep = pred_prep.fillna(0)

    # Reorder columns
    pred_prep = pred_prep[['same_origin', 'same_country', 'played_in_same_country', 'age_y',
                           'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y',
                           'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y',
                           'age_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x',
                           'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x',
                           'same_pos', 'same_role', 'num_transfer_x', 'num_transfer_y', 'num_positions_x',
                           'num_positions_y', 'num_formations_x', 'num_formations_y', 'chemistry']]

    # Get the column names as feature_columns
    feature_columns = pred_prep.columns

    # Get the input variables by excluding the 'chemistry' column
    input_variables = pred_prep.columns[feature_columns != 'chemistry']

    # Create input data by selecting the input variables from pred_prep
    input = pred_prep[input_variables]

    # Get the target variable (chemistry) as target_prepped
    target_prepped = pred_prep['chemistry']

    # Return the input and target_prepped
    return input, target_prepped


'''
This function finds potential fits for players based on established players, using a model for prediction.
It takes six arguments:
    1. A dataFrame of established players
    2- A dataframe of potential players
    3. A dataFrame containing player roles information
    4. A learning model for prediction
    5. A upper-bound model for prediction interval
    6. A lower: Lower-bound model for prediction interval
The method returns a dataFrame with player predictions, prediction intervals, and uncertainty levels
'''
def find_potential_fits(established, potentials, roles, model, upper, lower):
    # Merge the 'established' DataFrame with the 'roles' DataFrame based on 'playerId'
    established = established.merge(roles[['playerId', 'ip_cluster', 'pos_group']], on='playerId')
    # Merge the 'potentials' DataFrame with the 'roles' DataFrame based on 'playerId'
    potentials = potentials.merge(roles[['playerId', 'ip_cluster', 'pos_group']], on='playerId')
    predictions = pd.DataFrame()

    # Iterate over each potential player in the 'potentials' DataFrame
    for i, r in potentials.iterrows():
        # Iterate over each established player in the 'established' DataFrame
        for i2, r2 in established.iterrows():
            # Create an input DataFrame with features from both the potential and established players
            input = pd.DataFrame([
                [r.shortName, r.age, r.minutes_played_season, r['match appearances'], r.zone_1_pl,
                 r.zone_2_pl, r.zone_3_pl, r.zone_4_pl, r.zone_5_pl, r.zone_6_pl, r.chem_coef,
                 r.birthArea_name, r.passportArea_name, r.countries, r.ip_cluster, r.pos_group,
                 r2.shortName, r2.age, r2.minutes_played_season, r2['match appearances'], r2.zone_1_pl,
                 r2.zone_2_pl, r2.zone_3_pl, r2.zone_4_pl, r2.zone_5_pl, r2.zone_6_pl, r2.chem_coef,
                 r2.birthArea_name, r2.passportArea_name, r2.countries, r2.ip_cluster, r2.pos_group,
                 r2.num_transfer, r.num_transfer, r2.num_positions, r.num_positions, r2.num_formations, r.num_formations]
            ],
                columns=['shortName_y', 'age_y', 'minutes_played_season_y', 'match appearances_y',
                         'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y',
                         'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y', 'birthArea_name_y', 'passportArea_name_y',
                         'countries_y', 'ip_cluster_y', 'pos_group_y', 'shortName_x', 'weight_x',
                         'age_x', 'minutes_played_season_x', 'match appearances_x',
                         'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x',
                         'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x', 'birthArea_name_x', 'passportArea_name_x',
                         'countries_x', 'ip_cluster_x', 'pos_group_x',
                         'num_transfer_x', 'num_transfer_y', 'num_positions_x',
                         'num_positions_y', 'num_formations_x', 'num_formations_y']
            )
            # Concatenate the input DataFrame to the 'predictions' DataFrame
            predictions = pd.concat([predictions, input])

            # Perform clustering and feature engineering on the 'predictions' DataFrame
            predictions = produce_overall_cluster(predictions)

            # Calculate additional features
            predictions['same_pos'] = np.where(predictions['role_x'] == predictions['role_y'], 1, 0)
            predictions['same_role'] = np.where(predictions['pos_group_x'] == predictions['pos_group_y'], 1, 0)
            predictions['same_origin'] = np.where(predictions['birthArea_name_x'] == predictions['birthArea_name_y'], 1, 0)
            predictions['same_country'] = np.where(predictions['passportArea_name_x'] == predictions['passportArea_name_y'], 1, 0)
            predictions['played_in_same_country'] = predictions.apply(lambda x: 1 if x['countries_x'] in x['countries_y'] else 0, axis=1)

    # Extract the desired columns from the 'predictions' DataFrame
    desired_columns = ['shortName_y', 'shortName_x', 'same_origin', 'same_country', 'played_in_same_country',
                       'age_y', 'minutes_played_season_y', 'match appearances_y',
                       'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y',
                       'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y',
                       'age_x', 'minutes_played_season_x', 'match appearances_x',
                       'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x',
                       'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x', 'same_pos', 'same_role',
                       'num_transfer_x', 'num_transfer_y', 'num_positions_x',
                       'num_positions_y', 'num_formations_x', 'num_formations_y']
    extracted_columns = predictions.loc[:, desired_columns]

    # Define the desired data types for the extracted columns
    desired_data_types = {
        'same_origin': 'int32',
        'same_country': 'int32',
        'played_in_same_country': 'int64',
        'weight_y': 'float64',
        'age_y': 'int64',
        'minutes_played_season_y': 'float64',
        'match appearances_y': 'int64',
        'zone_1_pl_y': 'float64',
        'zone_2_pl_y': 'float64',
        'zone_3_pl_y': 'float64',
        'zone_4_pl_y': 'float64',
        'zone_5_pl_y': 'float64',
        'zone_6_pl_y': 'float64',
        'chem_coef_y': 'float64',
        'age_x': 'int64',
        'minutes_played_season_x': 'float64',
        'match appearances_x': 'int64',
        'zone_1_pl_x': 'float64',
        'zone_2_pl_x': 'float64',
        'zone_3_pl_x': 'float64',
        'zone_4_pl_x': 'float64',
        'zone_5_pl_x': 'float64',
        'zone_6_pl_x': 'float64',
        'chem_coef_x': 'float64',
        'same_pos': 'int32',
        'same_role': 'int32',
        'num_transfer_x': 'float64',
        'num_transfer_y': 'float64',
        'num_positions_x': 'int64',
        'num_positions_y': 'int64',
        'num_formations_x': 'int64',
        'num_formations_y': 'int64'
    }

    predicted = pd.DataFrame()
    confidence_level = 0.95  # Confidence level for the prediction interval (e.g., 0.95 for a 95% prediction interval)

    # Iterate over each row in the extracted columns DataFrame
    for i, r in extracted_columns.iterrows():
        shortName_x = r['shortName_x']
        shortName_y = r['shortName_y']
        columns_to_keep = [col for col in r.index if col not in ['shortName_x', 'shortName_y']]
        input_data = r[columns_to_keep]

        # Create a DataFrame 'df' with the input data and desired data types
        df = pd.DataFrame(input_data).T
        df = df.astype(desired_data_types)

        # Make predictions using the model and upper/lower bound models
        y_pred = model.predict(df)
        upper_pred = upper.predict(df)
        lower_pred = lower.predict(df)

        # Create a player prediction DataFrame with the results
        player_pred = pd.DataFrame(
            [[shortName_y, shortName_x, y_pred, lower_pred, upper_pred]],
            columns=['shortName_x', 'shortName_y', 'predicted_chem', 'lower_bound', 'upper_bound']
        )

        # Concatenate the player prediction DataFrame to the 'predicted' DataFrame
        predicted = pd.concat([predicted, player_pred])

    # Calculate the uncertainty level (width of the prediction interval) for each prediction
    predicted['uncertainty_level'] = predicted['upper_bound'] - predicted['lower_bound']

    # Return the final predicted DataFrame with player names, predictions, prediction intervals, and uncertainty levels
    return predicted



#----------------------------------------------------------- Archieved Uncommented----------------------------------------------

'''
def get_overview_frame_fac(df_chem, df_players):
    df_player_1_added = pd.merge(df_chem, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    df_filtered = df_player_2_added[['p1','p2', 'seasonId', 'shortName_x', 'shortName_y', 'minutes', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'factor1', 'factor2', 'df_jdi90', 'df_joi90','chemistry']]
   ## df_teams = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name = 'Scouting_Raw')
    df_filtered = df_filtered.merge(df_teams[['teamId', 'name']], left_on="teamId_x", right_on="teamId")
    df1 = df_filtered[['p1', 'shortName_x', 'role_name_x', 'areaName_x', 'factor1', 'df_jdi90', 'df_joi90', 'chemistry']].rename(
        columns={'p1': 'playerId', 'shortName_x': 'shortName', 'role_name_x': 'role_name','factor1':'factor', 'areaName_x': 'areaName'})
    df1 = df1.drop_duplicates()
    df2 = df_filtered[['p2', 'shortName_y', 'role_name_y', 'areaName_y', 'factor2', 'df_jdi90', 'df_joi90', 'chemistry']].rename(
        columns={'p2': 'playerId', 'shortName_y': 'shortName', 'role_name_y': 'role_name', 'factor2':'factor', 'areaName_y': 'areaName'})
    df2 = df2.drop_duplicates()
    players_and_chemistry = pd.concat([df1, df2])

    return players_and_chemistry
    
    
    

def derive_accurate_scores(df):
    df_final = pd.DataFrame()
    # Create an empty dictionary to store the sums
    scores = {}
    columns = ['aerial_duel', 'assist', 'carry', 'counterpressing_recovery', 'deep_completed_cross','deep_completition', 'defensive_duel', 'dribble', 'ground_duel', 'ground_duel', 'pass_into_penalty_area', 'pass_to_final_third', 'second_assist', 'sliding_tackle']
    for i in columns:
        # Select the rows where the column i is 1 and the 'accurate' column is 1
        df_a = df[[ 'playerId', i, 'accurate']]
        df_a = df_a[(df_a[i] == 1) & (df['accurate'] == 1)]

        # Group the data by playerId and sum the values in the i column
        df_g = df_a.groupby(['playerId'], as_index=False).agg({i: 'sum'})

        # Add the sum to the scores dictionary
        scores[i] = df_g
    for key, value in scores.items():
        if len(df_final) == 0:
            df_final = value
        else: df_final = df_final.merge(value, on = ['playerId'], how = 'left')

    return df_final

def create_strenghts(df):
    df['aerial_strength'] = df['accurate_aerial_duels'] / df['aerial_duel']
    df['carry_strength'] = df['accurate_carries'] / df['carry']
    df['pressing_recovery_strength'] = df['accurate_counterpressing_recoveries'] / df['counterpressing_recovery']
    df['defensive_duel_strength'] = df['accurate_defensive_duels'] / df['defensive_duel']
    df['dribbles_strength'] = df['accurate_dribbles'] / df['dribble']
    df['ground_duel_strength'] = df['accurate_ground_duels'] / df['ground_duel']
    df['sliding_strength'] = df['accurate_sliding_tackles'] / df['sliding_tackle']
    df['deep_crossing_strength'] = df['accurate_deep_completed_crosses'] / df['deep_completed_cross']
    return df



def compute_net_oi_game_silkeborg(df):
    # make a copy of the input dataframe to avoid modifying it directly
    copy = df

    # compute net offensive impact for each zone based on number of games played
    copy['zone_1_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_1_expected_vaep - copy.zone_1))
    copy['zone_2_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_2_expected_vaep - copy.zone_2))
    copy['zone_3_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_3_expected_vaep - copy.zone_3))
    copy['zone_4_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_4_expected_vaep - copy.zone_4))
    copy['zone_5_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_5_expected_vaep - copy.zone_5))
    copy['zone_6_net_oi'] = np.where(copy.games_played <= 1,
                                     copy.zone_1,
                                     (copy.zone_6_expected_vaep - copy.zone_6))

    # return the updated dataframe with net offensive impact for each zone
    return copy


def get_avg_chenm_pp(df):
    dfc = df.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    df_id = dfc[['p1', 'p2', 'teamId']]  # Extract columns that should not be scaled
    mask = ~dfc.columns.isin(['p1', 'p2', 'teamId'])  # Extract column names for scaling
    df_scale = dfc.loc[:, mask]  # Extract columns for scaling

    scale2 = MinMaxScaler()  # Initiate sclaing instance

    df_scale[df_scale.columns] = scale2.fit_transform(df_scale[df_scale.columns])  # Perform min/max scaling
    # Re-establosh dataframe with id's
    df_dk_SUPER = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)
    df_dk_SUPER['JOI_Rank'] = round(df_dk_SUPER.df_joi90.rank(pct=True) * 100, 0)
    df_dk_SUPER['JDI_Rank'] = round(df_dk_SUPER.df_jdi90.rank(pct=True) * 100, 0)
    df_dk_SUPER['Chemistry_Rank'] = round(df_dk_SUPER.chemistry.rank(pct=True) * 100, 0)
    df_dk_bif = df_dk_SUPER[df_dk_SUPER['teamId'] == 7453 ]
    bif_with_names = (df_dk_bif.merge(df_players_teams[['playerId', 'shortName', 'role_name']], left_on='p1', right_on='playerId')).rename(columns = {'df_jdi90': 'JDI', 'df_joi90': 'JOI', 'chemistry': 'Chemistry'})
    bif_with_names = (bif_with_names.merge(df_players_teams[['playerId', 'shortName', 'role_name']], left_on='p2', right_on='playerId')).rename(columns = {'df_jdi90': 'JDI', 'df_joi90': 'JOI', 'chemistry': 'Chemistry'})
    print(bif_with_names.columns)
    return bif_with_names[['p1', 'p2', 'shortName_x', 'shortName_y', 'role_name_x', 'role_name_y', 'JDI', 'JOI', 'Chemistry',  'JDI_Rank', 'JOI_Rank', 'Chemistry_Rank' ]]


def get_avg_chenm_p(df):
    chem_abilities = pd.DataFrame()
    teams  = df.teamId.unique()
    for team in teams:
        team_found = df[df['teamId'] == team]
        p1 = (team_found[['p1', 'teamId', 'df_joi90', 'df_jdi90', 'chemistry']]).rename(columns={'p1': 'playerId'})
        p2 = (team_found[['p1', 'teamId', 'df_joi90', 'df_jdi90', 'chemistry']]).rename(columns={'p1': 'playerId'})
        players = pd.concat([p1,p2])
        players = players.groupby(['playerId', 'teamId'], as_index = False).agg({'df_joi90': 'mean', 'df_jdi90':'mean', 'chemistry' :'mean' })
        chem_abilities = pd.concat([chem_abilities, players])
    dfc = chem_abilities.sort_values(by=['playerId', 'teamId'])  # Sort dataframe in ascending order(default)
    df_id = dfc[['playerId', 'teamId']]  # Extract columns that should not be scaled
    mask = ~dfc.columns.isin(['playerId', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2',
                              'combined_factor'])  # Extract column names for scaling
    df_scale = dfc.loc[:, mask]  # Extract columns for scaling

    scale2 = MinMaxScaler()  # Initiate sclaing instance
    df_scale[df_scale.columns] = scale2.fit_transform(df_scale[df_scale.columns])  # Perform min/max scaling
    # Re-establosh dataframe with id's
    df_dk_SUPER = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)
    df_dk_SUPER = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)
    df_dk_SUPER['JOI'] = round(df_dk_SUPER.df_joi90.rank(pct=True) * 100, 0)
    df_dk_SUPER['JDI'] = round(df_dk_SUPER.df_jdi90.rank(pct=True) * 100, 0)
    df_dk_SUPER['Chemistry'] = round(df_dk_SUPER.chemistry.rank(pct=True) * 100, 0)
    df_dk_bif = df_dk_SUPER[df_dk_SUPER['teamId'] == 7453 ]
    bif_with_names = df_dk_bif.merge(df_players_teams[['playerId', 'shortName', 'role_name']], on='playerId')

    return bif_with_names


'''