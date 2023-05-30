

'''
This method is used to determine the relative player impact of a player in a game
The method takes two arguments
    1. A dataframe with data specific to a player
    2. A dataframe with data specific of a team
the method return a dataframe with info on the impact of a player in a match in all six zones of the pitch
'''
def compute_relative_player_impact(df_player, df_team):
    # Merge two dataframes based on matchId and teamId using inner join
    df_full = pd.merge(df_player, df_team, on=['matchId', 'teamId'], how='inner')

    # Compute relative player impact for each of the six zones
    # If the number of passes in a zone is greater than 0, compute the impact as the ratio of passes completed to passes attempted in that zone
    # If the number of passes in a zone is 0, set the impact for that zone to 0
    df_full['zone_1_imp'] = np.where(df_full.zone_1_pl > 0, df_full.zone_1_pl / df_full.zone_1_t, 0)
    df_full['zone_2_imp'] = np.where(df_full.zone_2_pl > 0, df_full.zone_2_pl / df_full.zone_2_t, 0)
    df_full['zone_3_imp'] = np.where(df_full.zone_3_pl > 0, df_full.zone_3_pl / df_full.zone_3_t, 0)
    df_full['zone_4_imp'] = np.where(df_full.zone_4_pl > 0, df_full.zone_4_pl / df_full.zone_4_t, 0)
    df_full['zone_5_imp'] = np.where(df_full.zone_5_pl > 0, df_full.zone_5_pl / df_full.zone_5_t, 0)
    df_full['zone_6_imp'] = np.where(df_full.zone_6_pl > 0, df_full.zone_6_pl / df_full.zone_6_t, 0)

    # Return the dataframe with the relative player impact for each zone
    return df_full


'''
This method is used to find the zones and their counts for each player.
The method takes a dataframe as input.
The method performs the following steps:
    1. Uses the pandas "get_dummies" method to create dummy variables for each "zone" category, prefixing them with "zone_" in the column names.
    2. Concatenates the "zone_dummies" DataFrame with the input DataFrame "df" column-wise (axis=1).
    3. Groups the DataFrame by playerId, matchId, and teamId columns and calculates the sum of each zone column.
    4. Resets the index to create a flat DataFrame with the original columns and aggregated zone columns.
    5. Renames the aggregated zone columns, appending "_pl" to each column name to denote that the counts represent the number of times each player appeared in each zone.
    6. Returns the modified DataFrame with aggregated zone counts.
'''
def find_zones_and_counts_pl(df):
    # Use the pandas "get_dummies" method to create dummy variables for each "zone" category,
    # prefixing them with "zone_" in the column names.
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # Concatenate the "zone_dummies" DataFrame with the input DataFrame "df" column-wise (axis=1).
    df = pd.concat([df, zone_dummies], axis=1)

    # Group the DataFrame by playerId, matchId, and teamId columns, and calculate the sum of each zone column.
    # Reset the index to create a flat DataFrame with the original columns and aggregated zone columns.
    df = df.groupby([df.playerId, df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum'
    })

    # Rename the aggregated zone columns, appending "_pl" to each column name to denote that the counts
    # represent the number of times each player appeared in each zone.
    df = df.rename(columns={
        'zone_1': 'zone_1_pl',
        'zone_2': 'zone_2_pl',
        'zone_3': 'zone_3_pl',
        'zone_4': 'zone_4_pl',
        'zone_5': 'zone_5_pl',
        'zone_6': 'zone_6_pl'
    })

    # Return the modified DataFrame with aggregated zone counts.
    return df


'''
This method is used to find the zones and their counts for each team.
The method takes a dataframe as input.
The method performs the following steps:
    1. Uses the pandas "get_dummies" method to create dummy variables for each "zone" category, prefixing them with "zone_" in the column names.
    2. Concatenates the "zone_dummies" DataFrame with the input DataFrame "df" column-wise (axis=1).
    3. Groups the DataFrame by matchId and teamId columns and calculates the sum of each zone column.
    4. Resets the index to create a flat DataFrame with the original columns and aggregated zone columns.
    5. Renames the aggregated zone columns, appending "_t" to each column name to denote that the counts represent the number of times each team engaged in actions within each zone.
    6. Returns the modified DataFrame with aggregated zone counts.
'''
def find_zones_and_counts_t(df):
    #extact dummies from zone columns
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    #Append zone dummies to original dataframe - columnswise
    df = pd.concat([df, zone_dummies], axis=1)

    #Groupby statement to compute counts for the amount of times a teams has
    #engaged in actoins within a zone
    df = df.groupby([df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum'})
    df = df.rename(columns={'zone_1': 'zone_1_t',
                            'zone_2': 'zone_2_t',
                            'zone_3': 'zone_3_t',
                            'zone_4': 'zone_4_t',
                            'zone_5': 'zone_5_t',
                            'zone_6': 'zone_6_t'
                            })

    return df

'''
This method is used to determine in which zone an action has happened - this is used in a lambda expression
The method takes one argument
    1. A dataframe with one row
The name of the zone is returned as an integer
'''
def find_zone_chemistry(frame):
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

def non_possession_action(row):
    x = row['typePrimary']
    non_possession = ['interception', 'infraction', 'shot_against', 'clearance']
    column_list = ['aerial_duel', 'counterpressing_recovery', 'defensive_duel', 'dribbled_past_attempt', 'loose_ball_duel', 'recovery', 'sliding_tackle']
    if x in non_possession:
        return 1
    else:
        for col in column_list:
            if row[col] == 1:
                return 1
    return 0


def getResponsibilityShare(df):
    # Insert a new column 'nonPosAction' at index 57 to mark if the action is defensive
    df.insert(57, 'nonPosAction', df.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True)

    # Filter the dataframe for defensive actions
    df_def_actions_player = df[df['nonPosAction'] == 1]

    # Calculate the zone for each defensive action in the dataframe
    df_def_actions_player['zone'] = df_def_actions_player.apply(lambda row: find_zone_chemistry(row), axis=1)

    # Find the zones and counts of defensive actions for each player
    df_players_actions = find_zones_and_counts_pl(df_def_actions_player)

    # Find the zones and counts of defensive actions for the team
    df_team_actions = find_zones_and_counts_t(df_def_actions_player)

    # Compute the relative player impact based on defensive actions
    df_player_share = compute_relative_player_impact(df_players_actions, df_team_actions)

    # Return the computed player share and the dataframe with player actions in different zones
    return df_player_share, df_players_actions
