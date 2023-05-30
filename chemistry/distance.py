import numpy as np
import pandas as pd



'''
This method computes the distances between players in a DataFrame.
It takes one argument:
    A DataFrame containing player data
The method returns the merged DataFrame with the added 'distance' column being the average distance between two players in a game.
'''
def compute_distances(df_full):
    #Merge dataframw with itself to obtain pairing of players
    merged = df_full.merge(df_full, on=['matchId', 'teamId'], suffixes=('1', '2'))

    #Extract observatoins where players are not paired with itself and
    #Ensure that the player with the lowest playerId is always found on the 'playerId1' column
    merged = merged[(merged.playerId1 != merged.playerId2) & (merged.playerId1 < merged.playerId2)]
    x1 = merged['avg_x1'].values
    y1 = merged['avg_y1'].values
    x2 = merged['avg_x2'].values
    y2 = merged['avg_y2'].values

    # Calculate the Euclidean distance between two points and store the result in the 'distance' column
    merged['distance'] = np.linalg.norm(np.column_stack((x1 - x2, y1 - y2)), axis=1)
    return merged

'''
This method is used to determine the average position of a player in a game
The method takes two arguments
    1. A dictionary with values related to the position of a player in a game
    2. A new dict to add the average position of tha player in
The method returns the new dictionary containing only the player and his average position
'''
def get_average_positions(key_values, newDict):
    # Loop through each key in key_values
    for key in key_values:
        # Create a new dictionary entry for the key with the following keys and values
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'playerId': key_values[key]['playerId'],
                        'teamId': key_values[key]['teamId'],
                        'avg_x': sum(key_values[key]['x'])/len(key_values[key]['x']),
                        'avg_y': sum(key_values[key]['y'])/len(key_values[key]['y'])}
    # Return the new dictionary
    return newDict


'''
This method is used to create a dictionary with data on two players average positoins during a match
The method takes two arguments 
    1. A dataframe with one row
    2. A dictionary reponsible for storing the players and their relative positions to each other in a game
The method returns the dictionary with the populated data
'''
def allocate_position(frame, matches_positions):
    # extract relevant values from row
    x = frame['x']
    y = frame['y']
    mId = frame['matchId']
    pId = frame['playerId']
    tId = frame['teamId']

    # check if the (matchId, playerId) pair is already in the dictionary
    if (mId, pId) in matches_positions:
        # if it is, add the new x and y values to the existing list
        matches_positions[(mId, pId)]['x'].append(x)
        matches_positions[(mId, pId)]['y'].append(y)
    else:
        # if it isn't, create a new entry in the dictionary with the matchId, teamId, playerId, and the x and y values
        matches_positions[(mId, pId)] = {'matchId': mId, 'teamId': tId, 'playerId': pId, 'x': [x], 'y': [y]}



'''
This method is responsible for computing the average euclidian distance between two players in a match 
The method takes one argument
    1. This is a dataframe containing an event data stream for all matches during the season 21/22
    The method returns a dataframe with observations containing matches two players and their average distance two each other in a particular match
'''
def getDistance(df):
    # Dictionary to store positions for each match
    matches_positions = {}

    # Allocate positions for each row in the dataframe and update the matches_positions dictionary
    df.apply(lambda row: allocate_position(row, matches_positions), axis=1)

    # Convert the matches_positions dictionary to a dataframe and compute the average positions for each match, team, and player
    avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions, {}), orient='index',
                                                   columns=['matchId', 'teamId', 'playerId', 'avg_x',
                                                            'avg_y']).reset_index(drop=True)

    # Compute the distances between players based on their average positions
    ec = compute_distances(avg_position_match_df)

    # Return the dataframe with distances between players, and the dataframe with average positions for each match, team, and player
    return ec[['matchId', 'teamId', 'playerId1', 'playerId2', 'distance']], avg_position_match_df


