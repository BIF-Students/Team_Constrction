# Import necessary modules
import pandas as pd


'''
This is a helper method used in a lambda expression. It is responsible for a subsampling process where all 
observations among interactions are a part of offensive interactions. 
The method takes two arguments
    1. A row from an event data stram
    2. The successive row on the same data stream 
The method returns a boolean value being either true or false based the rows contain data that is accepted as valud 
offensive actions
'''
def check_actions(row1, row2):
    # Check if the actions in row1 and row2 indicate a valid sequence of actions

    # Check if any of the valid action flags in row1 are set to 1
    if (((row1['assist'] == 1) | (row1['back_pass'] == 1) | (row1['carry'] == 1) | (
                 row1['cross'] == 1) | (row1['dribble'] == 1) | (row1['forward_pass'] == 1) | (
                 row1['free_kick_cross'] == 1) | (row1['goal'] == 1) | (row1['head_pass'] == 1) | (
                 row1['key_pass'] == 1) | (row1['lateral_pass'] == 1) | (row1['linkup_play'] == 1) | (
                 row1['second_assist'] == 1) | (row1['short_or_medium_pass'] == 1) | (row1['shot_after_corner'] == 1) | (
                 row1['shot_after_free_kick'] == 1) | (row1['shot_after_throw_in'] == 1) | (row1['smart_pass'] == 1) | (
                 row1['third_assist'] == 1) | (row1['through_pass'] == 1))

            # Check if any of the valid action flags in row2 are set to 1
            & ((row2['assist'] == 1) | (row2['back_pass'] == 1) | (row2['carry'] == 1) | (row2['cross'] == 1) | (
                    row2['deep_completed_cross'] == 1) | (row2['dribble'] == 1) | (row2['forward_pass'] == 1) | (
                       row2['free_kick_cross'] == 1) | (row2['goal'] == 1) | (row2['head_pass'] == 1) | (
                       row2['key_pass'] == 1) | (row2['lateral_pass'] == 1) | (row2['linkup_play'] == 1) | (
                       row2['long_pass'] == 1) | (row2['offensive_duel'] == 1) | (
                       row2['pass_into_penalty_area'] == 1) | (row2['pass_to_final_third']) | (
                       row2['progressive_pass'] == 1) | (row2['progressive_run'] == 1) | (
                       row2['second_assist'] == 1) | (row2['short_or_medium_pass'] == 1) | (
                       row2['shot_after_corner'] == 1) | (row2['shot_after_free_kick'] == 1) | (
                       row2['shot_after_throw_in'] == 1) | (row2['smart_pass'] == 1) | (row2['third_assist'] == 1) | (
                       row2['through_pass'] == 1))):

        # Return True if both row1 and row2 have valid actions
        return True
    else:
        # Return False if either row1 or row2 does not have valid actions
        return False


'''
This method is responsible for computing the JOI produced by pairs of players across all matches in season 21/22
The method accept two dataframes as arguments:
    1. A dataframe consisting of event data with possession ids attached allowing for the identification of possession sequences and thereby interactions
    2. A dataframe with all unique possessionids
The method returns a dictionary with keys consisting of 'matchId' 
and the playerIds of each pair with the lowest id always presented first where alle assists, goals and vaep productions are saved by each pair in amatch
NOTE: Assists and goals can be removed - never used

'''
def generate_joi(df_pos, pos_ids):
    # Initialize an empty dictionary to store player joint offensive impact (JOI) data
    player_joi_dict = {}

    # Iterate over each position ID
    for pos_id in pos_ids:
        # Filter the dataframe to get the data for the current position ID
        df_for_analysis = df_pos[df_pos['possessionId'] == pos_id]

        # Sort the dataframe by possession event index in ascending order and reset the index
        df_for_analysis = df_for_analysis.sort_values(by='possessionEventIndex', ascending=True).reset_index(drop=True)

        # Iterate over each row in the dataframe, except the last one
        for i2 in range(len(df_for_analysis) - 1):
            # Get the current row and its relevant data
            row2 = df_for_analysis.iloc[i2]
            p1 = row2['playerId']
            teamId = row2['teamId']
            matchId = row2['matchId']

            # Get the next row and its relevant data
            next_val = df_for_analysis.iloc[i2 + 1]
            teamId_2 = next_val['teamId']
            p2 = next_val['playerId']

            # Create a key to identify the player pair uniquely
            key = matchId, min(p1, p2), max(p1, p2)

            # Get the offensive impact values and other statistics from the current and next rows
            v1 = row2['sumVaep']
            v2 = next_val['sumVaep']
            assist = next_val['assist'] + row2['assist']
            second_assist = next_val['second_assist'] + row2['second_assist']
            goal = next_val['goal'] + row2['goal']
            sum = v1 + v2

            # Check if the conditions for valid JOI data are met
            if (p1 != p2) and (teamId_2 == teamId) and (check_actions(row2, next_val)):
                # If the key already exists in the dictionary, update the values
                if key in player_joi_dict:
                    item = player_joi_dict[key].get('sumVaep')
                    second_assists = player_joi_dict[key].get('second_assists') + second_assist
                    assists = player_joi_dict[key].get('assists') + assist
                    goals = player_joi_dict[key].get('goals') + goal
                    item.append(sum)
                    player_joi_dict[key] = {
                        'matchId': matchId, 'teamId': teamId, 'p1': min(p1, p2), 'p2': max(p1, p2),
                        'second_assists': second_assists, 'assists': assists, 'goals': goals, 'sumVaep': item
                    }
                # If the key does not exist in the dictionary, create a new entry
                else:
                    player_joi_dict[key] = {
                        'matchId': matchId, 'teamId': teamId, 'p1': min(p1, p2), 'p2': max(p1, p2),
                        'second_assists': second_assist, 'assists': assist, 'goals': goal, 'sumVaep': [sum]
                    }

    # Return the dictionary containing the player JOI data
    return player_joi_dict


'''
This method is responsible for computing the total vaep production pre each pair of players 
The method takes one argument
    1. A dictionary containing identified by their playerids combined with a matchId 
        All VAEP production values stored in the dictionary for their interaction are summed and added as an attribute in the dict object
The method return the updated dict object
'''
def compute_total_vaep_game(dict_obj):
    for key in dict_obj:
    # Get the sum of the 'sumVaep' values for the current key
        sum_vaep = sum(dict_obj[key]['sumVaep'])
    # Add a new key-value pair to the nested dictionary with the sum as the value
        dict_obj[key]['joi'] = sum_vaep
    return dict_obj

'''
This is a method responsible for activating all JOI related method, conduct needed conversions and return a
dataframe containing observations of pairs of players and their VAEP contribution across a season
'''
def get_joi(df_pos):
    # Generate the Joint Offensive Impact (JOI) dictionary by calling the generate_joi function
    joi_dict = generate_joi(df_pos.copy(), df_pos['possessionId'].unique())

    # Compute the total value added/expected points for the game using the JOI dictionary
    joi_dict_v2 = compute_total_vaep_game(joi_dict)

    # Convert the JOI dictionary to a pandas DataFrame
    df_joi = pd.DataFrame.from_dict(joi_dict_v2, orient='index')

    # Select the desired columns in the DataFrame
    df_joi = df_joi[['matchId', 'teamId', 'p1', 'p2', 'second_assists', 'assists', 'goals', 'joi']]

    # Return the resulting DataFrame containing the JOI data
    return df_joi


