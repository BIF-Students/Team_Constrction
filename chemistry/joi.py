# Import necessary modules
import pandas as pd

from chemistry.chemistry_helpers import *

def check_actions(row1, row2):
    if (((row1['assist'] == 1) | (row1['back_pass'] == 1) | (row1['carry'] == 1) |(row1['cross'] == 1) |(row1['deep_completed_cross'] == 1) |(row1['dribble'] == 1) | (row1['forward_pass'] == 1) | (row1['free_kick_cross'] == 1) | (row1['goal'] == 1) | (row1['forward_pass'] == 1) |(row1['head_pass'] == 1) |(row1['key_pass'] == 1) | (row1['lateral_pass'] == 1) | (row1['linkup_play'] == 1) | (row1['long_pass'] == 1) | (row1['linkup_play'] == 1) |(row1['long_pass'] == 1) |(row1['offensive_duel'] == 1) |(row1['pass_into_penalty_area'] == 1) | (row1['pass_to_final_third']) | (row1['progressive_pass'] == 1) | (row1['progressive_run'] == 1) | (row1['second_assist']  == 1) | (row1['short_or_medium_pass']  == 1) | (row1['shot_after_corner']  == 1) | (row1['shot_after_free_kick']  == 1) | (row1['shot_after_throw_in']  == 1) | (row1['smart_pass']  == 1) | (row1['third_assist']  == 1) | (row1['through_pass']  == 1))
        & ((row2['assist'] == 1) | (row2['back_pass'] == 1) | (row2['carry'] == 1) |(row2['cross'] == 1) |(row2['deep_completed_cross'] == 1) |(row2['dribble'] == 1) | (row2['forward_pass'] == 1) | (row2['free_kick_cross'] == 1) | (row2['goal'] == 1) | (row2['forward_pass'] == 1) |(row2['head_pass'] == 1) |(row2['key_pass'] == 1) | (row2['lateral_pass'] == 1) | (row2['linkup_play'] == 1) | (row2['long_pass'] == 1) | (row2['linkup_play'] == 1) |(row2['long_pass'] == 1) |(row2['offensive_duel'] == 1) |(row2['pass_into_penalty_area'] == 1) | (row2['pass_to_final_third']) | (row2['progressive_pass'] == 1) | (row2['progressive_run'] == 1) | (row2['second_assist']  == 1) | (row2['short_or_medium_pass']  == 1) | (row2['shot_after_corner']  == 1) | (row2['shot_after_free_kick']  == 1) | (row2['shot_after_throw_in']  == 1) | (row2['smart_pass']  == 1) | (row2['third_assist']  == 1) | (row2['through_pass']  == 1))):
        return True
    else: return False


def generate_joi(df_pos, pos_ids):
 player_joi_dict = {}
 for pos_id in pos_ids:
  df_for_analysis = df_pos[df_pos['possessionId'] == pos_id]
  df_for_analysis = df_for_analysis.sort_values(by='possessionEventIndex', ascending=True).reset_index(drop=True)
  for i2 in range(len(df_for_analysis) - 1):
   row2 = df_for_analysis.iloc[i2]
   p1 = row2['playerId']
   teamId = row2['teamId']
   matchId = row2['matchId']
   next_val = df_for_analysis.iloc[i2 + 1]
   teamId_2 = next_val['teamId']
   p2 = next_val['playerId']
   key = matchId, min(p1, p2), max(p1, p2)
   v1 = row2['sumVaep']
   v2 = next_val['sumVaep']
   assist = next_val['assist'] + row2['assist']
   second_assist = next_val['second_assist'] + row2['second_assist']
   goal = next_val['goal'] + row2['goal']
   sum = v1+v2
   if (p1 != p2) & (teamId_2 == teamId) & (check_actions(row2, next_val)):
    if key in player_joi_dict:
       item = player_joi_dict[key].get('sumVaep')
       second_assists = player_joi_dict[key].get('second_assists') + second_assist
       assists = player_joi_dict[key].get('assists') + assist
       goals = player_joi_dict[key].get('goals') + goal
       item.append(sum)
       player_joi_dict[key] = {'matchId': matchId, 'teamId': teamId, 'p1': min(p1,p2), 'p2': max(p1,p2), 'second_assists': second_assists, 'assists': assists, 'goals': goals, 'sumVaep': item}
    else:
       player_joi_dict[key] = {'matchId': matchId, 'teamId': teamId, 'p1': min(p1, p2), 'p2': max(p1, p2), 'second_assists': second_assist, 'assists': assist, 'goals': goal, 'sumVaep': [sum]}
 return player_joi_dict

def compute_total_vaep_game(dict_obj):
    for key in dict_obj:
    # Get the sum of the 'sumVaep' values for the current key
        sum_vaep = sum(dict_obj[key]['sumVaep'])
    # Add a new key-value pair to the nested dictionary with the sum as the value
        dict_obj[key]['joi'] = sum_vaep
    return dict_obj

def get_joi(df_pos):
 joi_dict = generate_joi(df_pos.copy(), (df_pos['possessionId'].unique()))
 joi_dict_v2 = compute_total_vaep_game(joi_dict)
 df_joi = pd.DataFrame.from_dict(joi_dict_v2, orient='index')
 df_joi = df_joi[['matchId', 'teamId' , 'p1', 'p2', 'second_assists', 'assists', 'goals', 'joi']]
 return df_joi

