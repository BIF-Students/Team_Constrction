# Import necessary modules
import pandas as pd

from chemistry_v2.chemistry_helpers import *


def generate_joi(df_pos, pos_ids):
 player_joi_dict = {}
 for pos_id in pos_ids:
  df_for_analysis = df_pos[df_pos['possessionId'] == pos_id]
  for i2 in range(len(df_for_analysis) - 1):
   row2 = df_for_analysis.iloc[i2]
   p1 = row2['playerId']
   teamId = row2['teamId']
   matchId = row2['matchId']
   seasonId = row2['seasonId']
   next_val = df_for_analysis.iloc[i2 + 1]
   p2 = next_val['playerId']
   key = matchId, seasonId, min(p1, p2), max(p1, p2)
   v1 = row2['sumVaep']
   v2 = next_val['sumVaep']
   assist = next_val['assist'] + row2['assist']
   second_assist = next_val['second_assist'] + row2['second_assist']
   goal = next_val['goal'] + row2['goal']
   sum = v1+v2
   if p1 != p2:
    if key in player_joi_dict:
       item = player_joi_dict[key].get('sumVaep')
       second_assists = player_joi_dict[key].get('second_assists') + second_assist
       assists = player_joi_dict[key].get('assists') + assist
       goals = player_joi_dict[key].get('goals') + goal
       item.append(sum)
       player_joi_dict[key] = {'matchId': matchId, 'seasonId': seasonId, 'teamId': teamId, 'p1': min(p1,p2), 'p2': max(p1,p2), 'second_assists': second_assists, 'assists': assists, 'goals': goals, 'sumVaep': item}
    else:
       player_joi_dict[key] = {'matchId': matchId, 'seasonId': seasonId, 'teamId': teamId, 'p1': min(p1, p2), 'p2': max(p1, p2), 'second_assists': second_assist, 'assists': assist, 'goals': goal, 'sumVaep': [sum]}
 return player_joi_dict

def compute_total_vaep_game(dict_obj):
    for key in dict_obj:
    # Get the sum of the 'sumVaep' values for the current key
        sum_vaep = sum(dict_obj[key]['sumVaep'])
    # Add a new key-value pair to the nested dictionary with the sum as the value
        dict_obj[key]['joi'] = sum_vaep
    return dict_obj

def getJoi(df_pos):
 joi_dict = generate_joi(df_pos.copy(), (df_pos['possessionId'].unique()))
 joi_dict_v2 = compute_total_vaep_game(joi_dict)
 df_joi = pd.DataFrame.from_dict(joi_dict_v2, orient='index')
 df_joi = df_joi[['matchId', 'teamId', 'seasonId', 'p1', 'p2', 'second_assists', 'assists', 'goals', 'joi']]
 return df_joi
