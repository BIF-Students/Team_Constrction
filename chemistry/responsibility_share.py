from chemistry.chemistry_helpers import find_zones_and_counts_pl, find_zones_and_counts_t, \
    compute_relative_player_impact, find_zone_chemistry
from helpers.helperFunctions import non_possession_action

def getResponsibilityShare(df):
    df.insert(57, 'nonPosAction', df.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True) # Mark if action is defensive
    df_def_actions_player = df[df['nonPosAction'] == 1] # Filter dataframe for defensive actions
    df_def_actions_player['zone'] = df_def_actions_player.apply(lambda row: find_zone_chemistry(row), axis = 1) # Filter dataframe for defensive actions
    df_players_actions = find_zones_and_counts_pl(df_def_actions_player)
    df_team_actions = find_zones_and_counts_t(df_def_actions_player)
    df_player_share = compute_relative_player_impact(df_players_actions, df_team_actions)
    return df_player_share
