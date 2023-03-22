import pandas as pd

from chemistry.chemistry_helpers import allocate_position, get_average_positions, compute_distances
from helpers.student_bif_code import load_db_to_pd

def getDistance(df):
    matches_positions = {}
    df.apply(lambda row: allocate_position(row, matches_positions), axis = 1)
    avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions, {}), orient='index', columns=['matchId', 'teamId', 'playerId', 'avg_x', 'avg_y']).reset_index(drop=True)
    df_matches_and_teams = (df[['matchId', 'teamId']].drop_duplicates()).reset_index(drop=True)

    ec_dict = compute_distances(df_matches_and_teams, avg_position_match_df)
    ec_df = pd.DataFrame.from_dict(ec_dict, orient='index', columns=['matchId', 'teamId', 'player1', 'player2', 'distance']).reset_index(drop=True)
    return ec_df