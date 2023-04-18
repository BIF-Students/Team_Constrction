import pandas as pd

from chemistry_v2.chemistry_helpers import *
from helpers.student_bif_code import load_db_to_pd


def get_distance(df):
    matches_positions = {}
    df.apply(lambda row: allocate_position(row, matches_positions), axis=1)
    avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions, {}), orient='index', columns=['matchId', 'teamId', 'competitionId', 'seasonId', 'playerId', 'avg_x', 'avg_y']).reset_index(drop=True)
    ec = compute_distances(avg_position_match_df)
    #print(ec.columns)
    return ec