def find_zone_chemistry(row):
    s = ""
    #  id = row['id']
    # print(row)
    x = row['x']
    y = row['y']
    if (x >= 0 and x <= 33 and y >= 0 and y <= 33):
        s = 1
    elif (x >= 0 and x <= 33 and y > 33 and y <= 67):
        s = 2
    elif (x >= 0 and x <= 33 and y > 67 and y <= 100):
        s = 3
    elif (x > 33 and x <= 67 and y >= 0 and y <= 33):
        s = 4
    elif (x > 33 and x <= 67 and y > 33 and y <= 67):
        s = 5
    elif (x > 33 and x <= 67 and y > 67 and y <= 100):
        s = 6
    elif (x > 67 and x <= 100 and y >= 0 and y <= 33):
        s = 7
    elif (x > 67 and x <= 100 and y > 33 and y <= 67):
        s = 8
    elif (x >= 67 and x <= 100 and y >= 67 and y <= 100):
        s = 9
    else:
        s = 0
    return s

def generate_joi (df, df_related_ids):
    df_events_related_ids_2 = df_related_ids[['eventId', "relatedEventId"]]
    related_id_restored_df = pd.merge(df, df_events_related_ids_2, how = 'left', on='eventId')

    extracted = related_id_restored_df[['eventId', 'typePrimary', 'playerId', 'teamId', 'matchId', 'sumVaep']]
    extracted = extracted.rename(columns ={'eventId': 'relatedEventId', 'typePrimary': 'related_event', 'playerId': 'playerId_2', 'teamId': 'teamId_2', 'matchId': 'matchId_2'})

    merged =pd.merge(extracted, related_id_restored_df, how = 'right', on='relatedEventId')
    joined_df = merged[['eventId', 'relatedEventId', 'typePrimary', 'related_event', 'playerId', 'playerId_2', 'teamId', 'teamId_2', 'matchId', 'matchId_2', 'sumVaep_x', 'sumVaep_y']]

    joined_df = joined_df.rename(columns ={'playerId': 'playerId_1' ,'matchId': 'matchId_1', 'teamId': 'teamId_1' ,'sumVaep_x': 'sumVaep_1', 'sumVaep_y': 'sumVaep_2'})
    joined_df = joined_df[(joined_df['playerId_1'].notna()) & (joined_df['playerId_2'].notna()) & (joined_df['teamId_1'].notna()) & (joined_df['teamId_2'].notna()) & (joined_df['matchId_1'].notna()) & (joined_df['matchId_2'].notna())]

    joined_df['playerId_1'] = joined_df['playerId_1'].astype(int)
    joined_df['playerId_2'] = joined_df['playerId_2'].astype(int)
    joined_df['teamId_1'] = joined_df['teamId_1'].astype(int)
    joined_df['teamId_2'] = joined_df['teamId_2'].astype(int)
    joined_df['matchId_1'] = joined_df['matchId_1'].astype(int)
    joined_df['matchId_2'] = joined_df['matchId_2'].astype(int)

    joined_df['sumVaep_1'] = joined_df['sumVaep_1'].fillna(0)
    joined_df['sumVaep_2'] = joined_df['sumVaep_2'].fillna(0)

    joined_df_filtered = joined_df[(joined_df.teamId_1 == joined_df.teamId_2)
                               & (joined_df.matchId_1 == joined_df.matchId_2)
                               & (joined_df.playerId_1 != joined_df.playerId_2)
                               & (~joined_df.typePrimary.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))
                               & (~joined_df.related_event.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))]

    joined_df_filtered['p1'] = np.where(joined_df_filtered['playerId_1'] > joined_df_filtered['playerId_2'], joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])
    joined_df_filtered['p2'] = np.where(joined_df_filtered['playerId_1'] == joined_df_filtered['p1'], joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])
    joined_df_filtered = joined_df_filtered.drop(['playerId_1', 'playerId_2'], axis =1)

    joined_df_filtered['joi'] = joined_df_filtered['sumVaep_1'] + joined_df_filtered['sumVaep_2']

    joi_df = joined_df_filtered.groupby(['matchId_1', 'matchId_2', 'p1', 'p2', 'teamId_1', 'teamId_2'], as_index=False)['joi'].sum()
    return joi_df
