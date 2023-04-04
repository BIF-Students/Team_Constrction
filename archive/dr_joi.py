import pandas as pd


def generate_joi (df, df_goals):
    #Copy dataframe
    df_events_related_ids = df
    df_goals = df_goals[['eventId', 'goal']]

    #filter for relevant columns
    df_events_related_ids  =df_events_related_ids[['eventId', 'matchId', 'playerId', 'typePrimary', 'teamId', 'sumVaep']]

    #change naming convention
    df_events_related_ids = df_events_related_ids.rename(columns = {'eventId': 'relatedEventId', 'matchId': 'matchId_2', 'playerId': 'playerId_2', 'teamId' : 'teamId_2', 'typePrimary': 'related_event', 'sumVaep': 'sumVaep_2'})
    df_goals = df_goals.rename(columns = {'eventId': 'relatedEventId'})
    df = df.rename(columns = {'matchId': 'matchId_1', 'playerId': 'playerId_1', 'teamId' : 'teamId_1', 'sumVaep': 'sumVaep_1'})

    #Merge on relate ids to obtain a dataframe with atributes of main eventId and relatedEventId in same observation
    joined_df = pd.merge(df, df_events_related_ids, how = 'left', on='relatedEventId')

    #Remove missing values
    joined_df = joined_df[ (joined_df['playerId_1'].notna()) & (joined_df['playerId_2'].notna()) & (joined_df['teamId_1'].notna())& (joined_df['teamId_2'].notna()) & (joined_df['matchId_1'].notna()) & (joined_df['matchId_2'].notna())]

    #Remplace missing vaep values with zero-values
    joined_df['sumVaep_1'] = joined_df['sumVaep_1'].fillna(0)
    joined_df['sumVaep_2'] = joined_df['sumVaep_2'].fillna(0)

    '''
    Make a filter that extracts only observatoins where;
    1: The same team is represented in both main eventId and relatedEventID
    2: The same match is represented in both main eventId and relatedEventID
    3: It is not a sequence of events produced by the same player
    4 & 5: Not relevant events ar removed from both typePrimary and related events
    '''
    joined_df_filtered = joined_df[(joined_df.teamId_1 == joined_df.teamId_2)
                                   & (joined_df.matchId_1 == joined_df.matchId_2)
                                   & (joined_df.playerId_1 != joined_df.playerId_2)
                                   & (~joined_df.typePrimary.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))
                                   & (~joined_df.related_event.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))]

    #Order player ids such that the player with the lowest Id will be represented in the p1 attribute
    joined_df_filtered['p1'] = np.where(joined_df_filtered['playerId_1'] > joined_df_filtered['playerId_2'], joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])
    joined_df_filtered['p2'] = np.where(joined_df_filtered['playerId_1'] == joined_df_filtered['p1'],joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])

    # Drop unnecessary columns
    joined_df_filtered = joined_df_filtered.drop(['playerId_1', 'playerId_2'], axis=1)
    joined_df_filtered = joined_df_filtered.drop_duplicates(subset=["eventId"], keep=False)
    joined_df_filtered  = joined_df_filtered.merge(df_goals, on='relatedEventId', how = 'left').reset_index(drop=True)
    joined_df_filtered = joined_df_filtered.loc[:, joined_df_filtered.columns != 'goal_x']
    joined_df_filtered = joined_df_filtered.rename(columns = {'goal_y' : 'goal'})
    joined_df_filtered['joi'] = joined_df_filtered['sumVaep_1'] + joined_df_filtered['sumVaep_2']

    #Compute jois as a sum of the sumVaep values related to the main event and related event
    joi_df = joined_df_filtered.groupby(['matchId_1', 'matchId_2', 'p1', 'p2', 'teamId_1', 'teamId_2'], as_index=False).agg({
                                                                                                                        'joi':'sum',
                                                                                                                        'goal': 'sum',
                                                                                                                        'assist': 'sum',
                                                                                                                        'second_assist': 'sum',
                                                                                                                        })
    return  joi_df, joined_df_filtered

def getJoi_old(df, df_goals):
 df_joi, filtered= generate_joi(df, df_goals)
 return df_joi, filtered
