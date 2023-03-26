from helpers.student_bif_code import load_db_to_pd

'''
For now, we know prior to the statement that we only have data for the seasons:
s1:  2020/2021
s2: 2021/2022

For this reason, we only need the competition Id
The methods will have to be adjsuted if we want to scope into specific seasons
'''


def load_table_comp(competitionId):
    df = load_db_to_pd(sql_query="SELECT distinct(seasonId) FROM [Development].[dbo].[sd_tableF] as t "
                                 "where t.competitionId =%s" % competitionId, db_name='Development')
    seasons = tuple((df['seasonId'].values).tolist())

    table_df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF] as t "
                                       "where t.seasonId in %s" % str(seasons), db_name='Development')

    return table_df


def get_subs(seasonId):
    s = "(" + str(seasonId) + ")"
    table_df = load_db_to_pd(
        sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Substitutions] where matchId in (SELECT matchId FROM Scouting.dbo.Wyscout_Matches WHERE Scouting.dbo.Wyscout_Matches.seasonId in %s)" % s,
        db_name='Development')
    return table_df


def get_timestamps(seasonId):
    s = "(" + str(seasonId) + ")"
    table_df = load_db_to_pd(
        sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Events_Main_Info] where matchId in (SELECT matchId FROM Scouting.dbo.Wyscout_Matches WHERE Scouting.dbo.Wyscout_Matches.seasonId in %s)" % s,
        db_name='Development')
    return table_df
