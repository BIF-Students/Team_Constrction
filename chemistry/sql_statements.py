import pandas as pd

from helpers.student_bif_code import load_db_to_pd

'''
For now, we know prior to the statement that we only have data for the seasons:
s1: 2020/2021
s2: 2021/2022

For this reason, we only need the competition Id
The methods will have to be adjsuted if we want to scope into specific seasons
'''


def load_data(competitionId):
    df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF] as t "
                                 "where t.seasonId in "
                                 "(SELECT distinct(seasonId) FROM [Development].[dbo].[sd_tableF] as t"
                                 " where t.competitionId =%s)" % competitionId,
                                  db_name='Development')

    season_21_22 = max(df['seasonId'])
    df_matches_all = load_db_to_pd(
                    sql_query=  "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                                "WHERE matchId IN "
                                "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                                "where Scouting.dbo.Wyscout_Matches.seasonId = %s)" % season_21_22,
                                 db_name='Development')
    df_sqaud = load_db_to_pd(
                    sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Match_Squad] "
                              "WHERE matchId IN "
                              "(SELECT matchId FROM Scouting.dbo.Wyscout_Matches "
                              "WHERE Scouting.dbo.Wyscout_Matches.seasonId = %s)" % season_21_22,
                               db_name='Scouting_Raw')
    df_keepers = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players]"
                              "where role_code2 = 'GK'",
                               db_name="Scouting_Raw")

    df_related_ids = load_db_to_pd(sql_query = "select s.*, e.relatedEventId from sd_tableF as s left join [Scouting_Raw].[dbo].[Wyscout_Events_Main_Info] as e on e.eventId = s.eventId where seasonId = %s" % season_21_22, db_name='Development')

    df_players_teams = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players] as p join Wyscout_Teams as t on t.teamId = p.currentTeamId", db_name='Scouting_Raw')


    return df, df_matches_all, df_sqaud, df_keepers, df_related_ids, df_players_teams

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


