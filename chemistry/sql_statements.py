import pandas as pd

from helpers.student_bif_code import load_db_to_pd

'''
For now, we know prior to the statement that we only have data for the seasons:
s1: 2020/2021
s2: 2021/2022

For this reason, we only need the competition Id
The methods will have to be adjsuted if we want to scope into specific seasons
'''


def load_data(competitionIds):
    competitionIds_str = "("+ competitionIds +")"
    df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF] as t "
                                 "where t.seasonId in "
                                 "(SELECT distinct(seasonId) FROM [Development].[dbo].[sd_tableF] as t"
                                 " where t.competitionId in %s)" % competitionIds_str,
                                  db_name='Development')
    seasons = "("+(','.join(map(str, df['seasonId'].unique()))) + ")"
    df_matches_all = load_db_to_pd(
                    sql_query=  "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                                "WHERE matchId IN "
                                "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                                "where Scouting.dbo.Wyscout_Matches.seasonId in %s)" % seasons,
                                 db_name='Development')
    df_sqaud = load_db_to_pd(
                    sql_query="SELECT sq.*, m.seasonId FROM [Scouting_Raw].[dbo].[Wyscout_Match_Squad] as sq "
                              "join [Scouting_Raw].[dbo].[Wyscout_Matches_All] as m on m.matchId = sq.matchId  "
                              "WHERE sq.matchId IN "
                              "(SELECT matchId FROM Scouting.dbo.Wyscout_Matches "
                              "WHERE Scouting.dbo.Wyscout_Matches.seasonId in %s)" % seasons,
                               db_name='Scouting_Raw')

    df_keepers = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players]"
                              "where role_code2 = 'GK'",
                               db_name="Scouting_Raw")


    df_players_teams = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players] as p join Wyscout_Teams as t on t.teamId = p.currentTeamId", db_name='Scouting_Raw')
    df_events_with_goals = load_db_to_pd(sql_query = "SELECT t.eventId, goal, sumVaep "
                                                     "FROM [Scouting_Raw].[dbo].[Wyscout_Events_TypeSecondary] as t "
                                                     "join Scouting_Raw_Staging.dbo.Vaep as v on v.eventId = t.eventId "
                                                     "where t.matchId in(SELECT matchId FROM Scouting.dbo.Wyscout_Matches "
                                                     "WHERE Scouting.dbo.Wyscout_Matches.seasonId in %s) and goal = 1" % seasons,
                                                      db_name='Scouting_Raw')

    df_possesion_sequences_ordered = load_db_to_pd(sql_query="SELECT pos.*, competitionId, seasonId from sd_table_pos as pos join Scouting_Raw.dbo.Wyscout_Matches_All as m on pos.matchId = m.matchId where competitionId in %s" % competitionIds_str,
                                                              db_name='Development')


    return df, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_with_goals, df_possesion_sequences_ordered


def get_pos(comids):
    competitionIds_str = "("+ comids +")"
    df_possesion_sequences_ordered = load_db_to_pd(sql_query="SELECT pos.*, competitionId, seasonId "
                                                             "from sd_table_pos as pos "
                                                             "join Scouting_Raw.dbo.Wyscout_Matches_All as m "
                                                             "on pos.matchId = m.matchId where competitionId in %s" % competitionIds_str,
                                                              db_name='Development')
    return df_possesion_sequences_ordered


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


def get_sd_table(competitionIds):
    competitionIds_str = "("+ competitionIds +")"
    df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF] as t "
                                 "where t.seasonId in "
                                 "(SELECT distinct(seasonId) FROM [Development].[dbo].[sd_tableF] as t"
                                 " where t.competitionId in %s)" % competitionIds_str,
                                  db_name='Development')
    return df

def get_all_players():
    df = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players]", db_name='Scouting_Raw')
    return df
def get_all_teams():
    df = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name='Scouting_Raw')
    return df
def get_players_and_teams():
    df = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players] as p join Wyscout_Teams as t on t.teamId = p.currentTeamId", db_name='Scouting_Raw')
    return df

def get_transfers():
    df = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Player_Transfer]", db_name='Scouting_Raw')
    return df

def get_seasons_and_competitions():
    df = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Seasons]", db_name='Scouting_Raw')
    return df

def get_all_matches():
    df = load_db_to_pd( sql_query="select * from [Scouting_Raw].[dbo].[Wyscout_Matches_All]", db_name='Development')
    return df

def get_pos():
    df_possesion_sequences_ordered = load_db_to_pd(
        sql_query="SELECT pos.*, competitionId, seasonId from sd_table_pos as pos join Scouting_Raw.dbo.Wyscout_Matches_All as m on pos.matchId = m.matchId", db_name='Development')
    return df_possesion_sequences_ordered

def get_sd_table(competitionIds):
    df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF] as t",
                                  db_name='Development')
    return df


