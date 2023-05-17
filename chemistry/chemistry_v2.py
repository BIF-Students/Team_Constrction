# Import necessary modules
import pandas as pd
from matplotlib import pyplot as plt
from chemistry.distance import *
from chemistry.jdi import *
from chemistry.joi import *
from chemistry.netoi import *
from chemistry.responsibility_share import *
from chemistry.smallTest import test_players_in_a_match
from chemistry.sql_statements import *
from helpers.helperFunctions import possession_action
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
#sd_table, df_matches_all, df_squad, df_players_teams, df_events_goals, df_pos = load_data(competitionIds="412")
#df_matches_all = load_db_to_pd(sql_query="select * from [Scouting_Raw].[dbo].[Wyscout_Matches_All]", db_name='Development')
#df_pos = get_pos()
#sd_table = get_sd_table()
df_all_players = get_all_players()
df_all_teams = get_all_teams()
df_players_teams = get_players_and_teams()
df_keepers = df_all_players[df_all_players['role_code2'] == 'GK']
df_matches_all = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_matches_all.csv")

#, 426, 335, 707, 635, 852, 198, 795, 524, 364
df_squad = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_squad.csv", decimal=",", sep=(';'))
df_squad = df_squad.merge(df_matches_all[['matchId', 'seasonId']], on='matchId')
df_pos = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_pos.csv", decimal=",", sep=(';'))
sd_table = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_table.csv", decimal=",", sep=(';'))
def_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/def_success.csv", decimal=",")
air_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/air_suc.csv", decimal=",")
def_suc_tot = create_def_succes_frame(def_suc, air_suc)
off_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/off_success.csv", decimal=",", sep=(','))

comp_ids = sd_table.competitionId.unique()
seasons = []
for i in comp_ids:
    df_a = sd_table[sd_table['competitionId'] == i]
    seasons.append(max(df_a.seasonId))



df_successes = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_success.csv")

merged = sd_table.merge(df_successes, left_on='eventId', right_on='id')


len(merged)

players_chemistry_df_values = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv")


''' SQL
df_matches_all_esp = load_db_to_pd(sql_query= "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                                "WHERE matchId IN "
                                "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                                "where Scouting.dbo.Wyscout_Matches.seasonId  =187526 )",
                                 db_name='Development')

df_matches_all_eng = load_db_to_pd(sql_query=  "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                                "WHERE matchId IN "
                                "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                                "where Scouting.dbo.Wyscout_Matches.seasonId  =187475 )",
                                 db_name='Development')

df_matches_all_IT = load_db_to_pd(sql_query=  "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                                "WHERE matchId IN "
                                "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                                "where Scouting.dbo.Wyscout_Matches.seasonId =187528)",
                                 db_name='Development')
'''


#---ENG----------------------------------------------------------------------------------------------------------------------------------------------
#Extract keeper id's

#Remove keepers from all used dataframes
'''df_sqaud = df_sqaud.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

#Remove players not in the starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

#Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
df_pos['sumVaep'] = df_pos['sumVaep'].fillna(0)

league_df = sd_table[sd_table['competitionId'] == 364]

df_process = league_df.copy()
s_21_22 = max(df_process.seasonId)
df_process_20_21 = df_process[df_process['seasonId'] == min(df_process.seasonId)]
df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
df_sqaud_filtered = df_sqaud[df_sqaud['seasonId'] == s_21_22]
pos_filtered = df_pos[df_pos['seasonId'] == s_21_22]
pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

# Extract net offensive impact per game per team
df_net_oi = getOi(df_process.copy())
# Extract distance measures
df_ec = getDistance(df_process_21_22.copy())
# Extract players shares
df_player_share = getResponsibilityShare((df_process_21_22.copy()))
# Extract jdi
df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all_eng)

df_joi = get_joi(pos_sorted)
df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

players = df_joi[(df_joi['p1'] == 9616) & (df_joi['p2'] == 425911)]

stamps = get_timestamps(max(df_process.seasonId))
ready_for_scaling, tvp = prepare_for_scaling(df_process_21_22, df_sqaud_filtered, stamps)
ready_for_scaling['seasonId'] = s_21_22
df_joi90_and_jdi90['seasonId'] = s_21_22
df_players_teams_u = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90['teamId'].unique())]
df_chemistry = (get_chemistry(ready_for_scaling, df_joi90_and_jdi90, df_players_teams))[['p1', 'p2','teamId', 'seasonId','minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90','winners90', 'chemistry']]

df_overview = get_overview_frame(df_chemistry, df_players_teams)

'''

#---ESP--------------------------------------------------------------------------------------------------------------------------------------------------------
#Extract keeper id's
'''keepers = (df_keepers.playerId.values).tolist()

#Remove keepers from all used dataframes
df_sqaud_esp = df_sqaud.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

#Remove players not in the starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

#Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
df_pos['sumVaep'] = df_pos['sumVaep'].fillna(0)

league_df_esp = sd_table[sd_table['competitionId'] == 795]

df_process_esp = league_df_esp.copy()
s_21_22_esp = max(df_process_esp.seasonId)
df_process_20_21_esp = df_process_esp[df_process_esp['seasonId'] == min(df_process_esp.seasonId)]
df_process_21_22_esp = df_process_esp[df_process_esp['seasonId'] == s_21_22_esp]
df_sqaud_filtered_esp = df_sqaud[df_sqaud['seasonId'] == s_21_22_esp]
pos_filtered_esp = df_pos[df_pos['seasonId'] == s_21_22_esp]
pos_sorted_esp = pos_filtered_esp.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
df_pairwise_playing_time_esp = pairwise_playing_time(df_sqaud_filtered_esp)

# Extract net offensive impact per game per team
df_net_oi_esp = getOi(df_process_esp.copy())
# Extract distance measures
df_ec_esp = getDistance(df_process_21_22_esp.copy())
# Extract players shares
df_player_share_esp = getResponsibilityShare((df_process_21_22_esp.copy()))
# Extract jdi
df_jdi_esp = get_jdi(df_player_share_esp, df_ec_esp, df_net_oi_esp, df_matches_all_esp)

df_joi_esp = get_joi_tester(pos_sorted_esp)
df_joi90_and_jdi90_esp = compute_normalized_values(df_joi_esp, df_jdi_esp, df_pairwise_playing_time_esp)

stamps = get_timestamps(max(df_process_esp.seasonId))
ready_for_scaling_esp = prepare_for_scaling(df_process_21_22_esp, df_sqaud_filtered_esp, stamps)
ready_for_scaling_esp['seasonId'] = s_21_22_esp
df_joi90_and_jdi90_esp['seasonId'] = s_21_22_esp
df_players_teams_esp = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90_esp['teamId'].unique())]
df_chemistry_esp = (get_chemistry(ready_for_scaling_esp, df_joi90_and_jdi90_esp, df_players_teams_esp))[['p1', 'p2','teamId', 'seasonId','minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90','winners90', 'chemistry']]

print(df_chemistry_esp['teamId'].unique())

df_overview_esp = get_overview_frame(df_chemistry_esp, df_players_teams_esp)'''
#-------------------------------------------------------------------------------------------------------------------------------------

#Archieved- CHEM ABILITY
chem_ability_v2_eng = generate_chemistry_ability_v2(df_overview)
chem_ability_v2_esp = generate_chemistry_ability_v2(df_overview_esp)


#---IT----------------------------------------------------------------------------------------------------------------------------------------------
#Remove keepers from all used dataframes
'''keepers = (df_keepers.playerId.values).tolist()

df_sqaud_IT = df_sqaud.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

#Remove players not in the starting 11
df_sqaud = df_sqaud[(df_sqaud.bench == False)]

#Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
df_pos['sumVaep'] = df_pos['sumVaep'].fillna(0)

league_df_IT = sd_table[sd_table['competitionId'] == 524]

df_process_IT = league_df_IT.copy()
s_21_22_IT = max(df_process_IT.seasonId)
df_process_20_21_IT = df_process_IT[df_process_IT['seasonId'] == min(df_process_IT.seasonId)]
df_process_21_22_IT = df_process_IT[df_process_IT['seasonId'] == s_21_22_IT]
df_sqaud_filtered_IT = df_sqaud[df_sqaud['seasonId'] == s_21_22_IT]
pos_filtered_IT = df_pos[df_pos['seasonId'] == s_21_22_IT]
pos_sorted_IT = pos_filtered_IT.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
df_pairwise_playing_time_IT = pairwise_playing_time(df_sqaud_filtered_IT)

# Extract net offensive impact per game per team
df_net_oi_IT = getOi(df_process_IT.copy())
# Extract distance measures
df_ec_IT = getDistance(df_process_21_22_IT.copy())
# Extract players shares
df_player_share_IT = getResponsibilityShare((df_process_21_22_IT.copy()))
# Extract jdi
df_jdi_IT = get_jdi(df_player_share_IT, df_ec_IT, df_net_oi_IT, df_matches_all_IT)

df_joi_IT = get_joi_tester(pos_sorted_IT)
df_joi90_and_jdi90_IT = compute_normalized_values(df_joi_IT, df_jdi_IT, df_pairwise_playing_time_IT)

stamps = get_timestamps(max(df_process_IT.seasonId))
ready_for_scaling_IT = prepare_for_scaling(df_process_21_22_IT, df_sqaud_filtered_IT, stamps)
ready_for_scaling_IT['seasonId'] = s_21_22_IT
df_joi90_and_jdi90_IT['seasonId'] = s_21_22_IT
df_players_teams_IT = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90_IT['teamId'].unique())]
df_chemistry_IT = (get_chemistry_tester(ready_for_scaling_IT, df_joi90_and_jdi90_IT, df_players_teams_IT))[['p1', 'p2','teamId', 'seasonId','minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90','winners90', 'chemistry']]
'''
#----------------------------------------------------------------------------------------------------

def generate_chemistry(df_keepers, squad, league_df, pos, df_matches_all, def_suc):
    #Remove keepers from all used dataframes
    keepers = (df_keepers.playerId.values).tolist()
    squad = squad.query("playerId not in @keepers")
    league_df = league_df.query("playerId not in @keepers")

    # Remove players not in the starting 11
    squad = squad[(squad.bench == False)]

    # Impute zero values for sumVaep
    league_df['sumVaep'] = league_df['sumVaep'].fillna(0)
    pos['sumVaep'] = pos['sumVaep'].fillna(0)
    df_process = league_df.copy()
    s_21_22 = max(df_process.seasonId)
    def_per_player_per_match = league_df[['eventId', 'playerId', 'seasonId']].merge(def_suc, on='eventId')
    def_per_player_per_match = def_per_player_per_match[def_per_player_per_match['seasonId'] == s_21_22]
    d_m_s = (def_per_player_per_match.groupby(['playerId', 'matchId', 'seasonId'], as_index=False)['def_action'].sum())[['playerId', 'matchId', 'def_action']]
    df_matches_all = df_matches_all[df_matches_all['seasonId'] ==s_21_22 ]
    df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
    df_sqaud_filtered = squad[squad['seasonId'] == s_21_22]
    df_match_duration = df_sqaud_filtered.groupby('matchId', as_index=False)['minutes'].max()

    pos_filtered = pos[pos['seasonId'] == s_21_22]
    pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)

    df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

    # Extract net offensive impact per game per team
    df_net_oi = getOi(df_process.copy())

    # Extract distance measures
    df_ec, avg_pos = getDistance(df_process_21_22.copy())

    # Extract players shares
    df_player_share, zones_actions = getResponsibilityShare((df_process_21_22.copy()))

    # Extract jdi
    df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all, d_m_s)
    df_joi = get_joi(pos_sorted)
    df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)
    stamps = get_timestamps(max(df_process.seasonId))
    ready_for_scaling = prepare_for_scaling(df_process_21_22, df_sqaud_filtered, stamps, df_match_duration)
    ready_for_scaling['seasonId'] = s_21_22
    df_joi90_and_jdi90['seasonId'] = s_21_22
    df_players_teams_c = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90['teamId'].unique())]
    df_chemistry = (get_chemistry(ready_for_scaling, df_joi90_and_jdi90, df_players_teams_c))[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor', 'joi', 'jdi', 'df_jdi90', 'df_joi90', 'winners90', 'chemistry']]
    return df_chemistry, df_process_21_22
def prepare_data_for_chemistry(df_keepers, squad, league_df, pos, df_matches_all, def_suc):
    #Remove keepers from all used dataframes
    keepers = (df_keepers.playerId.values).tolist()
    squad = squad.query("playerId not in @keepers")
    league_df = league_df.query("playerId not in @keepers")

    # Remove players not in the starting 11
    squad = squad[(squad.bench == False)]

    # Impute zero values for sumVaep
    league_df['sumVaep'] = league_df['sumVaep'].fillna(0)
    pos['sumVaep'] = pos['sumVaep'].fillna(0)
    df_process = league_df.copy()
    s_21_22 = max(df_process.seasonId)
    def_per_player_per_match = league_df[['eventId', 'playerId', 'seasonId']].merge(def_suc, on='eventId')
    def_per_player_per_match = def_per_player_per_match[def_per_player_per_match['seasonId'] == s_21_22]
    d_m_s = (def_per_player_per_match.groupby(['playerId', 'matchId', 'seasonId'], as_index=False)['def_action'].sum())[['playerId', 'matchId', 'def_action']]
    df_matches_all = df_matches_all[df_matches_all['seasonId'] ==s_21_22 ]
    df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
    df_sqaud_filtered = squad[squad['seasonId'] == s_21_22]
    df_match_duration = df_sqaud_filtered.groupby('matchId', as_index=False)['minutes'].max()
    pos_filtered = pos[pos['seasonId'] == s_21_22]
    pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)

    df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

    # Extract net offensive impact per game per team
    df_net_oi = getOi(df_process.copy())

    # Extract distance measures
    df_ec, avg_pos = getDistance(df_process_21_22.copy())

    # Extract players shares
    df_player_share, zones_actions = getResponsibilityShare((df_process_21_22.copy()))

    # Extract jdi
    df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all, d_m_s)
    df_joi = get_joi(pos_sorted)
    df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)
    stamps = get_timestamps(max(df_process.seasonId))
    ready_for_scaling = prepare_for_scaling(df_process_21_22, df_sqaud_filtered, stamps, df_match_duration)
    ready_for_scaling['seasonId'] = s_21_22
    df_joi90_and_jdi90['seasonId'] = s_21_22
    df_players_teams_c = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90['teamId'].unique())]

    return ready_for_scaling, df_joi90_and_jdi90, df_players_teams_c


#---Expected goals
def compute_expected_outputs(df):
    xg_game = df.groupby(['teamId', 'matchId'], as_index = False)['xG'].sum()
    xg_game_v2 = xg_game.merge(xg_game,  on='matchId')
    xg_game_v2 = xg_game_v2[xg_game_v2['teamId_x'] != xg_game_v2['teamId_y'] ]
    xg_game_v3 = xg_game_v2.drop_duplicates(subset='matchId')
    xg_game_v3['teamId'] = np.where(xg_game_v3.xG_x > xg_game_v3.xG_y, xg_game_v3.teamId_x,np.where(xg_game_v3.xG_y > xg_game_v3.xG_x, xg_game_v3.teamId_y, -1))
    team_count_wins = xg_game_v3.groupby(['teamId']).size().reset_index(name='Count')
    xg_game_v4 = xg_game_v2.drop_duplicates(subset='matchId')
    xg_game_v4['teamId'] = np.where(xg_game_v4.xG_x < xg_game_v4.xG_y, xg_game_v4.teamId_x,np.where(xg_game_v4.xG_y < xg_game_v4.xG_x, xg_game_v4.teamId_y, -1))
    team_count_losses  = xg_game_v4.groupby(['teamId']).size().reset_index(name='Count')
    team_count_wins = team_count_wins.rename(columns = {'Count': 'expected_wins'})
    team_count_losses = team_count_losses.rename(columns = {'Count': 'expected_losses'})
    team_count_total = team_count_wins.merge(team_count_losses, on='teamId')

    return team_count_total

def compute_äctual_outputs(df, matches_all):
    matches = matches_all[matches_all['seasonId'] == (df.iloc[0].seasonId)]
    team_count_wins = matches.groupby(['winner']).size().reset_index(name='Count')
    team_count_wins = team_count_wins.rename(columns={'winner': 'teamId', 'Count': 'actual_wins'})
    return team_count_wins



def get_summed_xg(df):
    xg_game = df.groupby(['teamId'], as_index = False)['xG'].sum()
    xg_game = df.groupby(['teamId'], as_index = False)['xG'].sum()
    return xg_game


def check_dis(df, feature):
    plt.hist(df[feature], bins=10)
    plt.title("Histogram of " + feature +  " variable")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


def get_team_chemistry(df):
    df_team_chem = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].sum()
    #df_team_joi = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].mean()
    #df_team_jdi = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].mean()
    return df_team_chem

def plot_relationship(df, label):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # Create the first scatter plot
    ax1.scatter(df['chemistry'], df['expected_wins'])
    ax1.set_xlabel('Chemistry')
    ax1.set_ylabel('Expected Wins')

    # Create the second scatter plot
    ax2.scatter(df['chemistry'], df['expected_losses'])
    ax2.set_xlabel('Chemistry')
    ax2.set_ylabel('Expected Losses')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='lightgray', alpha=0.25, zorder=1)
    # Show the figure
    plt.show()

    # Show the figure

    # Extract 'chemistry' and 'expected_wins' columns from both DataFrames
    chemistry1 = df['chemistry']
    expected_wins1 = df['expected_wins']

    # Create a scatter plot with the first DataFrame
    plt.scatter(chemistry1, expected_wins1, label='DataFrame 1')

    plt.xlabel('Chemistry')
    plt.ylabel('Expected Wins')
    plt.title(label)
    plt.show()

'''
sd_table['posAction'] = sd_table.apply(lambda row: possession_action(row), axis=1)
sd_table['nonPosAction'] = sd_table.apply(lambda row: non_possession_action(row), axis=1)
merged = sd_table.merge(df_successes, left_on='eventId', right_on='id')
merged = merged[(merged['accurate'].notnull()) or (merged['not_accurate'].notnull()) ]

m_dk = merg ed[merged['competitionId'] == 335]
max_dk = max(m_dk['seasonId'])
m_dk = m_dk[m_dk['seasonId'] == max_dk]
m_dk_gr = m_dk.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_dk_gr = m_dk_gr.rename(columns={'playerId_x': 'playerId'})

m_eng = merged[merged['competitionId'] == 364]
max_eng = max(m_eng['seasonId'])
m_eng = m_eng[m_eng['seasonId'] == max_eng]
m_eng_gr = m_eng.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_eng_gr = m_eng_gr.rename(columns={'playerId_x': 'playerId'})


m_fr = merged[merged['competitionId'] == 412]
max_fr = max(m_fr['seasonId'])
m_fr = m_fr[m_fr['seasonId'] == max_fr]
m_fr_gr = m_fr.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_fr_gr = m_fr_gr.rename(columns={'playerId_x': 'playerId'})


m_it = merged[merged['competitionId'] == 524]
max_it = max(m_it['seasonId'])
m_it = m_it[m_it['seasonId'] == max_it]
m_it_gr = m_it.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_it_gr = m_it_gr.rename(columns={'playerId_x': 'playerId'})

m_ger = merged[merged['competitionId'] == 426]
max_ger = max(m_ger['seasonId'])
m_ger = m_ger[m_ger['seasonId'] == max_ger]
m_ger_gr = m_ger.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_ger_gr = m_ger_gr.rename(columns={'playerId_x': 'playerId'})


m_por = merged[merged['competitionId'] == 707]
max_por = max(m_por['seasonId'])
m_por = m_por[m_por['seasonId'] == max_por]
m_por_gr = m_por.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_por_gr = m_por_gr.rename(columns={'playerId_x': 'playerId'})


m_ned = merged[merged['competitionId'] == 635]
max_ned = max(m_ned['seasonId'])
m_ned = m_ned[m_ned['seasonId'] == max_ned]
m_ned_gr = m_ned.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_ned_gr = m_ned_gr.rename(columns={'playerId_x': 'playerId'})


m_esp = merged[merged['competitionId'] == 795]
max_esp = max(m_esp['seasonId'])
m_esp = m_esp[m_esp['seasonId'] == max_esp]
m_esp_gr = m_esp.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_esp_gr = m_esp_gr.rename(columns={'playerId_x': 'playerId'})


m_tur = merged[merged['competitionId'] == 852]
max_tur = max(m_tur['seasonId'])
m_tur = m_tur[m_tur['seasonId'] == max_tur]
m_tur_gr = m_tur.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_tur_gr = m_tur_gr.rename(columns={'playerId_x': 'playerId'})


m_bel = merged[merged['competitionId'] == 198]
max_bel = max(m_bel['seasonId'])
m_bel = m_bel[m_bel['seasonId'] == max_bel]
m_bel_gr = m_bel.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_bel_gr = m_bel_gr.rename(columns={'playerId_x': 'playerId'})

m_au = merged[merged['competitionId'] == 168]
max_au = max(m_au['seasonId'])
m_au = m_au[m_au['seasonId'] == max_au]
m_au_gr = m_au.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_au_gr = m_au_gr.rename(columns={'playerId_x': 'playerId'})

m_swi = merged[merged['competitionId'] == 830]
max_swi = max(m_swi['seasonId'])
m_swi = m_swi[m_swi['seasonId'] == max_swi]
m_swi_gr = m_swi.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_swi_gr = m_swi_gr.rename(columns={'playerId_x': 'playerId'})

m_cyo = merged[merged['competitionId'] == 310]
max_cyo = max(m_cyo['seasonId'])
m_cyo = m_cyo[m_cyo['seasonId'] == max_cyo]
m_cyo_gr = m_cyo.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_cyo_gr = m_cyo_gr.rename(columns={'playerId_x': 'playerId'})

m_ser = merged[merged['competitionId'] == 905]
max_ser = max(m_ser['seasonId'])
m_ser = m_ser[m_ser['seasonId'] == max_ser]
m_ser_gr = m_ser.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_ser_gr = m_ser_gr.rename(columns={'playerId_x': 'playerId'})

m_gre = merged[merged['competitionId'] == 448]
max_gre = max(m_gre['seasonId'])
m_gre = m_gre[m_gre['seasonId'] == max_gre]
m_gre_gr = m_gre.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_gre_gr = m_gre_gr.rename(columns={'playerId_x': 'playerId'})


m_cro = merged[merged['competitionId'] == 302]
max_cro = max(m_cro['seasonId'])
m_gre = m_cro[m_cro['seasonId'] == max_cro]
m_cro_gr = m_cro.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_cro_gr = m_cro_gr.rename(columns={'playerId_x': 'playerId'})

m_pol = merged[merged['competitionId'] == 692]
max_pol = max(m_pol['seasonId'])
m_pol = m_pol[m_pol['seasonId'] == max_pol]
m_pol_gr = m_pol.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_pol_gr = m_pol_gr.rename(columns={'playerId_x': 'playerId'})

m_cze = merged[merged['competitionId'] == 323   ]
max_cze = max(m_cze['seasonId'])
m_cze = m_cze[m_cze['seasonId'] == max_cze]
m_cze_gr = m_cze.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_cze_gr = m_cze_gr.rename(columns={'playerId_x': 'playerId'})

m_rus = merged[merged['competitionId'] == 729]
max_rus = max(m_rus['seasonId'])
m_rus = m_rus[m_rus['seasonId'] == max_rus]
m_rus_gr = m_rus.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_rus_gr = m_rus_gr.rename(columns={'playerId_x': 'playerId'})

m_sco = merged[merged['competitionId'] == 750]
max_sco = max(m_sco['seasonId'])
m_sco = m_sco[m_sco['seasonId'] == max_sco]
m_sco_gr = m_sco.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_sco_gr = m_sco_gr.rename(columns={'playerId_x': 'playerId'})

m_hun = merged[merged['competitionId'] == 465]
max_hun = max(m_hun['seasonId'])
m_hun = m_hun[m_hun['seasonId'] == max_hun]
m_hun_gr = m_hun.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_hun_gr = m_hun_gr.rename(columns={'playerId_x': 'playerId'})

m_sk = merged[merged['competitionId'] == 775]
max_sk = max(m_sk['seasonId'])
m_sk = m_sk[m_sk['seasonId'] == max_sk]
m_sk_gr = m_sk.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_sk_gr = m_sk_gr.rename(columns={'playerId_x': 'playerId'})

m_swe = merged[merged['competitionId'] == 808]
max_swe = max(m_swe['seasonId'])
m_swe = m_swe[m_swe['seasonId'] == max_swe]
m_swe_gr = m_swe.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_swe_gr = m_swe_gr.rename(columns={'playerId_x': 'playerId'})

m_fin = merged[merged['competitionId'] == 400]
max_fin = max(m_fin['seasonId'])
m_fin = m_fin[m_fin['seasonId'] == max_fin]
m_fin_gr = m_fin.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_fin_gr = m_fin_gr.rename(columns={'playerId_x': 'playerId'})

m_ice = merged[merged['competitionId'] == 480]
max_ice = max(m_ice['seasonId'])
m_ice = m_ice[m_ice['seasonId'] == max_ice]
m_ice_gr = m_ice.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_ice_gr = m_ice_gr.rename(columns={'playerId_x': 'playerId'})

m_nor = merged[merged['competitionId'] == 669]
max_nor = max(m_nor['seasonId'])
m_nor = m_nor[m_nor['seasonId'] == max_nor]
m_nor_gr = m_nor.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_nor_gr = m_nor_gr.rename(columns={'playerId_x': 'playerId'})

m_sl = merged[merged['competitionId'] == 776]
max_sl = max(m_sl['seasonId'])
m_sl = m_sl[m_sl['seasonId'] == max_sl]
m_sl_gr = m_sl.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_sl_gr = m_sl_gr.rename(columns={'playerId_x': 'playerId'})
'''
league_df_dk = sd_table[sd_table['competitionId'] == 335]
league_df_eng = sd_table[sd_table['competitionId'] == 364]
league_df_fr = sd_table[sd_table['competitionId'] == 412]
league_df_it = sd_table[sd_table['competitionId'] == 524]
league_df_ger = sd_table[sd_table['competitionId'] == 426]
league_df_por = sd_table[sd_table['competitionId'] == 707]
league_df_ned = sd_table[sd_table['competitionId'] == 635]
league_df_esp = sd_table[sd_table['competitionId'] == 795]
league_df_tur = sd_table[sd_table['competitionId'] == 852]
league_df_bel = sd_table[sd_table['competitionId'] == 198]
league_df_au = sd_table[sd_table['competitionId'] == 168]
league_df_swi = sd_table[sd_table['competitionId'] == 830]
league_df_cyo = sd_table[sd_table['competitionId'] == 310]
league_df_ser = sd_table[sd_table['competitionId'] == 905]
league_df_gre = sd_table[sd_table['competitionId'] == 448]
league_df_cro = sd_table[sd_table['competitionId'] == 302]
league_df_pol = sd_table[sd_table['competitionId'] == 692]
league_df_cze = sd_table[sd_table['competitionId'] == 323]
league_df_rus = sd_table[sd_table['competitionId'] == 729]
league_df_sco = sd_table[sd_table['competitionId'] == 750]
league_df_hun = sd_table[sd_table['competitionId'] == 465]
league_df_sk = sd_table[sd_table['competitionId'] == 775]
league_df_swe = sd_table[sd_table['competitionId'] == 808]
league_df_fin = sd_table[sd_table['competitionId'] == 400]
league_df_ice = sd_table[sd_table['competitionId'] == 480]
league_df_nor = sd_table[sd_table['competitionId'] == 669]
league_df_sl = sd_table[sd_table['competitionId'] == 776]

m_bel = merged[merged['competitionId'] == 198]
max_bel = max(m_bel['seasonId'])
m_bel = m_bel[m_bel['seasonId'] == max_bel]
m_bel_gr = m_bel.groupby(['playerId_x'], as_index=False).agg({'posAction':'sum', 'nonPosAction':'sum', 'accurate': 'sum'})
m_bel_gr = m_bel_gr.rename(columns={'playerId_x': 'playerId'})


'''
league_df_dk = sd_table[sd_table['competitionId'] == 335]
league_df_dk['posAction'] = league_df_dk.apply(lambda row: possession_action(row), axis=1)
league_df_dk['nonPosAction'] = league_df_dk.apply(lambda row: non_possession_action(row), axis=1)

league_df_eng = sd_table[sd_table['competitionId'] == 364]
league_df_eng['posAction'] = league_df_eng.apply(lambda row: possession_action(row), axis=1)
league_df_eng['nonPosAction'] = league_df_eng.apply(lambda row: non_possession_action(row), axis=1)

league_df_fr = sd_table[sd_table['competitionId'] == 412]
league_df_fr['posAction'] = league_df_fr.apply(lambda row: possession_action(row), axis=1)
league_df_fr['nonPosAction'] = league_df_fr.apply(lambda row: non_possession_action(row), axis=1)

league_df_it = sd_table[sd_table['competitionId'] == 524]
league_df_it['posAction'] = league_df_it.apply(lambda row: possession_action(row), axis=1)
league_df_it['nonPosAction'] = league_df_it.apply(lambda row: non_possession_action(row), axis=1)

league_df_ger = sd_table[sd_table['competitionId'] == 426]
league_df_ger['posAction'] = league_df_ger.apply(lambda row: possession_action(row), axis=1)
league_df_ger['nonPosAction'] = league_df_ger.apply(lambda row: non_possession_action(row), axis=1)

league_df_por = sd_table[sd_table['competitionId'] == 707]
league_df_por['posAction'] = league_df_por.apply(lambda row: possession_action(row), axis=1)
league_df_por['nonPosAction'] = league_df_por.apply(lambda row: non_possession_action(row), axis=1)

league_df_ned = sd_table[sd_table['competitionId'] == 635]
league_df_ned['posAction'] = league_df_ned.apply(lambda row: possession_action(row), axis=1)
league_df_ned['nonPosAction'] = league_df_ned.apply(lambda row: non_possession_action(row), axis=1)

league_df_esp = sd_table[sd_table['competitionId'] == 795]
league_df_esp['posAction'] = league_df_esp.apply(lambda row: possession_action(row), axis=1)
league_df_esp['nonPosAction'] = league_df_esp.apply(lambda row: non_possession_action(row), axis=1)

league_df_tur = sd_table[sd_table['competitionId'] == 852]
league_df_tur['posAction'] = league_df_tur.apply(lambda row: possession_action(row), axis=1)
league_df_tur['nonPosAction'] = league_df_tur.apply(lambda row: non_possession_action(row), axis=1)

league_df_bel = sd_table[sd_table['competitionId'] == 198]
league_df_bel['posAction'] = league_df_bel.apply(lambda row: possession_action(row), axis=1)
league_df_bel['nonPosAction'] = league_df_bel.apply(lambda row: non_possession_action(row), axis=1)
'''

merged = df_successes.merge(sd_table[['eventId', 'seasonId', 'competitionId']], left_on='id', right_on='eventId')



def merge_performance_stats(pos, min_match_pl, zones_actions  ):
    pos_tot = pos.groupby(['playerId'], as_index = False).agg({'avg_x': 'mean', 'avg_y': 'mean'})
    min_match_pl = min_match_pl[min_match_pl['minutes'] > 30]
    min_match_pl_tot = min_match_pl.groupby(['playerId'], as_index = False).agg({'minutes': 'sum', 'matchId': 'count'})
    zones_actions_tot = zones_actions.groupby(['playerId'], as_index = False).agg({'zone_1_pl': 'sum', 'zone_2_pl': 'sum', 'zone_3_pl': 'sum', 'zone_4_pl': 'sum', 'zone_5_pl': 'sum', 'zone_6_pl': 'sum'})
    pos_tot = pos_tot.merge(min_match_pl_tot, on='playerId')
    merged = pos_tot.merge(zones_actions_tot, on='playerId')
    return merged

league_df_dk = sd_table[sd_table['competitionId'] == 335]
league_df_eng = sd_table[sd_table['competitionId'] == 364]
league_df_fr = sd_table[sd_table['competitionId'] == 412]
league_df_it = sd_table[sd_table['competitionId'] == 524]
league_df_ger = sd_table[sd_table['competitionId'] == 426]

league_df_dk = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/league_df_dk.csv")
league_df_eng = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/league_df_eng.csv")
league_df_fr= pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/league_df_fr.csv")
league_df_it = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/league_df_it.csv")
league_df_ger = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/league_df_ger.csv")


df_dk, df_process_21_22_dk = generate_chemistry(df_keepers, df_squad, league_df_dk, df_pos, df_matches_all, def_suc_tot)
df_dk_bif = df_dk[df_dk['teamId'] == 7453]
df_overview_bif = get_overview_frame(df_dk_bif, df_players_teams)
df_overview_bif = df_overview_bif.loc[:, [ 'seasonId', 'shortName_x', 'shortName_y', 'df_jdi90', 'df_joi90', 'chemistry']]
df_overview_bif = df_overview_bif.loc[:, [ 'p1', 'p2', 'minutes', 'seasonId', 'shortName_x', 'shortName_y', 'df_jdi90', 'df_joi90', 'chemistry']]

print("asd")
bif_events = sd_table[(sd_table['seasonId'] == 187483) & (sd_table['teamId'] == 7453) ]
bif_players = bif_events.playerId.unique()

def get_avg_chenm_p(df):
    p1 =( df[['p1', 'chemistry']]).rename(columns={'p1': 'playerId'})
    p2 = (df[['p2', 'chemistry']]).rename(columns={'p1': 'playerId'})
    players = pd.concat([p1,p2])
    players = players.groupby('playerId', as_index = False)['chemistry'].mean()
    return players
df_overview_bif_avg = get_avg_chenm_p(df_overview_bif)

p1_unique_p1_bif = df_dk_bif['p1'].unique()
p1_unique_p2_bif = df_dk_bif['p2'].unique()
all_bif = np.unique(np.concatenate((p1_unique_p1_bif, p1_unique_p2_bif)))

df_dk, df_process_21_22_dk = generate_chemistry(df_keepers, df_squad, league_df_dk, df_pos, df_matches_all, def_suc_tot)
print("hej")

df_eng, df_process_21_22_eng = generate_chemistry(df_keepers, df_squad, league_df_eng, df_pos,df_matches_all, def_suc_tot)
df_fr, df_process_21_22_fr = generate_chemistry(df_keepers, df_squad, league_df_fr, df_pos,df_matches_all, def_suc_tot)
df_it, df_process_21_22_it = generate_chemistry(df_keepers, df_squad, league_df_it, df_pos,df_matches_all, def_suc_tot)
df_ger, df_process_21_22_ger = generate_chemistry(df_keepers, df_squad, league_df_ger, df_pos,df_matches_all, def_suc_tot)
df_por, df_process_21_22_por = generate_chemistry(df_keepers, df_squad, league_df_por, df_pos,df_matches_all, def_suc_tot)
df_ned, df_process_21_22_ned = generate_chemistry(df_keepers, df_squad, league_df_ned, df_pos,df_matches_all, def_suc_tot)
df_esp, df_process_21_22_esp = generate_chemistry(df_keepers, df_squad, league_df_esp, df_pos,df_matches_all, def_suc_tot)
df_tur, df_process_21_22_tur = generate_chemistry(df_keepers, df_squad, league_df_tur, df_pos,df_matches_all, def_suc_tot)
df_bel, df_process_21_22_bel = generate_chemistry(df_keepers, df_squad, league_df_bel, df_pos,df_matches_all, def_suc_tot)
df_swi, df_process_21_22_swi = generate_chemistry(df_keepers, df_squad, league_df_swi, df_pos,df_matches_all, def_suc_tot)
df_gre, df_process_21_22_gre = generate_chemistry(df_keepers, df_squad, league_df_gre, df_pos,df_matches_all, def_suc_tot)
df_cro, df_process_21_22_cro = generate_chemistry(df_keepers, df_squad, league_df_cro, df_pos,df_matches_all, def_suc_tot)
df_cze, df_process_21_22_cze = generate_chemistry(df_keepers, df_squad, league_df_cze, df_pos,df_matches_all, def_suc_tot)
df_rus, df_process_21_22_rus = generate_chemistry(df_keepers, df_squad, league_df_rus, df_pos,df_matches_all, def_suc_tot)
df_sco, df_process_21_22_sco = generate_chemistry(df_keepers, df_squad, league_df_sco, df_pos,df_matches_all, def_suc_tot)
df_hun, df_process_21_22_hun = generate_chemistry(df_keepers, df_squad, league_df_hun, df_pos,df_matches_all, def_suc_tot)
df_ice, df_process_21_22_ice = generate_chemistry(df_keepers, df_squad, league_df_ice, df_pos,df_matches_all, def_suc_tot)
df_nor, df_process_21_22_nor = generate_chemistry(df_keepers, df_squad, league_df_nor, df_pos,df_matches_all, def_suc_tot)
df_sl, df_process_21_22_sl = generate_chemistry(df_keepers, df_squad, league_df_sl, df_pos,df_matches_all, def_suc_tot)
df_sk, df_process_21_22_sk = generate_chemistry(df_keepers, df_squad, league_df_sk, df_pos,df_matches_all, def_suc_tot)
df_swe, df_process_21_22_swe = generate_chemistry(df_keepers, df_squad, league_df_swe, df_pos,df_matches_all, def_suc_tot)
df_pol, df_process_21_22_pol = generate_chemistry(df_keepers, df_squad, league_df_pol, df_pos,df_matches_all, def_suc_tot)
df_au, df_process_21_22_au = generate_chemistry(df_keepers, df_squad, league_df_au, df_pos,df_matches_all, def_suc_tot)
df_cyo, df_process_21_22_cyo = generate_chemistry(df_keepers, df_squad, league_df_cyo, df_pos,df_matches_all, def_suc_tot)
df_ser, df_process_21_22_ser = generate_chemistry(df_keepers, df_squad, league_df_ser, df_pos,df_matches_all, def_suc_tot)

print("hej")
rfs_dk, joi90_jdi90_dk, ptc_dk = prepare_data_for_chemistry(df_keepers, df_squad, league_df_dk, df_pos, df_matches_all, def_suc_tot)
rfs_eng, joi90_jdi90_eng, ptc_eng = prepare_data_for_chemistry(df_keepers, df_squad, league_df_eng, df_pos, df_matches_all, def_suc_tot)
rfs_fr, joi90_jdi90_fr, ptc_fr = prepare_data_for_chemistry(df_keepers, df_squad, league_df_fr, df_pos, df_matches_all, def_suc_tot)
rfs_it, joi90_jdi90_it, ptc_it = prepare_data_for_chemistry(df_keepers, df_squad, league_df_it, df_pos, df_matches_all, def_suc_tot)
rfs_ger, joi90_jdi90_ger, ptc_ger = prepare_data_for_chemistry(df_keepers, df_squad, league_df_ger, df_pos, df_matches_all, def_suc_tot)
rfs_por, joi90_jdi90_por, ptc_por = prepare_data_for_chemistry(df_keepers, df_squad, league_df_por, df_pos, df_matches_all, def_suc_tot)
rfs_ned, joi90_jdi90_ned, ptc_ned = prepare_data_for_chemistry(df_keepers, df_squad, league_df_ned, df_pos, df_matches_all, def_suc_tot)
rfs_esp, joi90_jdi90_esp, ptc_esp = prepare_data_for_chemistry(df_keepers, df_squad, league_df_esp, df_pos, df_matches_all, def_suc_tot)
rfs_tur, joi90_jdi90_tur, ptc_tur = prepare_data_for_chemistry(df_keepers, df_squad, league_df_tur, df_pos, df_matches_all, def_suc_tot)
rfs_bel, joi90_jdi90_bel, ptc_bel = prepare_data_for_chemistry(df_keepers, df_squad, league_df_bel, df_pos, df_matches_all, def_suc_tot)
rfs_swi, joi90_jdi90_swi, ptc_swi = prepare_data_for_chemistry(df_keepers, df_squad, league_df_swi, df_pos, df_matches_all, def_suc_tot)
rfs_ser, joi90_jdi90_ser, ptc_ser = prepare_data_for_chemistry(df_keepers, df_squad, league_df_ser, df_pos, df_matches_all, def_suc_tot)
rfs_gre, joi90_jdi90_gre, ptc_gre = prepare_data_for_chemistry(df_keepers, df_squad, league_df_gre, df_pos, df_matches_all, def_suc_tot)
rfs_cro, joi90_jdi90_cro, ptc_cro = prepare_data_for_chemistry(df_keepers, df_squad, league_df_cro, df_pos, df_matches_all, def_suc_tot)
rfs_pol, joi90_jdi90_pol, ptc_pol = prepare_data_for_chemistry(df_keepers, df_squad, league_df_pol, df_pos, df_matches_all, def_suc_tot)
rfs_cze, joi90_jdi90_cze, ptc_cze = prepare_data_for_chemistry(df_keepers, df_squad, league_df_cze, df_pos, df_matches_all, def_suc_tot)
rfs_rus, joi90_jdi90_rus, ptc_rus = prepare_data_for_chemistry(df_keepers, df_squad, league_df_rus, df_pos, df_matches_all, def_suc_tot)
rfs_sco, joi90_jdi90_sco, ptc_sco = prepare_data_for_chemistry(df_keepers, df_squad, league_df_sco, df_pos, df_matches_all, def_suc_tot)
rfs_hun, joi90_jdi90_hun, ptc_hun = prepare_data_for_chemistry(df_keepers, df_squad, league_df_hun, df_pos, df_matches_all, def_suc_tot)
rfs_ice, joi90_jdi90_ice, ptc_ice = prepare_data_for_chemistry(df_keepers, df_squad, league_df_ice, df_pos, df_matches_all, def_suc_tot)
rfs_nor, joi90_jdi90_nor, ptc_nor = prepare_data_for_chemistry(df_keepers, df_squad, league_df_nor, df_pos, df_matches_all, def_suc_tot)


for_scaling = pd.concat([rfs_dk, rfs_eng, rfs_fr, rfs_it,
                         rfs_ger, rfs_por, rfs_ned, rfs_esp,
                         rfs_tur, rfs_bel, rfs_swi, rfs_hun,
                         rfs_ser, rfs_gre, rfs_cro, rfs_ice,
                         rfs_pol, rfs_cze, rfs_rus, rfs_sco,
                         rfs_nor
                         ])
joi_jdi = pd.concat([joi90_jdi90_dk, joi90_jdi90_eng, joi90_jdi90_fr, joi90_jdi90_it,
                         joi90_jdi90_ger, joi90_jdi90_por, joi90_jdi90_ned, joi90_jdi90_esp,
                         joi90_jdi90_tur, joi90_jdi90_bel, joi90_jdi90_swi,joi90_jdi90_hun,
                         joi90_jdi90_ser, joi90_jdi90_gre, joi90_jdi90_cro, joi90_jdi90_ice,
                         joi90_jdi90_pol, joi90_jdi90_cze, joi90_jdi90_rus, joi90_jdi90_sco,
                         joi90_jdi90_nor
                         ])
ptc = pd.concat([ptc_dk, ptc_eng, ptc_fr, ptc_it,
                         ptc_ger, ptc_por, ptc_ned, ptc_esp,
                         ptc_tur, ptc_bel, ptc_swi, ptc_hun,
                         ptc_ser, ptc_gre, ptc_cro,ptc_ice,
                         ptc_pol, ptc_cze, ptc_rus, ptc_sco,
                         ptc_nor
                         ])


df_chem = (get_chemistry(for_scaling, joi_jdi, ptc))[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90', 'winners90', 'chemistry']]
team_chemistry_full = get_team_chemistry(df_chem)

pos_all = pd.concat([pl_avg_pos_dk, pl_avg_pos_eng, pl_avg_pos_fr, pl_avg_pos_it, pl_avg_pos_ger, pl_avg_pos_por, pl_avg_pos_ned, pl_avg_pos_esp, pl_avg_pos_tur, pl_avg_pos_bel, pl_avg_pos_au, pl_avg_pos_swi, pl_avg_pos_cyo, pl_avg_pos_swe, pl_avg_pos_gre, pl_avg_pos_ser, pl_avg_pos_cro, pl_avg_pos_pol, pl_avg_pos_cze, pl_avg_pos_rus, pl_avg_pos_sco, pl_avg_pos_hun, pl_avg_pos_sk, pl_avg_pos_fin, pl_avg_pos_ice, pl_avg_pos_nor, pl_avg_pos_sl])
min_match_all = pd.concat([games_players_minutes_dk, games_players_minutes_eng, games_players_minutes_fr, games_players_minutes_it, games_players_minutes_ger, games_players_minutes_por, games_players_minutes_ned, games_players_minutes_esp, games_players_minutes_tur, games_players_minutes_bel, games_players_minutes_au, games_players_minutes_swi, games_players_minutes_cyo, games_players_minutes_swe, games_players_minutes_gre, games_players_minutes_ser, games_players_minutes_cro, games_players_minutes_pol, games_players_minutes_cze, games_players_minutes_rus, games_players_minutes_sco, games_players_minutes_hun, games_players_minutes_sk, games_players_minutes_fin, games_players_minutes_ice, games_players_minutes_nor, games_players_minutes_sl])
zones_all = pd.concat([zones_actions_dk, zones_actions_eng, zones_actions_fr,zones_actions_it, zones_actions_ger, zones_actions_por, zones_actions_ned, zones_actions_esp, zones_actions_tur, zones_actions_bel, zones_actions_au, zones_actions_swi, zones_actions_cyo, zones_actions_swe, zones_actions_gre, zones_actions_ser, zones_actions_cro, zones_actions_pol, zones_actions_cze, zones_actions_rus, zones_actions_sco, zones_actions_hun, zones_actions_sk, zones_actions_fin, zones_actions_ice, zones_actions_nor, zones_actions_sl])

performance_stats = merge_performance_stats(pos_all, min_match_all, zones_all)
performance_stats = performance_stats.rename(columns ={'matchId': 'match appearances'})

df_expected_dk = compute_expected_outputs(df_process_21_22_dk)
df_expected_eng = compute_expected_outputs(df_process_21_22_eng)
df_expected_fr = compute_expected_outputs(df_process_21_22_fr)
df_expected_it = compute_expected_outputs(df_process_21_22_it)
df_expected_ger = compute_expected_outputs(df_process_21_22_ger)
df_expected_por = compute_expected_outputs(df_process_21_22_por)
df_expected_ned = compute_expected_outputs(df_process_21_22_ned)
df_expected_bel = compute_expected_outputs(df_process_21_22_bel)
df_expected_esp = compute_expected_outputs(df_process_21_22_esp)
df_expected_tur = compute_expected_outputs(df_process_21_22_tur)
df_expected_au = compute_expected_outputs(df_process_21_22_au)
df_expected_swi = compute_expected_outputs(df_process_21_22_swi)
df_expected_cyo = compute_expected_outputs(df_process_21_22_cyo)
df_expected_ser = compute_expected_outputs(df_process_21_22_ser)
df_expected_gre = compute_expected_outputs(df_process_21_22_gre)
df_expected_cro = compute_expected_outputs(df_process_21_22_cro)
df_expected_pol = compute_expected_outputs(df_process_21_22_pol)
df_expected_cze = compute_expected_outputs(df_process_21_22_cze)
df_expected_rus = compute_expected_outputs(df_process_21_22_rus)
df_expected_sco = compute_expected_outputs(df_process_21_22_sco)
df_expected_hun = compute_expected_outputs(df_process_21_22_hun)
df_expected_sk = compute_expected_outputs(df_process_21_22_sk)
df_expected_swe = compute_expected_outputs(df_process_21_22_swe)
df_expected_ice = compute_expected_outputs(df_process_21_22_ice)
df_expected_nor = compute_expected_outputs(df_process_21_22_nor)
df_expected_sl = compute_expected_outputs(df_process_21_22_sl)




df_actual_dk = compute_äctual_outputs(df_process_21_22_dk, df_matches_all)
df_actual_au = compute_äctual_outputs(df_process_21_22_au, df_matches_all)
df_tot_au = df_actual_au.merge(df_expected_au, on='teamId')
df_tot_au['difference'] = df_tot_au['actual_wins'] - df_tot_au['expected_wins']


df_expected_full = pd.concat([df_expected_dk, df_expected_eng, df_expected_fr, df_expected_it,
                              df_expected_ger, df_expected_por, df_expected_ned, df_expected_bel,
                              df_expected_esp, df_expected_tur, df_expected_au, df_expected_swi,
                              df_expected_cyo, df_expected_ser, df_expected_gre, df_expected_cro,
                              df_expected_pol, df_expected_cze, df_expected_rus, df_expected_sco,
                              df_expected_hun, df_expected_sk, df_expected_swe, df_expected_fin,
                              df_expected_ice, df_expected_nor, df_expected_sl
                              ])

df_merged_full = df_ch
em.merge(df_expected_full, on=['teamId'])

df_overview = get_overview_frame(df_eng, df_players_teams)

plot_relationship(df_merged_full, 'all')
df_merged_full['chemistry'].corr(df_merged_full['expected_wins'])
df_merged_full['chemistry'].corr(df_merged_full['expected_losses'])

team_chem_all = get_team_chemistry(players_chemistry_df_values)

team_chem_dk = get_team_chemistry(df_dk)
team_chem_eng = get_team_chemistry(df_eng)
team_chem_fr = get_team_chemistry(df_fr)
team_chem_it = get_team_chemistry(df_it)
team_chem_ger = get_team_chemistry(df_ger)
team_chem_ned = get_team_chemistry(df_ned)
team_chem_por = get_team_chemistry(df_por)
team_chem_bel = get_team_chemistry(df_bel)
team_chem_esp = get_team_chemistry(df_esp)
team_chem_tur = get_team_chemistry(df_tur)

team_chem_au = get_team_chemistry(df_au)
team_chem_swi = get_team_chemistry(df_swi)
team_chem_cyo = get_team_chemistry(df_cyo)
team_chem_ser = get_team_chemistry(df_ser)
team_chem_gre = get_team_chemistry(df_gre)
team_chem_cro = get_team_chemistry(df_cro)
team_chem_pol = get_team_chemistry(df_pol)
team_chem_cze = get_team_chemistry(df_cze)
team_chem_rus = get_team_chemistry(df_rus)
team_chem_sco = get_team_chemistry(df_sco)
team_chem_hun = get_team_chemistry(df_hun)
team_chem_sk = get_team_chemistry(df_sk)
team_chem_swe = get_team_chemistry(df_swe)
team_chem_ice = get_team_chemistry(df_ice)
team_chem_nor = get_team_chemistry(df_nor)
team_chem_sl = get_team_chemistry(df_sl)


plot_relationship(df_merged_full, 'full')
plot_relationship(df_merged_full, 'full')

df_merged_dk = team_chem_dk.merge(df_expected_dk, on=['teamId'])
df_merged_eng = team_chem_eng.merge(df_expected_eng, on=['teamId'])
df_merged_it = team_chem_it.merge(df_expected_it, on=['teamId'])
df_merged_fr = team_chem_fr.merge(df_expected_fr, on=['teamId'])
df_merged_ger = team_chem_ger.merge(df_expected_ger, on=['teamId'])
df_merged_ned = team_chem_ned.merge(df_expected_ned, on=['teamId'])
df_merged_por = team_chem_por.merge(df_expected_por, on=['teamId'])
df_merged_bel = team_chem_bel.merge(df_expected_bel, on=['teamId'])
df_merged_esp = team_chem_esp.merge(df_expected_esp, on=['teamId'])
df_merged_tur = team_chem_tur.merge(df_expected_tur, on=['teamId'])
df_merged_au = team_chem_au.merge(df_expected_au, on=['teamId'])
df_merged_swi = team_chem_swi.merge(df_expected_swi, on=['teamId'])
df_merged_cyo = team_chem_cyo.merge(df_expected_cyo, on=['teamId'])
df_merged_ser = team_chem_ser.merge(df_expected_ser, on=['teamId'])
df_merged_gre = team_chem_gre.merge(df_expected_gre, on=['teamId'])
df_merged_cro = team_chem_cro.merge(df_expected_cro, on=['teamId'])
df_merged_pol = team_chem_pol.merge(df_expected_pol, on=['teamId'])
df_merged_cze = team_chem_cze.merge(df_expected_cze, on=['teamId'])
df_merged_rus = team_chem_rus.merge(df_expected_rus, on=['teamId'])
df_merged_sco = team_chem_sco.merge(df_expected_sco, on=['teamId'])
df_merged_hun = team_chem_hun.merge(df_expected_hun, on=['teamId'])
df_merged_sk = team_chem_sk.merge(df_expected_sk, on=['teamId'])
df_merged_swe = team_chem_swe.merge(df_expected_swe, on=['teamId'])
df_merged_ice = team_chem_ice.merge(df_expected_ice, on=['teamId'])
df_merged_nor = team_chem_nor.merge(df_expected_nor, on=['teamId'])
df_merged_sl = team_chem_sl.merge(df_expected_sl, on=['teamId'])



f_all = pd.concat([df_merged_it, df_merged_fr, df_merged_eng, df_merged_dk,
                   df_merged_ger, df_merged_ned, df_merged_por,
                   df_merged_esp, df_merged_tur, df_merged_swi,df_merged_gre,
                   df_merged_cro, df_merged_cze, df_merged_rus,
                   df_merged_hun, df_merged_ice, df_merged_nor, df_merged_sl,
                   df_merged_sk
                   ])

#0,52
print("{:.2f}".format((f_all['chemistry'].corr(f_all['expected_wins'], method='spearman'))))


def make_boxplot(df):
    sns.boxplot(data=df[['chemistry']])
    plt.title('Chemistry Boxplot')
    plt.show()

    sns.boxplot(data=df[['expected_wins']])
    plt.title('expected_wins Boxplot')
    plt.show()

check_dis(f_all, 'chemistry')
check_dis(f_all, 'expected_wins')

f_all['chemistry'].corr(f_all['expected_wins'], method='spearman')
f_all['chemistry'].corr(f_all['expected_losses'], method='spearman')

# Assuming X is your data frame containing the chemistry and expected wins features
# Calculate the mean and covariance matrix of the data



for i in range(20, 101):
    #wins_chem = f_all[['chemistry', 'expected_wins']]
    mean = np.mean(f_all, axis=0)
    cov = np.cov(f_all, rowvar=False)

    # Calculate the Mahalanobis distance for each data point
    dist = np.apply_along_axis(lambda x: np.sqrt((x - mean).T.dot(np.linalg.inv(cov)).dot(x - mean)), 1, f_all)

    # Set a threshold for outlier detection
    threshold = np.percentile(dist, i)

    # Identify outliers
    outliers = dist > threshold

    # Subset the data to include only non-outliers
    wins_chem_clean = f_all[~outliers]

    corr = wins_chem_clean['chemistry'].corr(wins_chem_clean['expected_wins'], method='pearson')
    corr_v2 = wins_chem_clean['chemistry'].corr(wins_chem_clean['expected_losses'], method='spearman')
    print(i)
    print(corr, corr_v2, sep=(" "))
    print()



plot_relationship(df_merged_dk, 'DK')
plot_relationship(df_merged_eng, 'ENG')
plot_relationship(df_merged_it, 'IT')
plot_relationship(df_merged_fr, 'FR')
plot_relationship(df_merged_ger, 'GER')
plot_relationship(df_merged_por, 'POR')
plot_relationship(df_merged_ned, 'NED')
plot_relationship(df_merged_bel, 'BEL')
plot_relationship(df_merged_esp, 'ESP')
plot_relationship(df_merged_tur, 'TUR')
plot_relationship(f_all, 'Relationship Chemistry & xW')
plot_relationship(df_merged_au, 'AU')


f_all['chemistry'].corr(f_all['expected_wins'])
f_all_v2['chemistry'].corr(f_all_v2['expected_wins'])


print(df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins']))
print(df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins']))
print(df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins']))
print(df_merged_it['chemistry'].corr(df_merged_it['expected_wins']))
print(df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins']))
print(df_merged_esp['chemistry'].corr(df_merged_esp['expected_wins']))
print(df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins']))
#Out
#df_merged_au['chemistry'].corr(df_merged_au['expected_wins'])

print(df_merged_swi['chemistry'].corr(df_merged_swi['expected_wins']))
#Out
print(df_merged_cyo['chemistry'].corr(df_merged_cyo['expected_wins']))

print(df_merged_ser['chemistry'].corr(df_merged_ser['expected_wins']))
print(df_merged_gre['chemistry'].corr(df_merged_gre['expected_wins']))

print(df_merged_cro['chemistry'].corr(df_merged_cro['expected_wins']))
print(df_merged_pol['chemistry'].corr(df_merged_pol['expected_wins']))
print(df_merged_cze['chemistry'].corr(df_merged_cze['expected_wins']))
print(df_merged_rus['chemistry'].corr(df_merged_rus['expected_wins']))
print(df_merged_sco['chemistry'].corr(df_merged_sco['expected_wins']))
print(df_merged_hun['chemistry'].corr(df_merged_hun['expected_wins']))
print(df_merged_sk['chemistry'].corr(df_merged_sk['expected_wins']))
print(df_merged_swe['chemistry'].corr(df_merged_swe['expected_wins']))

#Fi
df_merged_ice['chemistry'].corr(df_merged_ice['expected_wins'])
df_merged_nor['chemistry'].corr(df_merged_nor['expected_wins'])
df_merged_sl['chemistry'].corr(df_merged_sl['expected_wins'])
df_merged_cyo['chemistry'].corr(df_merged_cyo['expected_wins'])


f_all['chemistry'].corr(f_all['expected_wins'], method='spearman')
f_all_v2['chemistry'].corr(f_all_v2['expected_wins'], method='spearman')
plot_relationship(f_all_v2, 'all')




def check_ranked_corr(df):
    corr, p_value = spearmanr(df['chemistry'], df['expected_wins'])
    print("Spearman's rank correlation coefficient:", corr)

df_no_outliers = remove_outliers_v2(pd.concat([df_merged_dk, df_merged_eng, df_merged_bel, df_merged_fr, df_merged_ger, df_merged_it, df_merged_ned, df_merged_por, df_merged_esp, df_merged_tur]))

plot_relationship(df_no_outliers)
perform_linear_regression(df_no_outliers)
make_boxplot(df_no_outliers)

print(df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins']))
print(df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins']))
df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins'])
df_merged_it['chemistry'].corr(df_merged_it['expected_wins'])

df_merged_esp['chemistry'].corr(df_merged_esp['expected_wins'])
df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins'])
df_merged_ger['chemistry'].corr(df_merged_ger['expected_wins'])
df_merged_ned['chemistry'].corr(df_merged_ned['expected_wins'])
df_merged_por['chemistry'].corr(df_merged_por['expected_wins'])
df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins'])

print("Correlation between 'chemistry' and 'expected_wins':", corr)

por_it_ger_ned = pd.concat([df_merged_ned, df_merged_ger, df_merged_it, df_merged_por])
plot_relationship(por_it_ger_ned, 'chosen')

por_it_ger_ned['chemistry'].corr(por_it_ger_ned['expected_wins'])





df_swi, df_process_21_22_swi, pl_avg_pos_swi, games_players_minutes_swi, zones_actions_swi = generate_chemistry(df_keepers, df_squad, league_df_swi, df_pos,df_matches_all)
df_cyo, df_process_21_22_cyo, pl_avg_pos_cyo, games_players_minutes_cyo, zones_actions_cyo = generate_chemistry(df_keepers, df_squad, league_df_cyo, df_pos,df_matches_all)
df_ser, df_process_21_22_ser, pl_avg_pos_ser, games_players_minutes_ser, zones_actions_ser = generate_chemistry(df_keepers, df_squad, league_df_ser, df_pos,df_matches_all)
df_gre, df_process_21_22_gre, pl_avg_pos_gre, games_players_minutes_gre, zones_actions_gre = generate_chemistry(df_keepers, df_squad, league_df_gre, df_pos,df_matches_all)
df_cro, df_process_21_22_cro, pl_avg_pos_cro, games_players_minutes_cro, zones_actions_cro = generate_chemistry(df_keepers, df_squad, league_df_cro, df_pos,df_matches_all)
df_pol, df_process_21_22_pol, pl_avg_pos_pol, games_players_minutes_pol, zones_actions_pol = generate_chemistry(df_keepers, df_squad, league_df_pol, df_pos,df_matches_all)
df_cze, df_process_21_22_cze, pl_avg_pos_cze, games_players_minutes_cze, zones_actions_cze = generate_chemistry(df_keepers, df_squad, league_df_cze, df_pos,df_matches_all)
df_rus, df_process_21_22_rus, pl_avg_pos_rus, games_players_minutes_rus, zones_actions_rus = generate_chemistry(df_keepers, df_squad, league_df_rus, df_pos,df_matches_all)
df_sco, df_process_21_22_sco, pl_avg_pos_sco, games_players_minutes_sco, zones_actions_sco = generate_chemistry(df_keepers, df_squad, league_df_sco, df_pos,df_matches_all)
df_hun, df_process_21_22_hun, pl_avg_pos_hun, games_players_minutes_hun, zones_actions_hun = generate_chemistry(df_keepers, df_squad, league_df_hun, df_pos,df_matches_all)
df_sk, df_process_21_22_sk, pl_avg_pos_sk, games_players_minutes_sk, zones_actions_sk = generate_chemistry(df_keepers, df_squad, league_df_sk, df_pos,df_matches_all)
df_swe, df_process_21_22_swe, pl_avg_pos_swe, games_players_minutes_swe, zones_actions_swe = generate_chemistry(df_keepers, df_squad, league_df_swe, df_pos,df_matches_all)
df_fin, df_process_21_22_fin, pl_avg_pos_fin, games_players_minutes_fin, zones_actions_fin = generate_chemistry(df_keepers, df_squad, league_df_fin, df_pos,df_matches_all)
df_ice, df_process_21_22_ice, pl_avg_pos_ice, games_players_minutes_ice, zones_actions_ice = generate_chemistry(df_keepers, df_squad, league_df_ice, df_pos,df_matches_all)
df_nor, df_process_21_22_nor, pl_avg_pos_nor, games_players_minutes_nor, zones_actions_nor = generate_chemistry(df_keepers, df_squad, league_df_nor, df_pos,df_matches_all)
df_sl, df_process_21_22_sl, pl_avg_pos_sl, games_players_minutes_sl, zones_actions_sl = generate_chemistry(df_keepers, df_squad, league_df_sl, df_pos,df_matches_all)
total_player_chemistry_scores = pd.concat([df_dk, df_fr, df_it, df_eng, df_ger, df_por, df_ned, df_esp, df_tur, df_bel, df_au, df_swi, df_cyo, df_ser, df_gre, df_cro, df_pol, df_cze, df_rus, df_rus, df_sco, df_hun, df_sk, df_swe, df_fin, df_ice, df_nor, df_sl])
duplicate = total_player_chemistry_scores[total_player_chemistry_scores.duplicated(['p1', 'p2', 'seasonId'])]



total_player_chemistry_scores.columns.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry.csv", index=False)
df_chem.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_new_formula_v2.csv")
df_merged_full[total_player_chemistry_scores.columns].to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv", index=False)


df_merged_swi = team_chem_swi.merge(df_expected_swi, on=['teamId'])
df_merged_cyo = team_chem_cyo.merge(df_expected_cyo, on=['teamId'])
df_merged_ser = team_chem_ser.merge(df_expected_ser, on=['teamId'])
df_merged_gre = team_chem_gre.merge(df_expected_gre, on=['teamId'])
df_merged_cro = team_chem_cro.merge(df_expected_cro, on=['teamId'])
df_merged_pol = team_chem_pol.merge(df_expected_pol, on=['teamId'])
df_merged_cze = team_chem_cze.merge(df_expected_cze, on=['teamId'])
df_merged_rus = team_chem_rus.merge(df_expected_rus, on=['teamId'])
df_merged_sco = team_chem_sco.merge(df_expected_sco, on=['teamId'])
df_merged_hun = team_chem_hun.merge(df_expected_hun, on=['teamId'])
df_merged_sk = team_chem_sk.merge(df_expected_sk, on=['teamId'])
df_merged_swe = team_chem_swe.merge(df_expected_swe, on=['teamId'])
df_merged_fin = team_chem_fin.merge(df_expected_in, on=['teamId'])
df_merged_ice = team_chem_ice.merge(df_expected_ice, on=['teamId'])
df_merged_nor = team_chem_nor.merge(df_expected_nor, on=['teamId'])
df_merged_sl = team_chem_sl.merge(df_expected_sl, on=['teamId'])

players_actions_success = pd.concat([m_dk_gr, m_eng_gr, m_fr_gr, m_it_gr, m_por_gr, m_bel_gr, m_tur_gr, m_esp_gr, m_ned_gr, m_ger_gr, m_au_gr, m_swi_gr, m_cyo_gr, m_ser_gr, m_gre_gr, m_cro_gr,
                                     m_pol_gr, m_cze_gr, m_rus_gr, m_sco_gr, m_hun_gr, m_sk_gr, m_swe_gr, m_fin_gr, m_ice_gr, m_nor_gr, m_sl_gr])
players_actions_success.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_actions_success.csv", index=False)
players_actions_success.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_actions_success_v2.csv", index=False)
performance_stats.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/performance_stats.csv", index=False)
performance_stats.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/performance_stats_v2.csv", index=False)

pc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv")
chem_ability_v2 = generate_chemistry_ability_v3(pc)
pc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv")

pc.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/chemAbility.csv", index=False)
pc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv")

pc.seasonId.unique()

sd_table_and_succ = sd_table.merge(df_s)
df_act_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_actions_success_v2.csv")
sd_table_and_succ.seasonId.unique()

sd_table_and_succ= pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_table_and_succ_columns.csv")

sd_table_and_succ = sd_table.merge(df_successes, left_on='eventId', right_on='id')
sd_table_and_succ.seasonId.unique()

sd_table_and_succ = sd_table_and_succ[(sd_table_and_succ['accurate'] == 1) | (sd_table_and_succ['not_accurate'] == 1)]
sd_table_and_succ = sd_table_and_succ.drop_duplicates()
sd_table_and_succ = sd_table_and_succ.drop('playerId_y', axis = 1)
sd_table_and_succ = sd_table_and_succ.rename(columns= {'playerId_x': 'playerId'})

def derive_accurate_scores(df):
    df_final = pd.DataFrame()
    # Create an empty dictionary to store the sums
    scores = {}
    columns = ['aerial_duel', 'assist', 'carry', 'counterpressing_recovery', 'deep_completed_cross','deep_completition', 'defensive_duel', 'dribble', 'ground_duel', 'ground_duel', 'pass_into_penalty_area', 'pass_to_final_third', 'second_assist', 'sliding_tackle']
    for i in columns:
        # Select the rows where the column i is 1 and the 'accurate' column is 1
        df_a = df[[ 'playerId', i, 'accurate']]
        df_a = df_a[(df_a[i] == 1) & (df['accurate'] == 1)]

        # Group the data by playerId and sum the values in the i column
        df_g = df_a.groupby(['playerId'], as_index=False).agg({i: 'sum'})

        # Add the sum to the scores dictionary
        scores[i] = df_g
    for key, value in scores.items():
        if len(df_final) == 0:
            df_final = value
        else: df_final = df_final.merge(value, on = ['playerId'], how = 'left')

    return df_final

def create_strenghts(df):
    df['aerial_strength'] = df['accurate_aerial_duels'] / df['aerial_duel']
    df['carry_strength'] = df['accurate_carries'] / df['carry']
    df['pressing_recovery_strength'] = df['accurate_counterpressing_recoveries'] / df['counterpressing_recovery']
    df['defensive_duel_strength'] = df['accurate_defensive_duels'] / df['defensive_duel']
    df['dribbles_strength'] = df['accurate_dribbles'] / df['dribble']
    df['ground_duel_strength'] = df['accurate_ground_duels'] / df['ground_duel']
    df['sliding_strength'] = df['accurate_sliding_tackles'] / df['sliding_tackle']
    df['deep_crossing_strength'] = df['accurate_deep_completed_crosses'] / df['deep_completed_cross']
    return df

df_acc = derive_accurate_scores(sd_table_and_succ)
chem_ability_all = generate_chemistry_ability_v3(pc)

df_acc = df_acc.rename(columns= {'aerial_duel': 'accurate_aerial_duels', 'assist': 'accurate_assists', 'carry':'accurate_carries', 'counterpressing_recovery': 'accurate_counterpressing_recoveries', 'deep_completed_cross': 'accurate_deep_completed_crosses', 'deep_completition': 'accurate_deep_completitions', 'defensive_duel': 'accurate_defensive_duels', 'dribble': 'accurate_dribbles', 'ground_duel': 'accurate_ground_duels', 'pass_into_penalty_area': 'accurate_passes_into_penalty_area', 'pass_to_final_third': 'accurate_pass_to_final_third', 'second_assist': 'accurate_second_assists', 'sliding_tackle': 'accurate_sliding_tackles'})
df_p_tot = sd_table_and_succ.groupby(['playerId'], as_index = False).agg(
    {'aerial_duel': 'sum', 'assist': 'sum', 'carry': 'sum', 'counterpressing_recovery': 'sum', 'deep_completed_cross':'sum', 'deep_completition': 'sum',
     'defensive_duel': 'sum', 'dribble': 'sum', 'ground_duel': 'sum', 'ground_duel': 'sum', 'pass_into_penalty_area': 'sum', 'pass_to_final_third': 'sum',
     'second_assist': 'sum', 'sliding_tackle': 'sum'})
df_acc_t_merged = df_acc.merge(df_p_tot, on = ['playerId'])
df_acc_t_merged = df_acc_t_merged.drop(['accurate_deep_completitions', 'deep_completition', 'accurate_passes_into_penalty_area', 'accurate_assists', 'accurate_passes_into_penalty_area', 'accurate_deep_completitions', 'accurate_second_assists'], axis = 1)
df_with_strengths = create_strenghts(df_acc_t_merged)
chem_ability_all = generate_chemistry_ability_v3(pc)
chem_ability_all_with_performance_metrics = chem_ability_all.merge(df_with_strengths, on = ['playerId'], how = 'left')
chem_ability_all_with_performance_metrics.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_table_and_performance_and_chemi_power.csv", index=False)


sd_table_and_succ['tester'] = sd_table_and_succ['playerId_x'] == sd_table_and_succ['playerId_y']
df_overview_bif.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/bif_chemistry.csv", decimal=',', sep=(';'), index=False)
df_overview_bif_avg.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/bif_chemistry_avg.csv", decimal=',', sep=(';'), index=False)
h.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/cluster_and_chem_AVG.csv", decimal=',', sep=(';'), index=False)

h =  results_df.merge(df_overview_bif_avg, on='playerId')

df_chem_all = pd.concat([
                   df_dk, df_fr, df_eng, df_dk,
                   df_ger, df_ned, df_por, df_esp,
                   df_tur, df_swi, df_gre, df_cro,
                   df_cze, df_rus, df_hun, df_ice,
                   df_sl, df_sk])
df_chem = (get_chemistry(for_scaling, joi_jdi, ptc))[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90', 'winners90', 'chemistry']]
df_chem_all = df_chem_all[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90', 'winners90', 'chemistry']]
df_chem_all.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_10.csv", index=False)