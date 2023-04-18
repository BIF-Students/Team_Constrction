# Import necessary modules
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from chemistry.distance import *
from chemistry.jdi import *
from chemistry.joi import *
from chemistry.netoi import *
from chemistry.responsibility_share import *
from chemistry.smallTest import test_players_in_a_match
from chemistry.sql_statements import *
from helpers.student_bif_code import load_db_to_pd  # custom module
from chemistry.chemistry_helpers import *
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr

sd_table, df_matches_all, df_sqaud, df_keepers, df_players_teams, df_events_goals, df_pos = load_data(competitionIds="412, 426, 335, 707, 635, 852, 198, 795, 524, 364")

sd_table = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/sd_table.csv")
df_events_goals = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_events_goals.csv")
df_keepers = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_keepers.csv")
df_players_teams = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_players_teams.csv")
df_pos = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_pos.csv")
df_sqaud = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/data/df_sqaud.csv")



df_matches_all_esp = load_db_to_pd(sql_query=  "select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
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



#---ENG----------------------------------------------------------------------------------------------------------------------------------------------
#Extract keeper id's
keepers = (df_keepers.playerId.values).tolist()

#Remove keepers from all used dataframes
df_sqaud = df_sqaud.query("playerId not in @keepers")
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



#---ESP--------------------------------------------------------------------------------------------------------------------------------------------------------
#Extract keeper id's
keepers = (df_keepers.playerId.values).tolist()

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

df_overview_esp = get_overview_frame(df_chemistry_esp, df_players_teams_esp)
#-------------------------------------------------------------------------------------------------------------------------------------

chem_ability_v2_eng = generate_chemistry_ability_v2(df_overview)
chem_ability_v2_esp = generate_chemistry_ability_v2(df_overview_esp)


#---IT----------------------------------------------------------------------------------------------------------------------------------------------
#Remove keepers from all used dataframes
keepers = (df_keepers.playerId.values).tolist()

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

#----------------------------------------------------------------------------------------------------


def generate_chemistry(df_keepers, squad, sd_table, pos, comId):

    # Remove keepers from all used dataframes
    keepers = (df_keepers.playerId.values).tolist()

    squad = squad.query("playerId not in @keepers")
    sd_table = sd_table.query("playerId not in @keepers")

    # Remove players not in the starting 11
    squad = squad[(squad.bench == False)]

    # Impute zero values for sumVaep
    sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
    pos['sumVaep'] = pos['sumVaep'].fillna(0)

    league_df = sd_table[sd_table['competitionId'] == comId]

    df_process = league_df.copy()
    s_21_22 = max(df_process.seasonId)
    df_matches_all = load_db_to_pd(
        sql_query="select matchId, home_teamId, away_teamId from [Scouting_Raw].[dbo].[Wyscout_Matches_All] "
                  "WHERE matchId IN "
                  "(SELECT matchId from Scouting.dbo.Wyscout_Matches "
                  "where Scouting.dbo.Wyscout_Matches.seasonId =%s)" % s_21_22, db_name='Development')
    df_process_20_21 = df_process[df_process['seasonId'] == min(df_process.seasonId)]
    df_process_21_22 = df_process[df_process['seasonId'] == s_21_22]
    df_sqaud_filtered = squad[squad['seasonId'] == s_21_22]
    pos_filtered = pos[pos['seasonId'] == s_21_22]
    pos_sorted = pos_filtered.sort_values(by=['possessionId', 'possessionEventIndex'], ascending=True)
    df_pairwise_playing_time = pairwise_playing_time(df_sqaud_filtered)

    # Extract net offensive impact per game per team
    df_net_oi = getOi(df_process.copy())
    # Extract distance measures
    df_ec = getDistance(df_process_21_22.copy())
    # Extract players shares
    df_player_share = getResponsibilityShare((df_process_21_22.copy()))
    # Extract jdi
    df_jdi = get_jdi(df_player_share, df_ec, df_net_oi, df_matches_all)

    df_joi = get_joi(pos_sorted)
    df_joi90_and_jdi90 = compute_normalized_values(df_joi, df_jdi, df_pairwise_playing_time)

    stamps = get_timestamps(max(df_process.seasonId))
    ready_for_scaling = prepare_for_scaling(df_process_21_22, df_sqaud_filtered, stamps)
    ready_for_scaling['seasonId'] = s_21_22
    df_joi90_and_jdi90['seasonId'] = s_21_22
    df_players_teams_c = df_players_teams[df_players_teams['teamId'].isin(df_joi90_and_jdi90['teamId'].unique())]
    df_chemistry = (get_chemistry(ready_for_scaling, df_joi90_and_jdi90, df_players_teams_c))[['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'joi', 'jdi', 'df_jdi90', 'df_joi90', 'winners90', 'chemistry']]
    return df_chemistry, df_process_21_22

#---Expected goals
def compute_expected_wins(df):
    xg_game = df.groupby(['teamId', 'matchId'], as_index = False)['xG'].sum()
    xg_game_v2 = xg_game.merge(xg_game,  on='matchId')
    xg_game_v2 = xg_game_v2[xg_game_v2['teamId_x'] != xg_game_v2['teamId_y'] ]
    xg_game_v2 = xg_game_v2.drop_duplicates(subset='matchId')
    xg_game_v2['expected_wins'] = np.where(xg_game_v2.xG_x > xg_game_v2.xG_y, xg_game_v2.teamId_x,
                                              np.where(xg_game_v2.xG_y > xg_game_v2.xG_x, xg_game_v2.teamId_y, -1))
    df_team_1 = (xg_game_v2[['teamId_x', 'matchId', 'xG_x', 'expected_wins']]).rename(columns = {'teamId_x': 'teamId', 'xG_x': 'xG' })
    df_team_2 = (xg_game_v2[['teamId_y', 'matchId', 'xG_y', 'expected_wins']]).rename(columns = {'teamId_y': 'teamId', 'xG_y': 'xG' })
    df_team_1 = df_team_1[df_team_1['teamId'] == df_team_1['expected_wins']]
    df_team_2 = df_team_2[df_team_2['teamId'] == df_team_2['expected_wins']]
    df_team_1 = df_team_1.groupby('teamId', as_index = False)['expected_wins'].count()
    df_team_2 = df_team_2.groupby('teamId', as_index = False)['expected_wins'].count()
    df_teams_and_wins = pd.concat([df_team_1, df_team_2])
    df_teams_and_wins = df_teams_and_wins.groupby('teamId', as_index = False)['expected_wins'].sum()
    return df_teams_and_wins

def get_summed_xg(df):
    xg_game = df.groupby(['teamId'], as_index = False)['xG'].sum()
    return xg_game


def check_dis(df, feature):
    plt.hist(df[feature], bins=10)
    plt.title("Histogram of " + feature +  " variable")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


def get_average_team_chemistry_tester(df):
    df_team_chem = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].sum()
    #df_team_joi = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].mean()
    #df_team_jdi = df.groupby(['teamId', 'seasonId'], as_index = False)['chemistry'].mean()
    return df_team_chem

def plot_relationship(df1, label):
    # Extract 'chemistry' and 'expected_wins' columns from both DataFrames
    chemistry1 = df1['chemistry']
    expected_wins1 = df1['expected_wins']

    # Create a scatter plot with the first DataFrame
    plt.scatter(chemistry1, expected_wins1, label='DataFrame 1')

    plt.xlabel('Chemistry')
    plt.ylabel('Expected Wins')
    plt.title(label)
    plt.legend()  # Show legend with labels for each DataFrame
    plt.show()

def make_boxplot(df):
    data_melted = pd.melt(df, var_name='variable', value_name='value')
    sns.boxplot(data=data_melted, x='variable', y='value', hue='category')
    plt.show()


df_dk, df_process_21_22_dk = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,335)
df_eng, df_process_21_22_eng = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,364)
df_fr, df_process_21_22_fr = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,412)
df_it, df_process_21_22_it = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,524)
df_ger, df_process_21_22_ger = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,426)
df_por, df_process_21_22_por = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,707)
df_ned, df_process_21_22_ned = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,635)
df_esp, df_process_21_22_esp = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,795)
df_tur, df_process_21_22_tur = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,852)
df_bel, df_process_21_22_bel = generate_chemistry(df_keepers, df_sqaud, sd_table, df_pos,198)


df_expected_wins_dk = compute_expected_wins(df_process_21_22_dk)
df_expected_wins_eng = compute_expected_wins(df_process_21_22_eng)
df_expected_wins_fr = compute_expected_wins(df_process_21_22_fr)
df_expected_wins_it = compute_expected_wins(df_process_21_22_it)
df_expected_wins_ger = compute_expected_wins(df_process_21_22_ger)
df_expected_wins_por = compute_expected_wins(df_process_21_22_por)
df_expected_wins_ned = compute_expected_wins(df_process_21_22_ned)
df_expected_wins_bel = compute_expected_wins(df_process_21_22_bel)
df_expected_wins_esp = compute_expected_wins(df_process_21_22_esp)
df_expected_wins_tur = compute_expected_wins(df_process_21_22_tur)

team_chem_dk = get_average_team_chemistry_tester(df_dk)
team_chem_eng = get_average_team_chemistry_tester(df_eng)
team_chem_fr = get_average_team_chemistry_tester(df_fr)
team_chem_it = get_average_team_chemistry_tester(df_it)
team_chem_ger = get_average_team_chemistry_tester(df_ger)
team_chem_ned = get_average_team_chemistry_tester(df_ned)
team_chem_por = get_average_team_chemistry_tester(df_por)
team_chem_bel = get_average_team_chemistry_tester(df_bel)
team_chem_esp = get_average_team_chemistry_tester(df_esp)
team_chem_tur = get_average_team_chemistry_tester(df_tur)

df_merged_dk = team_chem_dk.merge(df_expected_wins_dk, on=['teamId'])
df_merged_eng = team_chem_eng.merge(df_expected_wins_eng, on=['teamId'])
df_merged_it = team_chem_it.merge(df_expected_wins_it, on=['teamId'])
df_merged_fr = team_chem_fr.merge(df_expected_wins_fr, on=['teamId'])
df_merged_ger = team_chem_ger.merge(df_expected_wins_ger, on=['teamId'])
df_merged_ned = team_chem_ned.merge(df_expected_wins_ned, on=['teamId'])
df_merged_por = team_chem_por.merge(df_expected_wins_por, on=['teamId'])
df_merged_bel = team_chem_bel.merge(df_expected_wins_bel, on=['teamId'])
df_merged_esp = team_chem_esp.merge(df_expected_wins_esp, on=['teamId'])
df_merged_tur = team_chem_tur.merge(df_expected_wins_tur, on=['teamId'])

f_all = pd.concat([df_merged_it, df_merged_fr, df_merged_eng, df_merged_dk, df_merged_ger, df_merged_ned, df_merged_por, df_merged_bel, df_merged_esp, df_merged_tur])
def make_boxplot(df):
    sns.boxplot(data=df[['chemistry']])
    plt.title('Chemistry Boxplot')
    plt.show()

    sns.boxplot(data=df[['expected_wins']])
    plt.title('expected_wins Boxplot')
    plt.show()


def remove_outliers(df):
    Q1 = df['chemistry'].quantile(0.25)
    Q3 = df['chemistry'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['chemistry'] >= Q1 - 1.5 * IQR) & (df['chemistry'] <= Q3 + 1.5 * IQR)]
    return df
make_boxplot((remove_outliers(f_all)))

check_dis(f_all, 'chemistry')
check_dis(f_all, 'expected_wins')
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
plot_relationship(f_all, 'all')

df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins'])
df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins'])
df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins'])
df_merged_it['chemistry'].corr(df_merged_it['expected_wins'])
df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins'])
df_merged_esp['chemistry'].corr(df_merged_esp['expected_wins'])
df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins'])
f_all['chemistry'].corr(f_all['expected_wins'], method='spearman')


df_summed_xg_dk = get_summed_xg(df_process_21_22_dk)
df_summed_xg_eng = get_summed_xg(df_process_21_22_eng)
df_summed_xg_bel = get_summed_xg(df_process_21_22_bel)
df_summed_xg_fr = get_summed_xg(df_process_21_22_fr)
df_summed_xg_ger = get_summed_xg(df_process_21_22_ger)
df_summed_xg_it = get_summed_xg(df_process_21_22_it)
df_summed_xg_ned = get_summed_xg(df_process_21_22_ned)
df_summed_xg_por = get_summed_xg(df_process_21_22_por)
df_summed_xg_esp = get_summed_xg(df_process_21_22_esp)
df_summed_xg_tur = get_summed_xg(df_process_21_22_tur)




def perform_linear_regression(df):
    # Extract 'chemistry' and 'expected_wins' columns
    X = df['chemistry']
    y = df['expected_wins']

    # Add a constant term to the predictor variable for the intercept term
    X = sm.add_constant(X)

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Print the model summary
    print(model.summary())
    print("p-value for 'chemistry' coefficient:", model.pvalues['chemistry'])


def remove_outliers_v2(df):
    outliers = df[(df['chemistry'] > df['chemistry'].quantile(0.75) + 1.5 * (
                df['chemistry'].quantile(0.75) - df['chemistry'].quantile(0.25))) |
                  (df['chemistry'] < df['chemistry'].quantile(0.25) - 1.5 * (
                              df['chemistry'].quantile(0.75) - df['chemistry'].quantile(0.25)))]

    # Remove the outliers from the DataFrame
    df = df.drop(outliers.index)

    return df

def check_ranked_corr(df):
    corr, p_value = spearmanr(df['chemistry'], df['expected_wins'])
    print("Spearman's rank correlation coefficient:", corr)

df_no_outliers = remove_outliers_v2(pd.concat([df_merged_dk, df_merged_eng, df_merged_bel, df_merged_fr, df_merged_ger, df_merged_it, df_merged_ned, df_merged_por, df_merged_esp, df_merged_tur]))

plot_relationship(df_no_outliers)
perform_linear_regression(df_no_outliers)
make_boxplot(df_no_outliers)

df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins'])
df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins'])
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












