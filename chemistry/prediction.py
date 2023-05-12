from chemistry.chemistry_helpers import *
from datetime import datetime
import sys
from regression import *
from sklearn.decomposition import PCA
'''import sys
sys.path.append('C:/Users/jhs/factor_analyzer/factor_analyzer-main')
sys.path.append('C:/Users/jhs/umap/\pynndescent')
sys.path.append('C:/Users/jhs/\pynndescent/umap-master')
from factor_analyzer import FactorAnalyzer'''
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import cross_val_score, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



#Load data from csv files
#players_chemistry = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_v2.csv")
players_chemistry = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_new_formula.csv")
players_chemistry_v2 = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_chemistry_new_formula_v2.csv")
players_roles = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_clusters.csv")
df_transfer = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_transfer.csv")
df_players = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_players.csv")
df_teams = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_teams.csv")
df_seasons_competitions = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_seasons_competitions.csv")
df_matches_all = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_matches_all.csv")
df_areas = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/area_db.csv")
df_act_suc = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/players_actions_success_v2.csv")
df_performance_stats = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/performance_stats_v2.csv")
df_act_suc = df_act_suc[df_act_suc['playerId'] != 0]
df_chem_power_and_strength = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_table_and_performance_and_chemi_power.csv")

new_arrivals = get_new_arrivals(df_transfer, players_chemistry_v2)
training_set, test_set = get_chem_profficiency(new_arrivals, players_chemistry_v2)


#SQL
'''
df_transfer = get_transfers()
df_players = get_all_players()
df_teams = get_all_teams()
df_seasons_competitions = get_seasons_and_competitions()
df_matches_all = get_all_matches()
'''
df_players = df_players.dropna(subset=['birthDate'])
df_players = df_players.dropna(subset=['passportArea_name'])
df_transfer = df_transfer[df_transfer['startDate'] != '0000-00-00']
df_transfer = df_transfer[df_transfer['endDate'].notna()]

df_matches_all_filtered = df_matches_all[['seasonId', 'competitionId', 'home_teamId', 'away_teamId']]
df_matches_restructured = restructure_matches(df_matches_all_filtered)

df_players = df_players.merge(df_areas,left_on='passportArea_name', right_on='name')
df_players_filtered = (df_players[['playerId', 'shortName', 'birthDate', 'height', 'weight', 'birthArea_name', 'passportArea_name', 'role_name', 'currentTeamId']]).rename(columns={'role_name': 'position'})
df_players_filtered['age'] = df_players_filtered.apply(lambda row: get_age(row['birthDate']), axis=1)

#Fix countries - at another time
#df_playerssdf['passportArea_name'].unique()

#Not used at this time 26.04.2023
#df_transfers_players_curr_team = df_transfer.merge(df_players[['playerId', 'currentTeamId']], on='playerId')



def find_players_and_countries(df, transfer, teams):
    df_teams_and_transfers = pd.merge(transfer, teams, left_on='fromTeamId', right_on='teamId')
    df_teams_and_transfers = df_teams_and_transfers.merge(teams, left_on='toTeamId', right_on='teamId')
    df_teams_and_transfers['date'] = df_teams_and_transfers.apply(lambda row: datetime.strptime(row['startDate'], '%Y-%m-%d').date(), axis=1)
    p_dict = {}
    players = df['playerId'].unique()
    for e in players:
        rows = df_teams_and_transfers[df_teams_and_transfers['playerId'] == e]
        rows = rows.sort_values(by=['date'], ascending=True)
        row = rows.iloc[0]
        startCountry = row['areaName_x']
        df_a = df[df['playerId'] == e]
        countries = df_a['areaName_y'].unique()
        countries = countries.tolist()
        if startCountry not in countries:
            countries.append(startCountry)
        p_dict[e] = {'playerId': e, 'countries': countries}
    df_created = pd.DataFrame.from_dict(p_dict, orient='index').reset_index(drop=True)
    return df_created

def create_indicators(df):
    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'df_joi90','df_jdi90', 'countries_x', 'shortName_x', 'birthDate_x', 'height_x',
       'weight_x', 'birthArea_name_x', 'passportArea_name_x', 'position_x',
       'currentTeamId_x', 'age_x', 'pos_group_x', 'ip_cluster_x','zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x',
       'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'minutes_played_season_x', 'match appearances_x', 'chem_coef_x',
       'countries_y', 'shortName_y', 'birthDate_y', 'height_y',
       'weight_y', 'birthArea_name_y', 'passportArea_name_y', 'position_y',
       'currentTeamId_y', 'age_y', 'pos_group_y', 'ip_cluster_y','minutes_played_season_y',
       'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
       'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y'
             ]]
    df['same_origin'] = np.where(df['birthArea_name_x'] == df['birthArea_name_y'], 1, 0 )
    df['same_country'] = np.where(df['passportArea_name_x'] == df['passportArea_name_y'], 1, 0 )
    df['played_in_same_country'] = df.apply(lambda row: 1 if (len((check_country(row['countries_x'], row['countries_y']))) > 0) else 0, axis=1)

    df = df[['p1', 'p2', 'teamId', 'seasonId', 'chemistry', 'df_joi90','df_jdi90', 'same_origin', 'same_country', 'played_in_same_country', 'height_x',
       'weight_x', 'position_x', 'age_x', 'pos_group_x', 'ip_cluster_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x',
       'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x',
       'height_y','weight_y', 'position_y','age_y', 'pos_group_y', 'ip_cluster_y', 'minutes_played_season_y',
       'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
       'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y' ]]
    return df

def elapsed_time(start_date_str, end_date_str):
    if end_date_str != '0000-00-00':
        start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        delta = end - start
        return delta.days
    else: return 0


def handle_transfer_periods(df):
    df['transfer_time'] = df.apply(lambda row: elapsed_time(row['startDate'], row['endDate']), axis = 1)
    return df


def handle_transfer_data(transfer, teams):
    df_teams_and_transfers = pd.merge(transfer, teams, left_on='fromTeamId', right_on='teamId')
    df_teams_and_transfers = df_teams_and_transfers.merge(teams, left_on='toTeamId', right_on='teamId')
    df_teams_and_transfers = df_teams_and_transfers[['toTeamId', 'fromTeamId', 'startDate', 'endDate', 'transfer_time', 'areaName_x', 'areaName_y', 'timestamp', 'playerId']]
    transfers_handled_V2 = df_teams_and_transfers.groupby(['playerId', 'areaName_y'], as_index=False)['transfer_time'].sum()
    transfers_handled_V2['years_in_country'] = round((transfers_handled_V2['transfer_time'] / 365.2425), 2)
    transfers_handled_V2 = transfers_handled_V2[transfers_handled_V2['years_in_country'] >=2]
    return transfers_handled_V2


def add_roles(df, roles):
    merged = df.merge(df, roles, left_on='p1', right_on='playerId')
    merged2 = df.merge(merged, roles, left_on='p2', right_on='playerId')
    return merged2


def duplicate_and_order_data(df):
    df1 = df.copy()
    df1 = df1.rename(columns={'height_y': 'height_1','weight_y': 'weight_1', 'position_y': 'position_1', 'age_y': 'age_1', 'pos_group_y': 'pos_group_1' , 'ip_cluster_y': 'ip_cluster_1', 'minutes_played_season_y':  'minutes_played_season_1', 'match appearances_y': 'match appearances_1', 'zone_1_pl_y' : 'zone_1_pl_1',  'zone_2_pl_y': 'zone_2_pl_1', 'zone_3_pl_y': 'zone_3_pl_1', 'zone_4_pl_y': 'zone_4_pl_1', 'zone_5_pl_y': 'zone_5_pl_1', 'zone_6_pl_y': 'zone_6_pl_1', 'chem_coef_y': 'chem_coef_1' })

    df1 = df1.rename(columns={'height_x': 'height_y', 'weight_x': 'weight_y', 'position_x': 'position_y','age_x': 'age_y', 'pos_group_x': 'pos_group_y', 'ip_cluster_x': 'ip_cluster_y', 'minutes_played_season_x':  'minutes_played_season_y', 'match appearances_x': 'match appearances_y', 'zone_1_pl_x' : 'zone_1_pl_y','zone_2_pl_x': 'zone_2_pl_y', 'zone_3_pl_x': 'zone_3_pl_y', 'zone_4_pl_x': 'zone_4_pl_y', 'zone_5_pl_x': 'zone_5_pl_y', 'zone_6_pl_x': 'zone_6_pl_y', 'chem_coef_x': 'chem_coef_y' })

    df1 = df1.rename(columns={'height_1': 'height_x','weight_1': 'weight_x', 'position_1': 'position_x', 'age_1': 'age_x', 'pos_group_1': 'pos_group_x', 'ip_cluster_1': 'ip_cluster_x', 'minutes_played_season_1':  'minutes_played_season_x', 'match appearances_1': 'match appearances_x', 'zone_1_pl_1': 'zone_1_pl_x', 'zone_2_pl_1': 'zone_2_pl_x', 'zone_3_pl_1': 'zone_3_pl_x', 'zone_4_pl_1': 'zone_4_pl_x', 'zone_5_pl_1': 'zone_5_pl_x', 'zone_6_pl_1': 'zone_6_pl_x', 'chem_coef_1': 'chem_coef_x' })
    return df1

'''

    df1 = df1.rename(columns={'height_y': 'height_1','weight_y': 'weight_1', 'position_y': 'position_1', 'age_y': 'age_1', 'pos_group_y': 'pos_group_1' , 'ip_cluster_y': 'ip_cluster_1', 'minutes_played_season_y':  'minutes_played_season_1', 'match appearances_y': 'match appearances_1', 'zone_1_pl_y' : 'zone_1_pl_1',
             'zone_2_pl_y': 'zone_2_pl_1', 'zone_3_pl_y': 'zone_3_pl_1', 'zone_4_pl_y': 'zone_4_pl_1', 'zone_5_pl_y': 'zone_5_pl_1', 'zone_6_pl_y': 'zone_6_pl_1', 'chem_ability_y': 'chem_ability_1', 'aerial_strength_y': 'aerial_strength_1', 'carry_strength_y': 'carry_strength_1','pressing_recovery_strength_y' : 'pressing_recovery_strength_1' , 'defensive_duel_strength_y': 'defensive_duel_strength_1','dribbles_strength_y': 'dribbles_strength_1', 'ground_duel_strength_y': 'ground_duel_strength_1', 'sliding_strength_y': 'sliding_strength_1' ,'deep_crossing_strength_y': 'deep_crossing_strength_1' })

    df1 = df1.rename(columns={'height_x': 'height_y', 'weight_x': 'weight_y', 'position_x': 'position_y','age_x': 'age_y', 'pos_group_x': 'pos_group_y', 'ip_cluster_x': 'ip_cluster_y','posAction_x':'posAction_y', 'nonPosAction_x': 'nonPosAction_y', 'accurate_x': 'accurate_y', 'minutes_played_season_x':  'minutes_played_season_y', 'match appearances_x': 'match appearances_y', 'zone_1_pl_x' : 'zone_1_pl_y',
             'zone_2_pl_x': 'zone_2_pl_y', 'zone_3_pl_x': 'zone_3_pl_y', 'zone_4_pl_x': 'zone_4_pl_y', 'zone_5_pl_x': 'zone_5_pl_y', 'zone_6_pl_x': 'zone_6_pl_y', 'chem_ability_x': 'chem_ability_y', 'aerial_strength_x': 'aerial_strength_y', 'carry_strength_x': 'carry_strength_y','pressing_recovery_strength_x' : 'pressing_recovery_strength_y' , 'defensive_duel_strength_x': 'defensive_duel_strength_y','dribbles_strength_x': 'dribbles_strength_y', 'ground_duel_strength_x': 'ground_duel_strength_y', 'sliding_strength_x': 'sliding_strength_y' ,'deep_crossing_strength_x': 'deep_crossing_strength_y' })

    df1 = df1.rename(columns={'height_1': 'height_x','weight_1': 'weight_x', 'position_1': 'position_x', 'age_1': 'age_x', 'pos_group_1': 'pos_group_x', 'ip_cluster_1': 'ip_cluster_x', 'posAction_1':'posAction_x', 'nonPosAction_1': 'nonPosAction_x', 'accurate_1': 'accurate_x', 'minutes_played_season_1':  'minutes_played_season_x', 'match appearances_1': 'match appearances_x', 'minutes_played_season_x':  'minutes_played_season_y', 'match appearances_x': 'match appearances_y', 'zone_1_pl_x': 'zone_1_pl_y',
                              'zone_1_pl_1': 'zone_1_pl_x', 'zone_2_pl_1': 'zone_2_pl_x', 'zone_3_pl_1': 'zone_3_pl_x', 'zone_4_pl_1': 'zone_4_pl_x', 'zone_5_pl_1': 'zone_5_pl_x', 'zone_6_pl_1': 'zone_6_pl_x',
                              'chem_ability_1': 'chem_ability_x', 'aerial_strength_1': 'aerial_strength_x', 'carry_strength_1': 'carry_strength_x','pressing_recovery_strength_1' : 'pressing_recovery_strength_x' , 'defensive_duel_strength_1': 'defensive_duel_strength_x','dribbles_strength_1': 'dribbles_strength_x', 'ground_duel_strength_1': 'ground_duel_strength_x', 'sliding_strength_1': 'sliding_strength_x' ,'deep_crossing_strength_1': 'deep_crossing_strength_x' })

'''

def reg_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean squared error: {mse:.4f}')
    print(f'Root mean squared error: {rmse:.4f}')
    print(f'R-squared: {r2:.4f}')

def evaluate_model_tt(target_pred_test, target_preds_train, target_test, target_train, modelType, reg):
    f1 = f1_score(target_test, target_pred_test, average='weighted')
    if (f1* 100) > 72:
        print(reg)
        cohen_kappa = cohen_kappa_score(target_test, target_pred_test)
        print('Model accuracy score train ' + modelType + ': {0:0.4f}'.format(accuracy_score(target_train, target_preds_train)))
        print('Model accuracy score ' + modelType + ': {0:0.4f}'.format(accuracy_score(target_test, target_pred_test)))
        print('F1 Score: ', "%.2f" % (f1 * 100))
        #print('Cohen Kappa: ', "%.2f" % cohen_kappa)
        print()
       # print(classification_report(target_test, tar))


def check_country(c1, c2):
    return set(c1).intersection(set(c2))

def check_nan(val, area):
    if not isinstance(val, (list)) and math.isnan (val):
        return [area]
    else:return val
def boxplot(df):
    fig = px.box(df, y="chemistry")
    fig.show()
    pyo.plot(fig)

def check_dis(df, label):
    sns.distplot(df[label], hist=False, kde=True,
                 kde_kws={'linewidth': 0.5}, )
    plt.show()
def show_target_distribution(df, label):
    df = df[~df.index.duplicated(keep='first')]
    # Plot density plot of column 'petal_length'
    sns.kdeplot(data=df, x=label)
    plt.xlim(0, 0.06)
    plt.show()

def custom_binning(x):
    if x == 0:
        return 0  # assign 0 bin to 0-chemistry values
    else:
        quartiles = np.percentile(chem_values[chem_values != 0], [25, 50, 75])
        if x <= quartiles[0]:
            return 1
        elif x <= quartiles[1]:
            return 2
        elif x <= quartiles[2]:
            return 3
        else:
            return 4

def produce_overall_cluster(df):
    df['role_x'] = np.where((df.ip_cluster_x >= 0) & (df.ip_cluster_x <= 2), 1,
                                     np.where((df.ip_cluster_x >= 3) & (df.ip_cluster_x <= 5), 2,
                                              np.where((df.ip_cluster_x >= 6) & (df.ip_cluster_x <= 8), 3,
                                                  np.where((df.ip_cluster_x >= 9) & (df.ip_cluster_x <= 11), 4,
                                                           np.where((df.ip_cluster_x >= 12) & ( df.ip_cluster_x <= 14), 5,  6)))))
    df['role_y'] = np.where((df.ip_cluster_y >= 0) & (df.ip_cluster_y <= 2), 1,
                                     np.where((df.ip_cluster_y >= 3) & (df.ip_cluster_y <= 5), 2,
                                              np.where( (df.ip_cluster_y >= 6) & (df.ip_cluster_y <= 8), 3,
                                                  np.where((df.ip_cluster_y >= 9) & ( df.ip_cluster_y <= 11), 4,
                                                           np.where((df.ip_cluster_y >= 12) & (df.ip_cluster_y <= 14), 5,  6)))))
    return df



def show_feature_importances(model, input_train):
    # Genereate and print feature scores
    feature_scores = pd.Series(model.feature_importances_, index=input_train.columns).sort_values(ascending=False)
    fig = px.bar(feature_scores, orientation='h')
    fig.update_layout(
        title='Feature Importances',
        showlegend=False,
    )
    fig.show()
    pyo.plot(fig)

def show_heat_map(features):
    # assume X is your feature matrix as a pandas DataFrame
    corr_matrix = features.corr()

    # plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    return corr_matrix

def prepare_set(transfer, teams, roles, performance_stats, ability_set):
    df_transfer_periods = handle_transfer_periods(transfer)
    tf_final = handle_transfer_data(df_transfer_periods, teams)
    players_and_cultures = find_players_and_countries(tf_final, df_transfer_periods, df_teams)
    df_v3 = df_players_filtered.merge(players_and_cultures, on='playerId', how='left')
    #df_v3 = df_v3.merge(df_act_suc, on='playerId')

    df_v3['cultures'] = df_v3.apply(lambda row: check_nan(row['countries'], row['passportArea_name']) , axis = 1)

    df_v3 = (df_v3[['playerId', 'shortName', 'birthDate', 'height', 'weight','birthArea_name', 'passportArea_name', 'position', 'currentTeamId', 'age', 'cultures']]).rename(columns= {'cultures': 'countries'})
    df_v4 = df_v3.merge(roles, on='playerId')
    df_v4 = df_v4.merge(performance_stats, on='playerId')
    df_v4 = df_v4.rename(columns = {'minutes': 'minutes_played_season'})
    #df_v4 = df_v4.merge(df_chem_power_and_strength[['playerId', 'chem_ability', 'aerial_strength', 'carry_strength','pressing_recovery_strength', 'defensive_duel_strength','dribbles_strength', 'ground_duel_strength', 'sliding_strength','deep_crossing_strength']], on='playerId')

    players_chemistry_1 = ability_set.merge(df_v4, left_on=['p1'], right_on=['playerId'])
    players_chemistry_t = players_chemistry_1.merge(df_v4, left_on=['p2'], right_on=['playerId'])
    players_chemistry_t = players_chemistry_t[(players_chemistry_t['ip_cluster_x'] != -1) & (players_chemistry_t['ip_cluster_y'] != -1)]
    set_1 = players_chemistry_t.drop_duplicates(subset=['p1' , 'p2', 'playerId_x', 'playerId_y'])

    with_indicators = create_indicators(set_1)
    prep = duplicate_and_order_data(with_indicators)
    prepped = pd.concat([prep, with_indicators])
    prepped = produce_overall_cluster(prepped)
    #chem_values = train_prepped['chemistry'].fillna(0)
    #df_prepared_v2 = df_prepared.merge(df_chem_power_and_strength, left_on = ['p1', 'seasonId'], right_on=['playerId', 'seasonId'], how='left')
    #df_prepared_v3 = df_prepared_v2.merge(df_chem_power_and_strength, left_on = ['p2', 'seasonId'], right_on=['playerId', 'seasonId'], how='left')
    # Define the custom binning function

    # Apply the custom binning function to create the 'chem_groups' column
    #df_prepared['chem_groups'] = chem_values.apply(custom_binning)

    pred_prep = prepped.drop(['p1', 'p2', 'teamId', 'seasonId', 'pos_group_x' , 'df_jdi90', 'df_joi90', 'pos_group_y', 'ip_cluster_y', 'ip_cluster_x'], axis=1)
    pred_prep['same_pos'] = np.where(pred_prep['role_x'] == pred_prep['role_y'], 1, 0)
    pred_prep['same_role'] = np.where(pred_prep['position_x'] == pred_prep['position_y'], 1, 0)
    pred_prep = pred_prep.drop(['role_x', 'role_y', 'position_x', 'position_y'], axis = 1)
    #df_for_pred['chem_groups'] = pd.qcut(df_for_pred['chemistry'], 3, labels=['Low', 'Medium', 'High']).astype('category').cat.codes
    pred_prep = pred_prep.fillna(0)

    pred_prep = pred_prep[['same_origin', 'same_country', 'played_in_same_country', 'height_y', 'weight_y', 'age_y', 'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y', 'height_x', 'weight_x', 'age_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x', 'same_pos', 'same_role', 'chemistry']]
    feature_columns = pred_prep.columns
    input_variables = pred_prep.columns[feature_columns != 'chemistry']
    input = pred_prep[input_variables]
    input_prepped = StandardScaler().fit_transform(input)
    target_prepped = pred_prep['chemistry']
    return input_prepped, target_prepped

input_train, target_train = prepare_set(df_transfer, df_teams, players_roles, df_performance_stats, training_set )
input_test, target_test = prepare_set(df_transfer, df_teams, players_roles, df_performance_stats, (test_set).rename(columns = {'new_player': 'p1', 'existing_player': 'p2'}) )

#feature_columns = pred_train.columns
#input_variables = pred_train.columns[feature_columns != 'chem_groups']
#input = pred_train[input_variables]
#input_train = StandardScaler().fit_transform(input)
#target_train = pred_train['chem_groups']


pred_train = pred_train[['same_origin', 'same_country', 'played_in_same_country', 'height_y', 'weight_y', 'age_y', 'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem', 'height_x', 'weight_x', 'age_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_ability_x', 'same_pos', 'same_role', 'chemistry']]
show_target_distribution(pred_train, 'chemistry')


#input_train, input_test, target_train, target_test = train_test_split(input_scaled, target, test_size=0.2, random_state=42)

correlations = show_heat_map(input)
#cluster_weights_dict = dict(zip(cluster_and_weights.ip_cluster, cluster_and_weights.percentage))
random_weighted_predictions = produce_random_number(list(cluster_and_weights.ip_cluster), list(cluster_and_weights.percentage), input_test.shape[0])
mode_genereated_predictions = pd.DataFrame([statistics.mode(target_train)] * len(target_test))
check_random_number_distribution(random_weighted_predictions)

# Calculate class weights
class_weights = target_train.value_counts(normalize=True)
check_dis(pred_train, 'chem_groups')


# Generate random predictions based on class weights
def random_guesses(class_weights, n, num_classes):
    values = list(class_weights.index)
    probs = list(class_weights.values)
    preds = np.zeros(n)
    for i in range(n):
        preds[i] = random.choices(values, weights=probs)[0]
    return preds.astype(int)

y_train_mean = np.mean(target_train)

# Generate predictions using the mean target value for the test set
y_pred_baseline_mean = np.full_like(target_test, y_train_mean)

# Calculate the R-squared score for the baseline model
r2_baseline_mean = r2_score(target_test, y_pred_baseline_mean)


# Calculate the mean and standard deviation of the target variable in the training set
y_train_mean = np.mean(target_train)
y_train_std = np.std(target_train)

# Generate random predictions using a normal distribution with the same mean and standard deviation
y_pred_baseline_weighted = np.random.normal(loc=y_train_mean, scale=y_train_std, size=len(target_test))

# Calculate the R-squared score for the baseline model
r2_baseline_weighted = r2_score(target_test, y_pred_baseline_weighted)

dummy_reg = DummyRegressor(strategy='mean')
# fit it on the training set
dummy_reg.fit(input_train, target_train)
# make predictions on the test set
dummy_pred = dummy_reg.predict(input_test)

# calculate root mean squared error
mse = mean_squared_error(target_test, dummy_pred)
rmse = np.sqrt(mse)
print("Dummy RMSE:", rmse, "Dummy r-squared:", r2_score(target_test, dummy_pred), sep=" ")
print("Mean RMSE:",r2_baseline_mean, "mean r-squared:", (np.sqrt(mean_squared_error(target_test, y_pred_baseline_weighted))), sep=" ")
print("Weigted guesses RMSE:", r2_baseline_weighted, "Weigted r-squared:", (mean_squared_error(target_test, y_pred_baseline_mean, squared=False)))


#Regression model
xgb_reg = xgb.XGBRegressor( n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_reg.fit(input_train, target_train)
xgb_preds_test = xgb_reg.predict(input_test)
xgb_preds_train = xgb_reg.predict(input_train)
rmse = np.sqrt(np.mean((target_test - xgb_preds_test)**2))
r2 = r2_score(target_test, xgb_preds_test)
print("Root Mean Squared Error:", rmse)
print("R-squared value:", r2)

# Perform cross-validation
scores = cross_val_score(xgb_reg, input_train, target_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)


show_feature_importances(xgb_reg, input_train )

#archieve with testing of code to tryout different models and hypertuning
'''
for i in np.linspace(0.005, 0.02, num=200):
    
    xgb_m = xgb.XGBClassifier(n_estimators=93, min_child_weight=8, max_depth=6,
                              learning_rate=0.0101, gamma=0.7, colsample_bytree=0.5,
                              )

    xgb_m.fit(input_train, target_train)
    xgb_preds_test = xgb_m.predict(input_test)
    xgb_preds_train = xgb_m.predict(input_train)
    print(i)
    evaluate_model_original(xgb_preds_test, xgb_preds_train, target_test, target_train, 'xgb model',i)

for i in [i/10000 for i in range(10, 21)] :


lgb_model = lgb.LGBMClassifier(learning_rate=0.04320707070707071, max_bin=256, n_estimators=200,
                               objective='multiclass',
                               max_depth=7, num_leaves=10, feature_fraction=0.7, min_data_in_leaf=90,
                               reg_alpha=0.7, reg_lambda=9.89999999999999)

lgb_model.fit(input_train, target_train)
lgb_preds_test = lgb_model.predict(input_test)
lgb_preds_train = lgb_model.predict(input_train)
evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test, target_train, 'lgb model')

for i in range(3,12, 1):
    rdf_model =RandomForestClassifier(criterion ='gini',
                                       n_estimators=100,
                                       min_samples_split=18,
                                       min_samples_leaf=12,
                                       max_features='sqrt',
                                       max_depth=9)
    rdf_model.fit(input_train, target_train)
    rdf_preds_test = rdf_model.predict(input_test)
    rdf_preds_train = rdf_model.predict(input_train)
    print(i)
    evaluate_model_original(rdf_preds_test, rdf_preds_train, target_test, target_train, 'rdf model')
for i in range(3,12):
    lgb_model = lgb.LGBMClassifier(learning_rate=0.04202020202020202, max_bin = 256, n_estimators=116, objective='multiclass',
                                   max_depth=8, num_leaves=10, feature_fraction =1, min_data_in_leaf = 50,
                                    reg_alpha=0.7 , reg_lambda=9)
    lgb_model.fit(input_train, target_train)
    lgb_preds_test = lgb_model.predict(input_test)
    lgb_preds_train = lgb_model.predict(input_train)
    print(i)
    evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test, target_train, 'lgb model')




# Print evaluation metrics
#print(f"Log loss: {log_loss_val}")
print(f"Accuracy score: {acc_score_val}")
print(f"F1 score: {f1_score_val}")

for i in range(2, 100, 2):
    fa = FactorAnalyzer(n_factors=45, method='ml', rotation='varimax')
    # Fit the factor analyzer to the standardized data
    fa.fit(input_scaled)
    # Get the factor scores for the standardized data
    X_factors = fa.transform(input_scaled)
    loadings = fa.loadings_
    X = input_scaled.dot(loadings)
    input_train_fa, input_test_fa, target_train_fa, target_test_fa = train_test_split(X, target, test_size=0.2, random_state=42)
    lgb_model = lgb.LGBMClassifier(learning_rate=0.04320707070707071, max_bin = 256, n_estimators=200, objective='multiclass',
                                   max_depth=7, num_leaves= i, feature_fraction =0.7, min_data_in_leaf = 90,
                                   reg_alpha=0.7, reg_lambda=9.89999999999999)
    lgb_model.fit(input_train_fa, target_train_fa)
    lgb_preds_test = lgb_model.predict(input_test_fa)
    lgb_preds_train = lgb_model.predict(input_train_fa)
    print(i)
    evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test_fa, target_train_fa, 'lgb model')

    print()
    lgb_preds_test = lgb_model.predict(input_test_fa)
    lgb_proba_test = lgb_model.predict_proba(input_test_fa)
    logloss = log_loss(target_test_fa, lgb_proba_test, labels=lgb_model.classes_)

input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(learning_rate=0.04320707070707071, max_bin=256, n_estimators=200,
                               objective='multiclass',
                               max_depth=7, num_leaves=10, feature_fraction=0.7, min_data_in_leaf=90,
                               reg_alpha=0.7, reg_lambda=9.89999999999999)
lgb_model.fit(input_train, target_train)
lgb_preds_test = lgb_model.predict(input_test)
lgb_preds_train = lgb_model.predict(input_train)
evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test, target_train, 'lgb model')

ovr_model = OneVsRestClassifier(LGBMClassifier(learning_rate=0.04303030303030303,max_depth=5, num_leaves=10, feature_fraction=0.7, min_data_in_leaf=62,
                                      reg_alpha=3,  min_child_samples = 100, reg_lambda=9.89999999999999, n_estimators=426))
ovr_model.fit(input_train, target_train)
xgb_preds_test = ovr_model.predict(input_test)
xgb_preds_train = ovr_model.predict(input_train)
evaluate_model_original(xgb_preds_test, xgb_preds_train, target_test, target_train, 'xgb model')

lgb_m = LGBMClassifier(learning_rate=0.0101,max_depth=5, num_leaves=10, feature_fraction=0.7, min_data_in_leaf=62,
                                      reg_alpha=3,  min_child_samples = 100, reg_lambda=9.89999999999999, n_estimators=426)


scores = cross_val_score(xgb_m, input_train, target_train, cv=5)

xgb_m = xgb.XGBClassifier(n_estimators=110, min_child_weight=8, max_depth=6,
                          learning_rate=0.0101020201, gamma=0.7, colsample_bytree=0.5,
                          )

xgb_m.fit(input_train, target_train)
xgb_preds_test = xgb_m.predict(input_test)
xgb_preds_train = xgb_m.predict(input_train)
evaluate_model_original(xgb_preds_test, xgb_preds_train, target_test, target_train, 'xgb model')
scores = cross_val_score(xgb_m, input_train, target_train, cv=5)


'''




