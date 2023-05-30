from datetime import datetime
from random import random

from sklearn.model_selection import cross_val_score
from chemistry.chemistry_helpers import *
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

#extract data
players_chemistry, players_roles, df_performance_stats, df_transfer, df_positions_and_formations, df_players, df_teams, df_area = data_for_prediction()

# method is responsible for ensuring cleaning of data related to players and transfer history
df_players_filtered, df_transfer = prepare_player_data_and_transfer_data(df_players, df_transfer, df_area)

#Extract chemistry ability features per pair
chem_ability_scores = compute_player_chemistry_ability(players_chemistry)

#Identify all new arrivals
new_arrivals = get_new_arrivals(df_transfer, chem_ability_scores)

#Here we ensure that no rows in the new arrivals will be present in waht will be used as the traning set
#This ensuring that new Arrivals will work as the test set.

#The following steps are taken
#   1. Merge the 'chem_ability_scores' and 'new_arrivals' DataFrames using a left join
#   2. This will include all rows from 'chem_ability_scores' and matching rows from 'new_arrivals'
#   3. Any rows in 'chem_ability_scores' that don't have a match in 'new_arrivals' will have missing values for 'new_arrivals' columns
train_full = pd.merge(chem_ability_scores, new_arrivals, how='left', indicator=True)

#   4. Filter the merged DataFrame to keep only the rows where the merge indicator is 'left_only'
#   5. These are the rows from 'chem_ability_scores' that don't have a match in 'new_arrivals'
filtered_df = train_full[train_full['_merge'] == 'left_only']

#   6. Drop the '_merge' column from the filtered DataFrame
training_set = filtered_df.drop('_merge', axis=1)


#Extract test and training set
input_train, target_train = prepare_set(df_transfer, df_teams, players_roles, df_performance_stats, training_set, df_players_filtered, df_positions_and_formations)
input_test, target_test = prepare_set(df_transfer, df_teams, players_roles, df_performance_stats, new_arrivals, df_players_filtered, df_positions_and_formations)



# Create an instance of the XGBRegressor model with specified hyperparameters
xgb_reg = xgb.XGBRegressor(learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995,
                           reg_lambda=0.44599999999999995, n_estimators=100, max_depth=11, min_child_weight=14, gamma=0,
                           subsample=0.44599999999999995, colsample_bytree=0.44599999999999995)

# Train the XGBRegressor model using the training data
xgb_reg.fit(input_train, target_train)

# Make predictions on the testing data using the trained model
xgb_preds_train = xgb_reg.predict(input_train)

# Calculate the root mean squared error (RMSE) between the predicted values and the actual values
rmse = np.sqrt(np.mean((target_train - xgb_preds_train) ** 2))

# Calculate the R-squared value between the predicted values and the actual values
r2 = r2_score(target_train, xgb_preds_train)

# Calculate the R-squared value and RMSE for the predictions on the training data
r2_train = r2_score(target_train, xgb_preds_train)
rmse_train = mean_squared_error(target_train, xgb_preds_train, squared=False)

# Explore results from R squared and root mean squared error
print("Train set - R-squared:", r2_train)
print("Train set - RMSE:", rmse_train)

# Perform three-fold cross-validation
scores_rmse_xgb = -cross_val_score(xgb_reg, input_test, target_test, cv=3, scoring='neg_root_mean_squared_error')
scores_r2_xgb = cross_val_score(xgb_reg, input_test, target_test, cv=3, scoring='r2')

rmse_mean_xgb = np.mean(scores_rmse_xgb)
r2_mean_xgb = np.mean(scores_r2_xgb)

# Print the evaluation metrics for the CatBoostRegressor model
print("RMSE:", rmse_mean_xgb)
print("R-squared value:", r2_mean_xgb)



# Create an instance of the LGBMRegressor model with specified hyperparameters
lgb_reg = lgb.LGBMRegressor(
    learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995, reg_lambda=0.44599999999999995, n_estimators=100,
    max_depth=6, num_leaves=20, feature_fraction=1, min_data_in_leaf=100
)

# Train the LGBMRegressor model using the training data
lgb_reg.fit(input_train, target_train)

# Make predictions on the training and testing data using the trained model
lgb_preds_train = lgb_reg.predict(input_train)
lgb_preds_test = lgb_reg.predict(input_test)

# Calculate the R-squared value and RMSE for the predictions on the testing data
r2_test = r2_score(target_test, lgb_preds_test)
rmse_test = mean_squared_error(target_test, lgb_preds_test, squared=False)

# Calculate the R-squared value and RMSE for the predictions on the training data
r2_train = r2_score(target_train, lgb_preds_train)
rmse_train = mean_squared_error(target_train, lgb_preds_train, squared=False)

# Explore results from R squared and root mean squared error
print("Train set - R-squared:", r2_train)
print("Train set - RMSE:", rmse_train)

# Perform three-fold cross-validation
scores_rmse_lgb = -cross_val_score(lgb_reg, input_test, target_test, cv=3, scoring='neg_root_mean_squared_error')
scores_r2_lgb = cross_val_score(lgb_reg, input_test, target_test, cv=3, scoring='r2')

# Calculate the mean and standard deviation of the RMSE scores
mean_rmse_lgb = np.mean(scores_rmse_lgb)
meand_r2_lgb = np.mean(scores_r2_lgb)


# Print the evaluation metrics for the CatBoostRegressor model
print("RMSE:", mean_rmse_lgb)
print("R-squared value:", meand_r2_lgb)



#Here we prepare the dataset for purposes of the Catboost regressor that handles categorical data seamlessly
input_train_cat = input_train
input_test_cat = input_test

#Type transformations
target_train_cat = target_train
target_test_cat = target_test
input_train_cat['same_origin'] = input_train_cat['same_origin'].astype('category')
input_train_cat['same_country'] = input_train_cat['same_country'].astype('category')
input_train_cat['played_in_same_country'] = input_train_cat['played_in_same_country'].astype('category')
input_train_cat['same_pos'] = input_train_cat['same_pos'].astype('category')
input_train_cat['same_role'] = input_train_cat['same_role'].astype('category')
input_test_cat['same_origin'] = input_test_cat['same_origin'].astype('category')
input_test_cat['same_country'] = input_test_cat['same_country'].astype('category')
input_test_cat['played_in_same_country'] = input_test_cat['played_in_same_country'].astype('category')
input_test_cat['same_pos'] = input_test_cat['same_pos'].astype('category')
input_test_cat['same_role'] = input_test_cat['same_role'].astype('category')

#Array of labels describing what features are categorical to be used for the Catboosregressor
cat_features = ['same_origin', 'same_country', 'played_in_same_country', 'same_pos', 'same_role']


# Create an instance of the CatBoostRegressor model with specified hyperparameters
cat_reg = CatBoostRegressor(
    iterations=264, depth=6, learning_rate=0.11000011, loss_function='RMSE',
    random_strength=0.1, bagging_temperature=1, border_count=507,
    subsample=0.44599999999999995, l2_leaf_reg=1, cat_features=cat_features
)

# Train the CatBoostRegressor model using the categorical training data and labels
cat_reg.fit(input_train_cat, target_train_cat, verbose=False)

# Make predictions on the categorical testing data using the trained model
y_pred_cat = cat_reg.predict(input_test_cat)

# Perform three-fold cross-validation
scores_rmse_cat = -cross_val_score(cat_reg, input_test_cat, target_test_cat, cv=3, scoring='neg_root_mean_squared_error')
scores_r2_cat = cross_val_score(cat_reg, input_test_cat, target_test_cat, cv=3, scoring='r2')

# Calculate the mean and standard deviation of the RMSE scores
mean_rmse_cat = np.mean(scores_rmse_cat)
meand_r2_cat = np.mean(scores_r2_cat)


# Print the evaluation metrics for the CatBoostRegressor model
print("RMSE:", mean_rmse_cat)
print("R-squared value:", meand_r2_cat)




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






#--------------------------------------------------------------------------- Archieved code - NOT COMMENTED--------------------------------------------------------
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



for i in range(1000, 2005, 10):
    num = i / 10000
    #Regression model
    xgb_reg = xgb.XGBRegressor(learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995, reg_lambda=0.44599999999999995,  n_estimators = 100, max_depth=11, min_child_weight=14 , gamma=0 , subsample=0.44599999999999995 , colsample_bytree=0.3 )
    xgb_reg.fit(input_train, target_train)
    xgb_preds_test = xgb_reg.predict(input_test)
    rmse = np.sqrt(np.mean((target_test - xgb_preds_test)**2))
    r2 = r2_score(target_test, xgb_preds_test)
    print(num)
    print("Root Mean Squared Error:", rmse)
    print("R-squared value:", r2)
    print()

results = []

for i in np.linspace(0.1, 1, 100):
    num = 0.05 + i * 0.0045
    xgb_reg = xgb.XGBRegressor(learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995, reg_lambda=0.44599999999999995,  n_estimators = 100, max_depth=11, min_child_weight=14 , gamma=0 , subsample=0.44599999999999995 , colsample_bytree= 0.44599999999999995 )
    xgb_reg.fit(input_train, target_train)
    xgb_preds_test = xgb_reg.predict(input_test)
    rmse = np.sqrt(np.mean((target_test - xgb_preds_test)**2))
    r2 = r2_score(target_test, xgb_preds_test)
    print("Best learning rate:", rmse)
    print("Best R-squared value:", r2)
    results.append((i, rmse, r2))

best_learning_rate = max(results, key=lambda x: x[2])[0]
best_r2 = max(results, key=lambda x: x[2])[2]
print("Best learning rate:", best_learning_rate)
print("Best R-squared value:", best_r2)

'''

#Code to produce Brondby specific set

'''
def prepare_set_for_experiment(transfer, teams, roles, performance_stats, ability_set, players_filtered, positions_formations):
    df_transfer_periods = handle_transfer_periods(transfer)
    tf_final = handle_transfer_data(df_transfer_periods, teams)
    players_and_cultures = find_players_and_countries(tf_final, df_transfer_periods, df_teams)
    df_v3 = players_filtered.merge(players_and_cultures, on='playerId', how='left')
    df_v3['cultures'] = df_v3.apply(lambda row: check_nan(row['countries'], row['passportArea_name']), axis=1)

    df_v3 = (df_v3[
        ['playerId', 'shortName', 'birthDate', 'birthArea_name', 'passportArea_name', 'position',
         'currentTeamId', 'age', 'cultures']]).rename(columns={'cultures': 'countries'})
    df_v4 = df_v3.merge(roles, on='playerId')
    df_v4 = df_v4.merge(performance_stats, on='playerId')
    df_v4 = df_v4.rename(columns={'minutes': 'minutes_played_season'})
    players_chemistry_1 = ability_set.merge(df_v4, left_on=['p1'], right_on=['playerId'])
    players_chemistry_t = players_chemistry_1.merge(df_v4, left_on=['p2'], right_on=['playerId'])
    players_chemistry_t = players_chemistry_t[
        (players_chemistry_t['ip_cluster_x'] != -1) & (players_chemistry_t['ip_cluster_y'] != -1)]
    set_1 = players_chemistry_t.drop_duplicates(subset=['p1', 'p2', 'teamId'])
    with_indicators = create_indicators(set_1)
    prepped = produce_overall_cluster(with_indicators)
    num_transfer_df = count_teams(transfer, players_filtered)
    num_transfer_df_1 = prepped.merge(num_transfer_df, left_on=['p1'], right_on=['playerId'])
    num_transfer_df_2 = num_transfer_df_1.merge(num_transfer_df, left_on=['p2'], right_on=['playerId'])
    positions_formations_1 = num_transfer_df_2.merge(positions_formations, left_on=['p1'], right_on=['playerId'])
    positions_formations_2 = positions_formations_1.merge(positions_formations, left_on=['p2'], right_on=['playerId'])
    positions_formations_2 = positions_formations_2.drop_duplicates(subset=['p1', 'p2', 'teamId'])
    prepped = positions_formations_2
    pred_prep = prepped.drop(['teamId', 'seasonId', 'pos_group_x', 'ip_cluster_y', 'ip_cluster_x'], axis=1)
    pred_prep['same_pos'] = np.where(pred_prep['role_x'] == pred_prep['role_y'], 1, 0)
    pred_prep['same_role'] = np.where(pred_prep['position_x'] == pred_prep['position_y'], 1, 0)
    pred_prep = pred_prep.drop(['role_x', 'role_y', 'position_x', 'position_y'], axis=1)
    # df_for_pred['chem_groups'] = pd.qcut(df_for_pred['chemistry'], 3, labels=['Low', 'Medium', 'High']).astype('category').cat.codes
    pred_prep = pred_prep.fillna(0)
    pred_prep = pred_prep[[ 'p1','p2', 'same_origin', 'same_country', 'played_in_same_country', 'age_y',
                           'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y',
                           'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y',
                           'age_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x',
                           'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x',
                           'same_pos', 'same_role', 'num_transfer_x', 'num_transfer_y', 'num_positions_x',
                           'num_positions_y', 'num_formations_x', 'num_formations_y', 'chemistry']]
    feature_columns = pred_prep.columns
    input_variables = pred_prep.columns[feature_columns != 'chemistry']
    input = pred_prep[input_variables]
    target_prepped = pred_prep['chemistry']
    return input, target_prepped, df_v3

'''

#Experimental efforts with prediction intervals
'''
lower = lgb.LGBMRegressor(
    learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995, reg_lambda=0.44599999999999995, n_estimators=100,
    max_depth=6,
    num_leaves=20, feature_fraction=1, min_data_in_leaf=100,
    objective = 'quantile', alpha = 1 - 0.95
)
upper = lgb.LGBMRegressor(
    learning_rate=0.11000000000000001, reg_alpha=0.31999999999999995, reg_lambda=0.44599999999999995, n_estimators=100,
    max_depth=6,
    num_leaves=20, feature_fraction=1, min_data_in_leaf=100,
    objective = 'quantile', alpha = 0.95
)




upper.fit(input_train, target_train)
xgb_preds_test_upper = upper.predict(input_test)
xgb_preds_train_upper = upper.predict(input_train)

lower.fit(input_train, target_train)
xgb_preds_test_lower = lower.predict(input_test)
xgb_preds_train_lower = lower.predict(input_train)



'''


#Archieved used for purposes relate to validatoin with Brødnby
'''
input_train_ex, target_train_ex, countries_train = prepare_set_for_experiment(df_transfer, df_teams, players_roles, df_performance_stats, training_set, df_players_filtered, df_positions_and_formations)
input_test_ex, target_test_ex, countries_test = prepare_set_for_experiment(df_transfer, df_teams, players_roles, df_performance_stats, new_arrivals, df_players_filtered, df_positions_and_formations)

brondby_arrivals = [483796, 562633, 536812, 434296, 334043, 607268, 652912, 571292, 494548, 332593, 562633, 494548]
brondby_existing = [207480, 56079, 169212, 257943, 399566, 676912, 412012, 544937, 69374, 356393, 244135, 513075, 510447, 334043, 363625]
R = 69374

full_set = pd.concat([input_train_ex, input_test_ex])
p_f_p1_e = ((full_set[full_set['p1'].isin(brondby_existing)])[['p1',
       'height_x', 'weight_x', 'age_x', 'minutes_played_season_x',
       'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x',
       'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x', 'num_transfer_x', 'num_positions_x', 'num_formations_x']]).rename(columns = {'p1': 'playerId'})
p_f_p1_e.columns = p_f_p1_e.columns.str.replace('_x', '')



p_f_p2_e = ((full_set[full_set['p2'].isin(brondby_existing)])[['p2',
       'height_y', 'weight_y', 'age_y', 'minutes_played_season_y',
       'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
       'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y', 'num_transfer_y', 'num_positions_y', 'num_formations_y']]).rename(columns = {'p2': 'playerId'})
p_f_p2_e.columns = p_f_p1_e.columns.str.replace('_y', '')
brondby_players = ((pd.concat([p_f_p1_e, p_f_p2_e])).drop_duplicates(subset=('playerId')).reset_index(drop=True)).merge(countries_train[['playerId', 'shortName', 'birthArea_name', 'passportArea_name', 'position', 'currentTeamId', 'countries']], on = 'playerId', how='left')
brondby_players.playerId
p_f_p1_n = ((full_set[full_set['p1'].isin(brondby_arrivals)])[['p1',
       'height_x', 'weight_x', 'age_x', 'minutes_played_season_x',
       'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x',
       'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_coef_x', 'num_transfer_x', 'num_positions_x', 'num_formations_x']]).rename(columns = {'p1': 'playerId'})
p_f_p1_n.columns = p_f_p1_e.columns.str.replace('_x', '')

p_f_p2_n = ((full_set[full_set['p2'].isin(brondby_arrivals)])[['p2',
       'height_y', 'weight_y', 'age_y', 'minutes_played_season_y',
       'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y',
       'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem_coef_y', 'num_transfer_y', 'num_positions_y', 'num_formations_y']]).rename(columns = {'p2': 'playerId'})
p_f_p2_n.columns = p_f_p2_e.columns.str.replace('_y', '')
potential_talent = ((pd.concat([p_f_p1_n, p_f_p2_n])).drop_duplicates(subset=('playerId')).reset_index(drop=True)).merge(countries_test[['playerId', 'shortName', 'birthArea_name', 'passportArea_name', 'position', 'currentTeamId', 'countries']], on='playerId', how = 'left')

'''

'''
def calculate_confidence_interval(predictions, alpha=0.05):
    lower_bound = np.percentile(predictions, (alpha / 2) * 100)
    upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100)
    return lower_bound, upper_bound
'''


'''
predictions = find_potential_fits(brondby_players, potential_talent, players_roles, lgb_reg, upper, lower)
predictions = predictions.drop_duplicates(subset=('shortName_y', 'shortName_x'))
pairwise_predictions = predictions[['shortName_x', 'shortName_y', 'predicted_chem']]
general_ability = predictions[['shortName_x', 'predicted_chem','lower_bound', 'upper_bound']]
general_ability_v2 = general_ability.groupby(['shortName_x'], as_index = False).agg({'predicted_chem': 'mean', 'lower_bound':'mean',  'upper_bound': 'mean'})

predictions['chemistry'] =  predictions.apply(lambda row: row.predicted_chem[0], axis = 1)
lower_b = predictions[['shortName_x', 'shortName_y', 'lower_bound']]
upper_b = predictions[['shortName_x', 'shortName_y', 'upper_bound']]
lower_b['chemistry'] = lower_b['lower_bound'].apply(lambda row: row[0])
lower_b = lower_b.drop('lower_bound', axis=1)
upper_b['chemistry'] =  (upper_b.apply(lambda row: row.upper_bound[0], axis = 1))
upper_b = upper_b.drop('upper_bound', axis=1)

'''

'''



general_ability_v2.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/silke_predicted_ability.csv", index=False, decimal = (','), sep=(';'))
pairwise_predictions.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/silke_pairwise_predicted.csv", index=False, decimal = (','), sep=(';'))



#feature_columns = pred_train.columns
#input_variables = pred_train.columns[feature_columns != 'chem_groups']
#input = pred_train[input_variables]
#input_train = StandardScaler().fit_transform(input)
#target_train = pred_train['chem_groups']


pred_train = pred_train[['same_origin', 'same_country', 'played_in_same_country', 'height_y', 'weight_y', 'age_y', 'minutes_played_season_y', 'match appearances_y', 'zone_1_pl_y', 'zone_2_pl_y', 'zone_3_pl_y', 'zone_4_pl_y', 'zone_5_pl_y', 'zone_6_pl_y', 'chem', 'height_x', 'weight_x', 'age_x', 'minutes_played_season_x', 'match appearances_x', 'zone_1_pl_x', 'zone_2_pl_x', 'zone_3_pl_x', 'zone_4_pl_x', 'zone_5_pl_x', 'zone_6_pl_x', 'chem_ability_x', 'same_pos', 'same_role', 'chemistry']]
show_target_distribution(pred_train, 'chemistry')


#input_train, input_test, target_train, target_test = train_test_split(input_scaled, target, test_size=0.2, random_state=42)

#cluster_weights_dict = dict(zip(cluster_and_weights.ip_cluster, cluster_and_weights.percentage))
random_weighted_predictions = produce_random_number(list(cluster_and_weights.ip_cluster), list(cluster_and_weights.percentage), input_test.shape[0])
mode_genereated_predictions = pd.DataFrame([statistics.mode(target_train)] * len(target_test))
check_random_number_distribution(random_weighted_predictions)

# Calculate class weights
class_weights = target_train.value_counts(normalize=True)
check_dis(pred_train, 'chem_groups')

'''