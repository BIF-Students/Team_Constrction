from itertools import cycle
import pandas as pd
from lightgbm import LGBMClassifier
from numpy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, \
    mean_squared_error, r2_score, roc_curve, auc, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import plotly.express as px
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from helpers.helperFunctions import *
from helpers.student_bif_code import load_db_to_pd
from helpers.visualizations import *
import random
import statistics
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
import plotly.offline as pyo
from sklearn.dummy import DummyClassifier
import sys
sys.path.append('C:/Users/jhs/factor_analyzer/factor_analyzer-main')
sys.path.append('C:/Users/jhs/umap/\pynndescent')
sys.path.append('C:/Users/jhs/\pynndescent/umap-master')
from factor_analyzer import FactorAnalyzer
import umap
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import log_loss

df_players = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_players.csv")
df_players = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players]", db_name='Scouting_Raw')

def show_target_distribution(df, label):
    # create a figure and axis
    fig, ax = plt.subplots()

    # plot a histogram of the column values
    ax.hist(df[label], bins=50)

    # set the axis labels
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    # show the plot
    plt.show()

def check_random_number_distribution(vals):
    sum = vals.value_counts().sum()
    counts = pd.DataFrame(vals.value_counts())
    counts['percentage'] = counts.iloc[:, 0].apply(lambda row: (row / sum) * 100)
    counts = counts.rename(columns={counts.columns[0]: "counts"})
    counts = counts.rename_axis('clusters').reset_index()
    plt.bar(counts['clusters'], counts['percentage'])
    plt.show()

def number_distribution(vals):
    sum = vals.value_counts().sum()
    counts = pd.DataFrame(vals.value_counts())
    counts['percentage'] = counts.iloc[:, 0].apply(lambda row: (row / sum) * 100)
    counts = counts.rename(columns={counts.columns[0]: "counts"})
    counts = counts.rename_axis('clusters').reset_index()
    plt.bar(counts['clusters'], counts['percentage'])
    plt.show()

def evaluate_model_t(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean squared error: {mse:.4f}')
    print(f'Root mean squared error: {rmse:.4f}')
    print(f'R-squared: {r2:.4f}')


def produce_random_number(values, weights, length):
    predictions = random.choices(values, weights=(weights), k=length)
    return pd.DataFrame(predictions)


def evaluate_model_original(input_train, target_train, input_test, target_test, model):
    scoring = {'f1': 'f1_weighted', 'loss': 'neg_log_loss', 'accuracy': 'accuracy',
               'cohen_kappa': make_scorer(cohen_kappa_score)}
    scores = cross_validate(model, input_train, target_train, cv=5, scoring=scoring)  # Perform 5-fold cross-validation

    avg_f1_score = np.mean(scores['test_f1'])
    avg_loss_score = -np.mean(scores['test_loss'])
    avg_accuracy_score = np.mean(scores['test_accuracy'])
    avg_cohen_kappa_score = np.mean(scores['test_cohen_kappa'])

    print('Average F1 Score: {:.2f}'.format(avg_f1_score))
    print('Average Loss Score: {:.2f}'.format(avg_loss_score))
    print('Average Accuracy Score: {:.2f}'.format(avg_accuracy_score))
    print('Average Cohen Kappa Score: {:.2f}'.format(avg_cohen_kappa_score))

    model.fit(input_train, target_train)
    preds_test = model.predict(input_test)

    test_accuracy = accuracy_score(target_test, preds_test)
    print('Test Accuracy Score: {:.4f}'.format(test_accuracy))




'''def evaluate_model_original(target_pred_test, target_preds_train, target_test, target_train, modelType):
    f1 = f1_score(target_test, target_pred_test, average='weighted')
    cohen_kappa = cohen_kappa_score(target_test, target_pred_test)
    print('Model accuracy score train ' + modelType + ': {0:0.4f}'.format(accuracy_score(target_train, target_preds_train)))
    print('Model accuracy score ' + modelType + ': {0:0.4f}'.format(accuracy_score(target_test, target_pred_test)))
    print('F1 Score: ', "%.2f" % (f1 * 100))
    print('Cohen Kappa: ', "%.2f" % cohen_kappa)
    # print(classification_report(target_test, tar))
'''

def heatmap_probability_inspection(target_pred, target_test, title):
    matrix = confusion_matrix(target_test, target_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fig = px.imshow(matrix, text_auto=True, aspect="auto")
    fig.update_layout(title=title)
    fig.show()


def display_results_classification_report(target_pred, target_test):
    report = classification_report(target_test, target_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.show()

# Helper function
def show_feature_importances(model, input_train, input_test, target_test):
    # Genereate and print feature scores
    feature_scores = pd.Series(model.feature_importances_, index=input_train.columns).sort_values(ascending=False)
    fig = px.bar(feature_scores, orientation='h')
    fig.update_layout(
        title='Feature Importances',
        showlegend=False,
    )
    fig.show()
    pyo.plot(fig)

data  = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/brondbyProjekter/data/players_clusters.csv",
                                sep=",",
                                encoding='unicode_escape')
event_data = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/brondbyProjekter/data/events_clean.csv",
                                sep=",",
                                encoding='unicode_escape')

df_data_with_clusters = data.merge(event_data, on = ['playerId', 'seasonId'])
#df_data_with_clusters = df_data_with_clusters[df_data_with_clusters['187483']]

#for_prediction =df_data_with_clusters[(df_data_with_clusters['playerId'].isin(bif_players)) ]

#for_prediction = for_prediction[(for_prediction['seasonId'] == 187483 )]

df =df_data_with_clusters.iloc[:, 3:50]
df = df.drop(['x','y'], axis=1)
df = df[df['ip_cluster'] !=-1 ]


show_target_distribution(df, 'ip_cluster')

feature_columns = df.columns
input_variables = df.columns[feature_columns != 'ip_cluster']
input = df[input_variables]
target = df['ip_cluster']

input_scaled = StandardScaler().fit_transform(input)

input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.33, random_state=42)

# Calculate class weights
class_weights = target_train.value_counts(normalize=True)

# Generate random predictions based on class weights
def random_guesses(class_weights, n, num_classes):
    values = list(class_weights.index)
    probs = list(class_weights.values)
    preds = np.zeros(n)
    for i in range(n):
        preds[i] = random.choices(values, weights=probs)[0]
    return preds.astype(int)

# Make random predictions on test set
num_classes = len(target.unique())
test_preds = random_guesses(class_weights, len(target_test), num_classes)
list(range(num_classes))
log_loss_val = log_loss(target_test, test_preds, labels=list(range(num_classes)))

# Calculate evaluation metrics
log_loss_val = log_loss(target_test, test_preds)
acc_score_val = accuracy_score(target_test, test_preds)
f1_score_val = f1_score(target_test, test_preds, average='macro')

# Print evaluation metrics
#print(f"Log loss: {log_loss_val}")
print(f"Accuracy score: {acc_score_val}")
print(f"F1 score: {f1_score_val}")



# Shows roc curve for each class and labels the class and teh auc score


# Players to try and predict
# 38021 De buyne
# 3359 Messi
# 217031 PArtey


# Create models

rdf_model = RandomForestClassifier(criterion ='gini', n_estimators=100, min_samples_split=2, min_samples_leaf=7, max_features='sqrt',max_depth=5)
rdf_model.fit(input_train, target_train)
evaluate_model_original(input_train, target_train, input_test, target_test, rdf_model)
for i in range(2,5):
    num = i/10
    xgb_model = xgb.XGBClassifier(learning_rate=0.004555, reg_alpha=1, reg_lambda=1, objective= 'multi:softprob',  n_estimators = 500, max_depth=5, min_child_weight=2 , gamma=0 , subsample=0.2 , colsample_bytree=0.5 )
    xgb_model.fit(input_train, target_train)
    #xgb_preds_test = xgb_model.predict(input_test)
    #xgb_preds_train = xgb_model.predict(input_train)
    evaluate_model_original(input_train, target_train, input_test, target_test, xgb_model)


lgb_model = lgb.LGBMClassifier(learning_rate=0.045001 , n_estimators=110, objective='multiclass',
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=10 )
lgb_model.fit(input_train, target_train)
lgb_preds_test = lgb_model.predict(input_test)
lgb_preds_train = lgb_model.predict(input_train)
evaluate_model_original(input_train, target_train, input_test, target_test, lgb_model)


ovr_lgb = OneVsRestClassifier(LGBMClassifier(learning_rate=0.045001 , n_estimators=110,
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=10) )
ovr_lgb.fit(input_train, target_train)
ovr_lgb_preds_test = ovr_lgb.predict(input_test)
ovr_lgb_preds_train = ovr_lgb.predict(input_train)
evaluate_model_original(input_train, target_train, input_test, target_test, ovr_lgb)

ovr_xgb = OneVsRestClassifier(xgb.XGBClassifier(learning_rate=0.045001 , n_estimators=110,
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=10 ))
ovr_xgb.fit(input_train, target_train)
ovr_xgb_preds_test = ovr_xgb.predict(input_test)
ovr_xgb_preds_train = ovr_xgb.predict(input_train)
evaluate_model_original(ovr_xgb_preds_test, ovr_xgb_preds_train, target_test, target_train, 'ovr xgb model')

#fa analysis-----------------------------------------
'''
fa = FactorAnalyzer(n_factors=35, method='ml', rotation='varimax')
# Fit the factor analyzer to the standardized data
fa.fit(input_scaled)

# Get the factor scores for the standardized data
X_factors = fa.transform(input_scaled)
loadings = fa.loadings_
X = input_scaled.dot(loadings)
input_train_fa, input_test_fa, target_train_fa, target_test_fa = train_test_split(X, target, test_size=0.2, random_state=42)

lgb_model_fa = lgb.LGBMClassifier(learning_rate=0.04320707070707071 , n_estimators=116, objective='multiclass',
                               max_depth=7, num_leaves= 10, feature_fraction =1, min_data_in_leaf = 100,
                                reg_alpha=0.7 , reg_lambda=9.89999999999999)
lgb_model_fa.fit(input_train, target_train)
lgb_preds_test = lgb_model_fa.predict(input_test)
lgb_preds_train = lgb_model_fa.predict(input_train)
evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test, target_train, 'lgb model' )
lgb_preds_test = lgb_model_fa.predict(input_test)
lgb_proba_test = lgb_model_fa.predict_proba(input_test)
logloss_lgb = log_loss(target_test, lgb_proba_test, labels=lgb_model_fa.classes_)
scores_lgb = cross_val_score(lgb_model, input_train, target_train, cv=5)
print('lgb: ', scores_lgb)
print( 'log_loss lgb:', logloss_lgb)

xgb_model_fa = xgb.XGBClassifier(learning_rate=0.04320707070707071, reg_alpha=1, reg_lambda=1, objective= 'multi:softprob',  n_estimators = 98, max_depth=7, min_child_weight=6 , gamma=0 , subsample=0.3 , colsample_bytree=0.6 )
xgb_model_fa.fit(input_train, target_train)
xgb_preds_test = xgb_model.predict(input_test)
xgb_preds_train = xgb_model.predict(input_train)
evaluate_model_original(xgb_preds_test, xgb_preds_train, target_test, target_train, 'xgb model')
xgb_preds_test = xgb_model_fa.predict(input_test)
xgb_proba_test = xgb_model_fa.predict_proba(input_test)
logloss_xgb = log_loss(target_test, xgb_proba_test, labels=xgb_model_fa.classes_)
scores_xgb = cross_val_score(xgb_model_fa, input_train, target_train, cv=5)
print('xgb: ', scores_xgb)
print('log_loss xgb: ', logloss_xgb)


'''

p1 = for_prediction.iloc[[2]]
p2 = for_prediction.iloc[[8]]

df_bif_2 = p2.iloc[:, 3:50]
df_bif_2 = df_bif_2.drop(['seasonId_y', 'x', 'y'], axis=1)
feature_columns_bif_v = df_bif_2.columns
input_variables_bif_v = feature_columns_bif_v[feature_columns_bif_v != 'ip_cluster']
input_bif_v2 = p2[input_variables_bif_v]
xgb_preds_bif_v2 = xgb_model_fa.predict_proba(input_bif_v2)
top3_indices = np.argsort(xgb_preds_bif_v2, axis=1)[:,::-1][:,:3]


results = []
for i in range(len(for_prediction)):
    print(p1.columns)
    p1 = for_prediction.iloc[[i]]
    id = p1['playerId'].values[0]
    pos_group = p1['pos_group'].values[0]
    ip_cluster = p1['ip_cluster'].values[0]
    df_bif_2 = p1.iloc[:, 3:50]
    df_bif_2 = df_bif_2.drop(['x', 'y'], axis=1)
    feature_columns_bif_v = df_bif_2.columns
    input_variables_bif_v = feature_columns_bif_v[feature_columns_bif_v != 'ip_cluster']
    input_bif_v2 = p1[input_variables_bif_v]
    xgb_preds_bif_v3 = ovr_lgb.predict_proba(input_bif_v2)
    top3_indices = np.argsort(xgb_preds_bif_v3, axis=1)[:, ::-1][:, :3]
    top3_clusters = top3_indices[0]
    top3_probs = xgb_preds_bif_v3[0][top3_clusters]
    result = (id, pos_group, ip_cluster, top3_clusters[0], top3_probs[0], top3_clusters[1], top3_probs[1], top3_clusters[2], top3_probs[2])
    results.append(result)

results_df = pd.DataFrame(results, columns=['playerId', 'pos_group', 'ip_cluster', 'top_cluster1', 'top_prob1', 'top_cluster2', 'top_prob2', 'top_cluster3', 'top_prob3'])
results_df = results_df.merge(df_players[['playerId', 'shortName']], on = 'playerId')
results_df = results_df.loc[:, ['playerId', 'shortName', 'pos_group', 'ip_cluster', 'top_cluster1', 'top_cluster2', 'top_cluster3', 'top_prob1', 'top_prob2', 'top_prob3']]

earlier_results = pd.read_csv('C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/clusters_and_chem_AVG.csv', sep=(';'))

gg = results_df.merge(earlier_results[['playerId', 'chemistry']], how='left')


rdf_model_fa = RandomForestClassifier(criterion ='gini', n_estimators=100, min_samples_split=2, min_samples_leaf=7, max_features='sqrt',max_depth=10)
rdf_model_fa.fit(input_train_fa, target_train_fa)
rdf_preds_test = rdf_model_fa.predict(input_test_fa)
rdf_preds_train = rdf_model_fa.predict(input_train_fa)
evaluate_model_original(rdf_preds_test, rdf_preds_train, target_test_fa, target_train_fa, 'rdf model')
rdf_preds_test = rdf_model_fa.predict(input_test_fa)
rdf_proba_test = rdf_model_fa.predict_proba(input_test_fa)
logloss_rdf = log_loss(target_test_fa, rdf_proba_test, labels=rdf_model_fa.classes_)
scores_rdf = cross_val_score(rdf_model_fa, input_train_fa, target_train_fa, cv=5)
print('rdf: ', scores_rdf)
print('rdf loss: ', logloss_rdf)

ovr_lgb = OneVsRestClassifier(LGBMClassifier(learning_rate=0.04320707070707071, max_bin=256, n_estimators=200,
                               max_depth=7, num_leaves=10, feature_fraction=0.7, min_data_in_leaf=90,
                               reg_alpha=0.7, reg_lambda=9.89999999999999))
ovr_lgb.fit(input_train, target_train)
ovr_lgb_preds_test = ovr_lgb.predict(input_test)
ovr_lgb_preds_train = ovr_lgb.predict(input_train)
evaluate_model_original(ovr_lgb_preds_test, ovr_lgb_preds_train, target_test, target_train, 'ovr_lgb model')
ovr_lgb_preds_test = ovr_lgb.predict(input_test)
ovr_lgb_proba_test = ovr_lgb.predict_proba(input_test)
logloss_ovr_lgb = log_loss(target_test, ovr_lgb_proba_test, labels=ovr_lgb.classes_)
scores_ovr_lgb = cross_val_score(ovr_lgb, input_train, target_train, cv=5)
print('ovr_lgb loss: ', logloss_ovr_lgb)

def plot_multiclass_roc(model, X_test, y_test, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict_proba(X_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for a multi-class classifier')
    plt.legend(loc="lower right")
    plt.show()
target_test_fa_np = pd.get_dummies(target_test_fa).to_numpy()
plot_multiclass_roc(lgb_model_fa, input_test_fa, target_test_fa_np, n_classes=18)


#umap analysis----------------
# Use UMAP to create features
reducer = umap.UMAP(n_components=20, random_state=42)
X_umap = reducer.fit_transform(input_scaled)
for i in range(1,40):
    print(i)
    reducer = umap.UMAP(n_components=i, random_state=42)
    X_umap = reducer.fit_transform(input_scaled)
    input_train_u, input_test_u, target_train_u, target_test_u = train_test_split(X_umap, target,test_size=0.2,random_state=42)
    lgb_model_u = lgb.LGBMClassifier(learning_rate=0.04320707070707071 , n_estimators=116, objective='multiclass',
                               max_depth=7, num_leaves= 10, feature_fraction =1, min_data_in_leaf = 100,
                                reg_alpha=0.7 , reg_lambda=9.89999999999999 )
    lgb_model_u.fit(input_train_u, target_train_u)
    lgb_preds_test = lgb_model_u.predict(input_test_u)
    lgb_preds_train = lgb_model_u.predict(input_train_u)
    evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test_u, target_train_u, 'lgb model')


heatmap_probability_inspection(rdf_preds_test, target_test, 'rdf model')
heatmap_probability_inspection(lgb_preds_test, target_test, 'lgb model')


show_feature_importances(rdf_model, input_train, input_test, target_test)
show_feature_importances(xgb_model, input_train, input_test, target_test)
show_feature_importances(lgb_model, input_train, input_test, target_test)


def optimize_for_rdf(input, target):
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(0, 20, num = 1)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
    rfc = RandomForestClassifier()
    rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)
    rfc_random.fit(input, target)
    return rfc_random.best_params_

def optimize_for_lgb(input_train, target_train):
    lgb_model = lgb.LGBMClassifier()

    param_distributions = {
        'learning_rate': np.arange(0.01, 0.1, 0.01),
        'n_estimators': np.arange(50, 200, 10),
        'max_depth': np.arange(3, 10),
        'num_leaves': np.arange(10, 100, 10),
        'min_child_samples': np.arange(5, 50, 5),
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1]
    }

    lgb_random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    lgb_random_search.fit(input, target)
    return lgb_random_search.best_params_

def optimize_for_xgb(input_train, target_train):
    xgb_model = xgb.XGBClassifier()

    param_distributions = {
        'learning_rate': np.arange(0.01, 0.1, 0.01),
        'n_estimators': np.arange(50, 200, 50),
        'max_depth': np.arange(3, 10),
        'min_child_weight': np.arange(1, 10, 1),
        'gamma': [0, 0.1, 0.5, 1],
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    }
    # Define the Randomized Grid Search
    xgb_random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # Perform the Randomized Grid Search
    xgb_random_search.fit(input_train, target_train)

    # Print the best hyperparameters and evaluation metrics
    return xgb_random_search.best_params_


params_rdf = optimize_for_rdf(input_train, target_train)
params_xgb = optimize_for_xgb(input_train, target_train)
params_lgb = optimize_for_lgb(input_train, target_train)


#Baseline models
# Calculate class frequencies
class_frequencies = target_train.value_counts(normalize=True).to_dict()

# Create the baseline model using the most frequent class as the strategy
dummy = DummyClassifier(strategy='constant', constant=max(class_frequencies, key=class_frequencies.get))

# Fit the model on the training data
dummy.fit(input_train, target_train)

# Print the results
print('F1 score: ', f1)
print('Accuracy: ', accuracy)
print('Cohen kappa: ', kappa)
print('Log loss: ', logloss)

# Compute class weights based on the distribution of the target variable
class_weights = {c: len(target_train) / (len(np.where(target_train == c)[0]) * len(np.unique(target_train))) for c in np.unique(target_train)}

# Create a dummy classifier that makes weighted guesses based on the class distribution
dummy = DummyClassifier(strategy='constant', constant=max(class_weights, key=class_weights.get))

# Fit the dummy classifier on the training set
dummy.fit(input_train, target_train)
evaluate_model_original(input_train, target_train, input_test, target_test, dummy)


#display_target_counts(data)
results_df.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/brondby_clusters.csv", decimal=',', sep=(';'), index=False)
gg.to_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/clusters_and_chem_AVG.csv", decimal=',', sep=(';'), index=False)


'''
# Load dataset
DEF = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/DEF.csv', sep=",", encoding='unicode_escape')
MID = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/MID.csv', sep=",", encoding='unicode_escape')
WIDE = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/WIDE.csv', sep=",", encoding='unicode_escape')
ATT = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/ATT.csv', sep=",", encoding='unicode_escape')
raw = pd.read_csv("C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv", sep = ",", encoding='unicode_escape')
#Merge clusters with features
DEF_v2 = pd.merge(DEF, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
MID_v2 = pd.merge(MID, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
WIDE_v2 = pd.merge(WIDE, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
ATT_v2 = pd.merge(ATT, raw, on=['playerId','seasonId', 'map_group', 'pos_group'])
#removal of redundant columns
DEF_v3 = DEF_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
MID_v3 = MID_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
ATT_v3 = ATT_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
WIDE_v3 = WIDE_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
# Extract input and target values for each pos_gropu
feature_columns = DEF_v3.columns
input_variables = DEF_v3.columns[feature_columns != 'cluster']
# Def
def_input = DEF_v3[input_variables]
def_target = DEF_v3['cluster']
# mid
mid_input = MID_v3[input_variables]
mid_target = MID_v3['cluster']
# wing
wide_input = WIDE_v3[input_variables]
wide_target = WIDE_v3['cluster']
# att
att_input = ATT_v3[input_variables]
att_target = ATT_v3['cluster']
# Split of datasets for each position
def_input_train, def_input_test, def_target_train, def_target_test = train_test_split(def_input, def_target,
                                                                     test_size=0.33, random_state=42)
mid_input_train, mid_input_test, mid_target_train, mid_target_test = train_test_split(mid_input, mid_target,
                                                                     test_size=0.33, random_state=42)
wide_input_train, wide_input_test, wide_target_train, wide_target_test = train_test_split(wide_input, wide_target,
                                                                         test_size=0.33, random_state=42)
att_input_train, att_input_test, att_target_train, att_target_test = train_test_split(att_input, att_target,

        #MID
perform_regression(model, mid_input_train, mid_input_test, mid_target_train, mid_target_test, "MID")
#WIDE
perform_regression(model, wide_input_train, wide_input_test, wide_target_train, wide_target_test, "WIDE")
#ATT
perform_regression(model, att_input_train, att_input_test, att_target_train, att_target_test, "ATT")
                                                             test_size=0.33, random_state=42)
                                                             
                                                             

reg_alpha = 0.1
while reg_alpha <= 10:
    # Increase reg_alpha with a small step
    reg_alpha += 0.1
    print(reg_alpha)
    lgb_model_fa = lgb.LGBMClassifier(learning_rate=0.04320707070707071, n_estimators=116, objective='multiclass',
                                      max_depth=7, num_leaves=10, feature_fraction=1, min_data_in_leaf=100,
                                      reg_alpha=0.7, reg_lambda=reg_alpha)
    lgb_model_fa.fit(input_train_fa, target_train_fa)
    lgb_preds_test = lgb_model_fa.predict(input_test_fa)
    lgb_preds_train = lgb_model_fa.predict(input_train_fa)
    evaluate_model_t(lgb_preds_test, lgb_preds_train, target_test_fa, target_train_fa, 'lgb model', reg_alpha)


for i in range(80, 110):
    lgb_model_fa = lgb.LGBMClassifier(learning_rate=0.04320707070707071 , n_estimators=101, objective='multiclass',
                               max_depth=7, num_leaves= 10, feature_fraction =1, min_data_in_leaf = 100,
                                reg_alpha=0.7 , reg_lambda=9.89999999999999, max_bin = 256)
    lgb_model_fa.fit(input_train_fa, target_train_fa)
    lgb_preds_test = lgb_model_fa.predict(input_test_fa)
    lgb_preds_train = lgb_model_fa.predict(input_train_fa)
    evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test_fa, target_train_fa, 'lgb model')
#--------------------------------------------------------------------------------------------------------
# OneVsRest model
ovr_model = OneVsRestClassifier(LGBMClassifier(learning_rate=0.04320707070707071, max_depth=6, num_leaves=10, feature_fraction=0.7,
                   min_data_in_leaf=62,
                   reg_alpha=3, reg_lambda=9.89999999999999, n_estimators=400))

lgb_model_fa.fit(input_train_fa, target_train_fa)
lgb_preds_test = lgb_model_fa.predict(input_test_fa)
lgb_preds_train = lgb_model_fa.predict(input_train_fa)
evaluate_model_original(lgb_preds_test, lgb_preds_train, target_test_fa, target_train_fa, 'lgb model')

#---------------------------------------------
'''