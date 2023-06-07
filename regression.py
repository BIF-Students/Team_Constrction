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
from xgboost import XGBClassifier

from archive.helpers.helperFunctions import *
from archive.helpers.student_bif_code import load_db_to_pd
from archive.helpers.visualizations import *
import random
import statistics
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
import plotly.offline as pyo
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import time

from sklearn.metrics import log_loss

df_players = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/data_files/df_players.csv")
df_players = load_db_to_pd(sql_query = "SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Players]", db_name='Scouting_Raw')


def plot_confusion_matrix(target_test, lgb_preds_test):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(target_test, lgb_preds_test)

    # Calculate the percentages
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(18, 8))

    # Set the color theme to "plasma"
    cmap = sns.color_palette("plasma")

    # Plot the confusion matrix with percentages using Seaborn
    sns.heatmap(conf_matrix_percent, fmt='.0f', annot =  True, cmap=cmap, cbar=False, ax=ax, annot_kws={"fontsize": 12})

    # Set the tick positions to the middle of each square
    ax.set_xticks([i + 0.5 for i in range(18)], minor=False)
    ax.set_yticks([i + 0.5 for i in range(18)], minor=False)

    # Set the tick labels
    ax.set_xticklabels(range(18), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(range(18), rotation=0, fontsize=10)

    # Set the axis labels and title
    ax.set_xlabel('Predicted Roles', fontsize=12)
    ax.set_ylabel('True Roles', fontsize=12)
    ax.set_title('Confusion Matrix (Percentage)', fontsize=14)
    plt.savefig('C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/img/heat_1.png')

    # Show the plot
    plt.tight_layout()
    plt.show()
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




# Create models

rdf_model = RandomForestClassifier(criterion ='gini', n_estimators=100, min_samples_split=2, min_samples_leaf=7, max_features='sqrt',max_depth=5)
rdf_model.fit(input_train, target_train)
evaluate_model_original(input_train, target_train, input_test, target_test, rdf_model)

lgb_model = lgb.LGBMClassifier(learning_rate=0.045001 , n_estimators=110, objective='multiclass',
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=1 )
start_time = time.time()
lgb_model.fit(input_train, target_train)
end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

lgb_preds_test = lgb_model.predict(input_test)
lgb_preds_train = lgb_model.predict(input_train)
evaluate_model_original(input_train, target_train, input_test, target_test, lgb_model)


#plot_confusion_matrix(target_test, lgb_preds_test)


ovr_lgb = OneVsRestClassifier(LGBMClassifier(learning_rate=0.045001 , n_estimators=110,
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=10) )
start_time2 = time.time()
ovr_lgb.fit(input_train, target_train)
end_time2 = time.time()
execution_time2 = end_time2 - start_time2

print("Execution Time:", execution_time2, "seconds")
ovr_lgb_preds_test = ovr_lgb.predict(input_test)
ovr_lgb_preds_train = ovr_lgb.predict(input_train)
evaluate_model_original(input_train, target_train, input_test, target_test, ovr_lgb)
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

xgb_model = XGBClassifier(learning_rate=0.045001 , n_estimators=110, objective='multiclass',
                               max_depth=3, num_leaves= 100, feature_fraction = 0.7, min_data_in_leaf = 100,
                               min_child_samples=1 , reg_alpha=1 , reg_lambda=10 )
xgb_model.fit(input_train, target_train)
xgb_preds_test = xgb_model.predict(input_test)
xgb_preds_train = xgb_model.predict(input_train)
evaluate_model_original(input_train, target_train, input_test, target_test, xgb_model)


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


