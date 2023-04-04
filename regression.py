import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from helpers.student_bif_code import load_db_to_pd


#Helper function
def perform_regression(model, input_train, input_test, target_train, target_test, label):
    # Fit model for each position
    model.fit(input_train, target_train)
    # Predict
    target_pred = model.predict(input_test)

    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(target_test, target_pred)))

    # Genereate and print feature scores
    feature_scores = pd.Series(model.feature_importances_, index=input_train.columns).sort_values(ascending=False)
    topTen = feature_scores.head(15)
    feature_scores_df = feature_scores.to_frame(feature_scores)

    # Create and display an vizualization of feature importances
    sns.barplot(x=topTen, y=topTen.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Features importances " + label)
    plt.show()

    print(classification_report(target_test, target_pred))


df = load_db_to_pd(sql_query="SELECT * FROM [Development].[dbo].[sd_tableF]",
                   db_name='Development')

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
                                                                     test_size=0.33, random_state=42)

# Creation of randomForestClassifier model
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=4)

#Exexcute function to retrieve feature imporances and accuracy scores
#DEF


perform_regression(model, def_input_train, def_input_test, def_target_train, def_target_test, "DEF")
#MID
perform_regression(model, mid_input_train, mid_input_test, mid_target_train, mid_target_test, "MID")
#WIDE
perform_regression(model, wide_input_train, wide_input_test, wide_target_train, wide_target_test, "WIDE")
#ATT
perform_regression(model, att_input_train, att_input_test, att_target_train, att_target_test, "ATT")
