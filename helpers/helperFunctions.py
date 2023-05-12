import math

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, classification_report
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics


# Filter to determine where an  occured
def findArea(row):
    s = ""
    #  id = row['id']
    # print(row)
    x = row['x']
    y = row['y']
    if (x >= 0 and x <= 16 and y >= 0 and y <= 19):
        s = 1
    elif (x > 16 and x <= 33 and y >= 0 and y <= 19):
        s = 5
    elif (x > 33 and x <= 50 and y >= 0 and y <= 19):
        s = 9
    elif (x > 50 and x <= 67 and y >= 0 and y <= 19):
        s = 15
    elif (x > 67 and x <= 84 and y >= 0 and y <= 19):
        s = 19
    elif (x > 84 and x <= 100 and y >= 0 and y <= 19):
        s = 24
    elif (x > 16 and x <= 33 and y > 19 and y <= 37):
        s = 7
    elif (x > 33 and x <= 50 and y > 19 and y <= 37):
        s = 11
    elif (x >= 50 and x <= 67 and y > 19 and y <= 37):
        s = 17
    elif (x > 67 and x <= 84 and y > 19 and y <= 37):
        s = 21
    elif (x > 16 and x <= 33 and y > 37 and y <= 63):
        s = 8
    elif (x > 33 and x <= 50 and y > 37 and y <= 63):
        s = 13
    elif (x > 50 and x <= 67 and y > 37 and y <= 63):
        s = 18
    elif (x > 67 and x <= 84 and y > 37 and y <= 63):
        s = 23
    elif (x > 16 and x <= 33 and y > 63 and y <= 81):
        s = 6
    elif (x > 33 and x <= 50 and y > 63 and y <= 81):
        s = 12
    elif (x > 50 and x <= 67 and y > 63 and y <= 81):
        s = 16
    elif (x > 67 and x <= 84 and y > 63 and y <= 81):
        s = 22
    elif (x >= 0 and x <= 16 and y > 81 and y <= 100):
        s = 2
    elif (x > 16 and x <= 33 and y > 81 and y <= 100):
        s = 4
    elif (x > 33 and x <= 50 and y > 81 and y <= 100):
        s = 10
    elif (x > 50 and x <= 67 and y > 81 and y <= 100):
        s = 14
    elif (x > 67 and x <= 84 and y > 81 and y <= 100):
        s = 20
    elif (x > 84 and x <= 100 and y > 81 and y <= 100):
        s = 25
    elif (x >= 0 and x <= 16 and y > 19 and y <= 81):
        s = 3
    elif (x >= 84 and x <= 100 and y > 19 and y <= 81):
        s = 26
    else:
        s = 0
    return s

def zone(row):
    s = ""
    #  id = row['id']
    # print(row)
    x = row['x']
    y = row['y']
    if (x >= 0 and x <= 33 and y >= 0 and y <= 33):
        s = "Zone 1 Actions"
    elif (x >= 0 and x <= 33 and y > 33 and y <= 67):
        s = "Zone 2 Actions"
    elif (x >= 0 and x <= 33 and y > 67 and y <= 100):
        s = "Zone 1 Actions"
    elif (x > 33 and x <= 67 and y >= 0 and y <= 33):
        s = "Zone 3 Actions"
    elif (x > 33 and x <= 67 and y > 33 and y <= 67):
        s = "Zone 4 Actions"
    elif (x > 33 and x <= 67 and y > 67 and y <= 100):
        s = "Zone 3 Actions"
    elif (x > 67 and x <= 100 and y >= 0 and y <= 33):
        s = "Zone 5 Actions"
    elif (x > 67 and x <= 100 and y > 33 and y <= 67):
        s = "Zone 6 Actions"
    elif (x >= 67 and x <= 100 and y >= 67 and y <= 100):
        s = "Zone 5 Actions"
    else:
        s = "Zone 0 Actions"
    return s



def pen_shots(x, y):
    return np.where(x > 83,
                    np.where(x < 101,
                             np.where(y > 18,
                                      np.where(y < 82, 1.00000, 0.0000), 0.00000), 0.00000), 0.00000)

def non_pen_shots(x, y):
    return np.where(x > 83,
                    np.where(x < 101,
                             np.where(y > 18,
                                      np.where(y < 82, 0.00000, 1.0000), 1.00000), 1.00000), 1.00000)

def last_third_def(x):
    return np.where(x > 66, 1.00000, 0.00000)

def isWhiteSpaceCross (eventType, row):
    x_start = row['x']
    y_start = row['y']
    x_end = row['end_x']
    y_end = row['end_y']
    if(row[eventType] == 1):
        ws_start_condition = x_start > 50 and x_start <= 100 and y_start >= 0 and y_start < 19 or x_start > 50 and x_start <= 100 and y_start > 81 and y_start <= 100
        end_condition = x_end > 84 and x_end <= 100 and y_end > 19 and y_end < 81
        if(ws_start_condition and end_condition):
            return 1
        else: return 0
    else: return 0


def isHalfSpaceCross (eventType, row):
    x_start = row['x']
    y_start = row['y']
    x_end = row['end_x']
    y_end = row['end_y']
    if(row[eventType] == 1):
        hs_start_condition = x_start > 50 and x_start <= 84 and y_start >= 19 and y_start <= 37 or x_start > 50 and x_start <= 84 and y_start >= 63 and y_start <= 81
        end_condition = x_end > 84 and x_end <= 100 and y_end > 19 and y_end < 81
        if(hs_start_condition and end_condition):
            return 1
        else: return 0
    else: return 0


def ec(x1, x2, y1, y2):
    return np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


def pp(x1, x2):
    dist = x2 - x1
    return np.where(x1 > 50, np.where(dist > 10, 1.00000, 0.00000),
                    np.where(dist > 30, 1.00000, 0.00000))


def direction(x1, x2):
    dist = x2 - x1
    return np.where(dist > 4, 'forward',
                    np.where(dist < -4, 'backward', 'horizontal'))


def non_forward(x1, x2):
    dist = x2 - x1
    return np.where(dist > 3, 0.00000, 1.00000)


def switch(y1, y2):
    dist = y2 - y1
    return np.where(dist > 35, 1.00000, 0.00000)


def gmode(df):
    temp = df.groupby(['playerId', 'seasonId']).obj.columns
    temp = temp.drop('playerId')
    temp = temp.drop('seasonId')
    print(temp)
    gtemp1 = df.groupby(['playerId', 'seasonId'], as_index=False)['Simple pass_zone'].agg(pd.Series.mode)
    gtemp1.pop('Simple pass_zone')
    for i in temp:
        print(i)
        gtemp2 = df.groupby(['playerId', 'seasonId'], as_index=False)[i].agg(gmodeHelp)
        print(gtemp2)
        gtemp1 = pd.merge(gtemp1, gtemp2, on=['playerId', 'seasonId'])
        print(gtemp1)
    return gtemp1


def gmodeHelp(x):
    m = pd.Series.mode(x)
    return m.values[0] if not m.empty else np.nan

def pos_group(row):
    x = row['map_group']
    g = ['GK']
    d = ['CB']
    m = ['AM', 'CM', 'DM']
    w = ['LM', 'RM', 'LW', 'RW', 'RB', 'LB', 'LWB', 'RWB']
    f = ['FW']
    if x in g:
        return "GK"
    elif x in d:
        return "DEF"
    elif x in m:
        return "MID"
    elif x in w:
        return "WIDE"
    elif x in f:
        return "ATT"
    else:
        return "other"


def off_def(row):
    x = row['map_group']
    off = ['FW', 'LM', 'RM', 'LW', 'RW', 'AM', 'CM']
    deff = ['CB', 'RB', 'LB', 'LWB', 'RWB', 'DM']
    if x in off:
        return "off"
    elif x in deff:
        return "def"
    else:
        return "other"

def opt_clus(dr):
    n_range = range(2, 21)
    # bic_score = []
    # aic_score = []
    sil_score = []
    chi_score = []
    dbi_score = []

    for n in n_range:
        gm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gm.fit(dr)
        labels = gm.predict(dr)
        #labels = labels.reshape(labels.shape[0], 1) # for 3D
        print(("Clusters: ", n, "Siloutte: ", metrics.silhouette_score(dr, labels, metric='euclidean')))
        # bic_score.append(gm.bic(dr))
        # aic_score.append(gm.aic(dr))
        sil_score.append(metrics.silhouette_score(dr, labels, metric='euclidean'))
        chi_score.append(metrics.calinski_harabasz_score(dr, labels)/10000)
        dbi_score.append(metrics.davies_bouldin_score(dr, labels))

    fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
    # ax.plot(n_range, bic_score, '-o', color='orange', label='BIC')
    # ax.plot(n_range, aic_score, '-o', color='blue', label='AIC')
    ax.plot(n_range, sil_score, '-o', color='orange', label='SIL')
    ax.plot(n_range, chi_score, '-o', color='blue', label='CHI')
    ax.plot(n_range, dbi_score, '-o', color='green', label='DBI')
    ax.set(xlabel='Number of Clusters', ylabel='Score')
    ax.set_xticks(n_range)
    ax.set_title('Evaluation Scores Per Number Of Clusters')
    ax.legend(fontsize='x-large')
    plt.show()


def gmm_to_df(df, phase):
    if phase == 'ip':
        frame = pd.DataFrame(df.reshape(df.shape[0], 1), columns=["ip_cluster"])
    elif phase == 'op':
        frame = pd.DataFrame(df.reshape(df.shape[0], 1), columns=["op_cluster"])
    return frame


# Identifying highly correlated features
def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-3*IQR)) | (df>(q3+3*IQR)))]
   return outliers


def names_clusters(data, cluster):
    df = data.iloc[:, np.r_[0, 5, 6:12]]
    players = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Players.csv', sep=";", encoding='unicode_escape')
    players.drop(columns=players.columns[0], axis=1, inplace=True)
    dfp = pd.merge(players, df, on='playerId')
    dfp = dfp[dfp.ip_cluster == cluster]
    dfp = dfp.iloc[:, np.r_[4, 2, 1, 18, 17, 15, 16, 13, 14]]
    return dfp

def possession_action(row):
    x = row['typePrimary']
    possession = ['pass', 'free_kick', 'shot', 'throw_in', 'shot_against', 'touch', 'goal_kick', 'corner', 'acceleration', 'offside', 'penalty']
    column_list = ['assist', 'carry', 'dribble', 'foul_suffered', 'linkup_play', 'offensive_duel', 'progressive_run', 'second_assist', 'third_assist']
    if x in possession:
        return 1
    else:
        for col in column_list:
            if row[col] == 1:
                return 1
    return 0


def non_possession_action(row):
    x = row['typePrimary']
    non_possession = ['interception', 'infraction', 'shot_against', 'clearance']
    column_list = ['aerial_duel', 'counterpressing_recovery', 'defensive_duel', 'dribbled_past_attempt', 'loose_ball_duel', 'recovery', 'sliding_tackle']
    if x in non_possession:
        return 1
    else:
        for col in column_list:
            if row[col] == 1:
                return 1
    return 0


def opp_space(df, cols):
    possession = ["assist", "back_pass", "carry", "deep_completed_cross", "deep_completition", "dribble", "forward_pass", "foul_suffered", "goal", "head_shot", "key_pass", "lateral_pass", "linkup_play", "long_pass", "offensive_duel", "pass_into_penalty_area", "pass_to_final_third", "progressive_pass", "progressive_run", "second_assist", "short_or_medium_pass", "smart_pass", "third_assist", "through_pass", "touch_in_box", "under_pressure", "cross", "shots_PA", "shots_nonPA", "ws_cross", "hs_cross"]
    non_possession = ["aerial_duel", "conceded_goal", "counterpressing_recovery", "defensive_duel", "dribbled_past_attempt", "ground_duel", "loose_ball_duel", "penalty_foul", "pressing_duel", "recovery", "sliding_tackle"]
    zone = ['Zone 1 Actions', 'Zone 2 Actions', 'Zone 3 Actions', 'Zone 4 Actions', 'Zone 5 Actions', 'Zone 6 Actions']
    for i in cols:
        name = i + '_tendency'
        if i in possession:
            df[name] = df[i] / df['posAction'] * df[i]
            df = df.drop([i], axis=1)
        elif i in non_possession:
            df[name] = df[i] / df['nonPosAction'] * df[i]
            df = df.drop([i], axis=1)
        elif i in zone:
            df[name] = df[i] / (df['posAction'] + df['nonPosAction']) * df[i]
            df = df.drop([i], axis=1)
    return df

def evaluate_model(target_pred, target_test, modelType):
    f1 = f1_score(target_test, target_pred, average='weighted')
    cohen_kappa = cohen_kappa_score(target_test, target_pred)
    print('Model accuracy score ' + modelType + ': {0:0.4f}'. format(accuracy_score(target_test, target_pred)))
    print('F1 Score: ', "%.2f" % (f1*100))
    print('Cohen Kappa: ', "%.2f" % cohen_kappa)
    #print(classification_report(target_test, target_pred))

