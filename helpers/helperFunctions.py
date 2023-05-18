import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.cm as cm
from datetime import datetime
from helpers.student_bif_code import *
from matplotlib.colors import LinearSegmentedColormap


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
    x = row['position']
    g = ['gk']
    cb = ['rcb', 'lcb', 'cb', 'lcb3', 'rcb3']
    b = ['rb', 'lb', 'rwb', 'lwb', 'rb5', 'lb5']
    m = ['lcmf3', 'rdmf', 'lcmf', 'rcmf3', 'rcmf', 'dmf', 'ldmf']
    am = ['lamf', 'amf', 'ramf', 'ramf']
    w = ['lw', 'rw', 'lwf', 'rwf']
    a = ['cf', 'ss']
    if x in g:
        return "GK"
    elif x in cb:
        return "CB"
    elif x in b:
        return "FB"
    elif x in m:
        return "MC"
    elif x in am:
        return "AM"
    elif x in w:
        return "W"
    elif x in a:
        return "ST"
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
    threshold_step = 0.05
    sil_score = []
    chi_score = []
    dbi_score = []

    for n in n_range:
        best_sil = 0
        best_chi = 0
        best_dbi = 100
        gm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gm.fit(dr)
        probs = gm.predict_proba(dr)

        for threshold in np.arange(0.3, 0.8 + threshold_step, threshold_step):
            labels = np.argmax(probs, axis=1)
            labels[probs.max(axis=1) < threshold] = -1
            labels, dr = zip(*[(c, o) for c, o in zip(labels, dr) if c >= 0])

            sil_val = metrics.silhouette_score(dr, labels, metric='euclidean', sample_size=None, random_state=None)
            chi_val = metrics.calinski_harabasz_score(dr, labels)
            dbi_val = metrics.davies_bouldin_score(dr, labels)

            if sil_val > best_sil:
                best_sil = sil_val
                print("Threshold: ", round(threshold, 2))
            if chi_val > best_chi: best_chi = chi_val
            if dbi_val < best_dbi: best_dbi = dbi_val

        print(("Cluster: ", n, "SIL: ", round(best_sil, 2), "CHI: ", round(best_chi, 2), "DBI: ", round(best_dbi, 2)))
        sil_score.append(best_sil)
        chi_score.append(best_chi/20000)
        dbi_score.append(best_dbi)

    fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
    ax.plot(n_range, sil_score, '-o', color='gold', label='SIL')
    ax.plot(n_range, chi_score, '-o', color='darkblue', label='CHI')
    ax.plot(n_range, dbi_score, '-o', color='darkorange', label='DBI')
    ax.set(xlabel='Number of Clusters', ylabel='Score')
    ax.set_xticks(n_range)
    ax.legend(fontsize='x-large')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('Evaluation Scores Per Number Of Clusters', fontsize=14, fontweight='bold')
    plt.grid(color='lightgray', alpha=0.25, zorder=1)
    plt.savefig('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/clustMetrics.png')
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
    non_possession = ["aerial_duel", "conceded_goal", "counterpressing_recovery", "defensive_duel", "dribbled_past_attempt", "loose_ball_duel", "penalty_foul", "pressing_duel", "recovery", "sliding_tackle"]
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


'''The weights are calculated using the method of variance explained by the projection (VEP), which is a technique for 
measuring the importance of a feature for a particular cluster. The VEP weight for a feature is proportional to the 
difference between the mean value of the feature in the target cluster and the mean value of the feature in all the 
other clusters, weighted by the variance of the feature in the target cluster. The intuition behind this is that if a 
feature has a high VEP weight, then it is a good predictor of the target cluster.'''
def get_weight_dicts(X, clusters):
    weight_dicts = {}
    for cluster_label in np.unique(clusters):
        weight_dicts[f'Cluster {cluster_label}'] = {}

    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]
        cluster_means = X.iloc[cluster_indices].median()

        for feature in X.columns:
            # feature_name = feature.replace("_tendency", "_vaep")
            feature_name = feature # fjern
            feature_mean = X[feature].median()
            cluster_feature_mean = X.iloc[cluster_indices][feature].median()

            other_cluster_means = X[clusters != cluster_label].median()
            other_cluster_feature_mean = other_cluster_means[feature]

            # weight = ((cluster_feature_mean - feature_mean) + (cluster_feature_mean - other_cluster_feature_mean)) * np.abs(cluster_feature_mean -
            weight = (cluster_feature_mean - feature_mean)
            weight_dicts[f'Cluster {cluster_label}'][feature_name] = weight

    return weight_dicts


def get_weight_dicts2(X, cluster_labels, num_factors=10):
    subsets = {}
    for label in set(cluster_labels):
        mask = (cluster_labels == label)
        subsets[label] = X[mask]

    feature_weights = {}
    for label, subset in subsets.items():
        if num_factors is None:
            num_factors = min(subset.shape[0], subset.shape[1])

        fa = FactorAnalysis(n_components=num_factors, svd_method='lapack')
        fa.fit(subset)

        feature_names = fa.feature_names_in_

        weights = np.abs(fa.components_)
        weights /= np.sum(weights, axis=1, keepdims=True)
        cluster_weights = {}
        for i, feature_name in enumerate(feature_names):
            cluster_weights[feature_name] = np.mean(weights[:, i])

        feature_weights[f'Cluster {label}'] = cluster_weights

    return feature_weights


def cluster_to_dataframe(weight_dicts, cluster_name):
    cluster_weights = weight_dicts[cluster_name]
    df = pd.DataFrame.from_dict(cluster_weights, orient='index').T
    return df


def plot_sorted_bar_chart(df):
    df.columns = [col.replace('_tendency', '') for col in df.columns]
    series = df.T.squeeze()
    sorted_series = series.sort_values(ascending=False)

    ax = sorted_series.plot(kind='bar', figsize=(12, 6), color=cm.plasma_r(sorted_series / float(max(sorted_series))))
    ax.set_xlabel('Features')
    ax.set_ylabel('Weights')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('Pareto Feature Weight', fontsize=14, fontweight='bold')
    plt.grid(color='lightgray', alpha=0.25, zorder=1)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/perf/W.png')
    plt.show()

def plot_sorted_bar_chart_p(df, player):
    player_df = df[df['shortName'] == player]
    player_df.columns = [col.replace('Weighted Score', 'Score') for col in player_df.columns]
    player_df = player_df.loc[:, player_df.notna().any()]
    player_df = player_df.loc[:, ~player_df.columns.str.contains('Trend')]
    player_df = player_df.loc[:, ~player_df.columns.str.contains('ip_cluster')]
    weights = player_df.iloc[0, 5:]  # Get the weights for the player
    weights_sorted = weights.sort_values(ascending=False)  # Sort the weights in descending order
    weights_normalized = pd.to_numeric(weights_sorted) / float(max(weights_sorted))
    ax = weights_sorted.plot(kind='bar', figsize=(12, 8), color=cm.plasma_r(weights_normalized))
    ax.set_xlabel('Features', weight='bold')
    ax.set_ylabel('Score', weight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('Cluster Score Pareto for {}'.format(player), fontsize=14, fontweight='bold')
    plt.grid(color='lightgray', alpha=0.25, zorder=1)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/perf/SH_perf1.png')
    plt.show()


def calculate_weighted_scores(data, weight_dicts):
    data.columns = [col.replace('_vaep', '_tendency') for col in data.columns]
    score_data = pd.DataFrame()  # create a new dataframe to store the scores
    for name, weights in weight_dicts.items():
        scores = []
        for index, row in data.iterrows():
            weighted_score = sum(row[feature] * weight for feature, weight in weights.items())
            scores.append(weighted_score)
        score_data[f'{name} Weighted Score'] = pd.Series(scores)
    return pd.concat([data, score_data], axis=1)


def calculate_weighted_scores2(data, weight_dicts):
    data.columns = [col.replace('_vaep', '_tendency') for col in data.columns]

    penalization = {
        'Cluster -1': {'CB': 0, 'FB': 0, 'MC': 0, 'AM': 0, 'W': 0, 'ST': 0},
        'Cluster 0': {'CB': 0, 'FB': 0.1, 'MC': 0.3, 'AM': 0.5, 'W': 0.5, 'ST': 0.5},
        'Cluster 1': {'CB': 0, 'FB': 0.1, 'MC': 0.3, 'AM': 0.5, 'W': 0.5, 'ST': 0.5},
        'Cluster 2': {'CB': 0, 'FB': 0.1, 'MC': 0.3, 'AM': 0.5, 'W': 0.5, 'ST': 0.5},
        'Cluster 3': {'CB': 0.5, 'FB': 0.3, 'MC': 0, 'AM': 0, 'W': 0, 'ST': 0.3},
        'Cluster 4': {'CB': 0.3, 'FB': 0.1, 'MC': 0, 'AM': 0.1, 'W': 0.3, 'ST': 0.5},
        'Cluster 5': {'CB': 0.5, 'FB': 0.3, 'MC': 0, 'AM': 0, 'W': 0.1, 'ST': 0.3},
        'Cluster 6': {'CB': 0.5, 'FB': 0.2, 'MC': 0.15, 'AM': 0.1, 'W': 0, 'ST': 0},
        'Cluster 7': {'CB': 0.5, 'FB': 0.2, 'MC': 0.15, 'AM': 0.1, 'W': 0.1, 'ST': 0},
        'Cluster 8': {'CB': 0.5, 'FB': 0.2, 'MC': 0.1, 'AM': 0, 'W': 0, 'ST': 0},
        'Cluster 9': {'CB': 0.05, 'FB': 0, 'MC': 0.1, 'AM': 0.1, 'W': 0.1, 'ST': 0.3},
        'Cluster 10': {'CB': 0.2, 'FB': 0, 'MC': 0.15, 'AM': 0.1, 'W': 0.05, 'ST': 0.2},
        'Cluster 11': {'CB': 0.05, 'FB': 0, 'MC': 0.15, 'AM': 0.2, 'W': 0.1, 'ST': 0.3},
        'Cluster 12': {'CB': 0.5, 'FB': 0.1, 'MC': 0.1, 'AM': 0, 'W': 0, 'ST': 0.1},
        'Cluster 13': {'CB': 0.5, 'FB': 0.2, 'MC': 0.2, 'AM': 0.05, 'W': 0, 'ST': 0},
        'Cluster 14': {'CB': 0.5, 'FB': 0.2, 'MC': 0.2, 'AM': 0.05, 'W': 0, 'ST': 0},
        'Cluster 15': {'CB': 0.1, 'FB': 0.05, 'MC': 0, 'AM': 0.05, 'W': 0.2, 'ST': 0.3},
        'Cluster 16': {'CB': 0.1, 'FB': 0, 'MC': 0, 'AM': 0, 'W': 0.15, 'ST': 0.3},
        'Cluster 17': {'CB': 0.05, 'FB': 0.05, 'MC': 0, 'AM': 0.2, 'W': 0.3, 'ST': 0.5}
    }

    score_data = pd.DataFrame()  # create a new dataframe to store the scores
    for name, weights in weight_dicts.items():
        scores = []
        print(type(weights))
        for index, row in data.iterrows():
            weighted_score = sum((row[feature] * weight * (1 - penalization[name][row['pos_group']])) for feature, weight in weights.items())
            scores.append(weighted_score)
        score_data[f'{name} Weighted Score'] = pd.Series(scores)
    return pd.concat([data, score_data], axis=1)


def perf(df, df2, mode, cluster=None, age=None):
    df = pd.merge(df2, df, on='playerId')
    df['birthDate'] = pd.to_datetime(df['birthDate']).dt.year
    df['birthDate'] = df['birthDate'].apply(calculate_age_bracket)
    df = df.rename(columns={'birthDate': 'ageBracket'})

    trend = perf_trend(df)
    trend = trend[trend.filter(like='Weighted Score Trend').columns]

    df_ids = df.iloc[:, : 6]
    stats = df[df.filter(like='Weighted Score').columns]
    stats = stats.drop(['Cluster -1 Weighted Score'], axis=1)

    if mode == "scaled":
        custom_scaler = MinMaxScaler(feature_range=(1, 100))
        stats[stats.columns] = custom_scaler.fit_transform(stats[stats.columns])
    if mode == "percentiles":
        CB = df[df['ip_cluster'].isin([0,1,2])]
        CB_ids = CB.iloc[:, : 6]
        CB_stats = CB.filter(like='Weighted Score').filter(items=['Cluster 0 Weighted Score', 'Cluster 1 Weighted Score', 'Cluster 2 Weighted Score'])
        CB_stats = round(CB_stats.rank(pct=True) * 100, 0)
        CB_stats = pd.concat([CB_ids.reset_index(drop=True), CB_stats.reset_index(drop=True)], axis=1)
        AM = df[df['ip_cluster'].isin([3,4,5])]
        AM_ids = AM.iloc[:, : 6]
        AM_stats = AM.filter(like='Weighted Score').filter(items=['Cluster 3 Weighted Score', 'Cluster 4 Weighted Score', 'Cluster 5 Weighted Score'])
        AM_stats = round(AM_stats.rank(pct=True) * 100, 0)
        AM_stats = pd.concat([AM_ids.reset_index(drop=True), AM_stats.reset_index(drop=True)], axis=1)
        ST = df[df['ip_cluster'].isin([6,7,8])]
        ST_ids = ST.iloc[:, : 6]
        ST_stats = ST.filter(like='Weighted Score').filter(items=['Cluster 6 Weighted Score', 'Cluster 7 Weighted Score', 'Cluster 8 Weighted Score'])
        ST_stats = round(ST_stats.rank(pct=True) * 100, 0)
        ST_stats = pd.concat([ST_ids.reset_index(drop=True), ST_stats.reset_index(drop=True)], axis=1)
        FB = df[df['ip_cluster'].isin([9,10,11])]
        FB_ids = FB.iloc[:, : 6]
        FB_stats = FB.filter(like='Weighted Score').filter(items=['Cluster 9 Weighted Score', 'Cluster 10 Weighted Score', 'Cluster 11 Weighted Score'])
        FB_stats = round(FB_stats.rank(pct=True) * 100, 0)
        FB_stats = pd.concat([FB_ids.reset_index(drop=True), FB_stats.reset_index(drop=True)], axis=1)
        W = df[df['ip_cluster'].isin([12,13,14])]
        W_ids = W.iloc[:, : 6]
        W_stats = W.filter(like='Weighted Score').filter(items=['Cluster 12 Weighted Score', 'Cluster 13 Weighted Score', 'Cluster 14 Weighted Score'])
        W_stats = round(W_stats.rank(pct=True) * 100, 0)
        W_stats = pd.concat([W_ids.reset_index(drop=True), W_stats.reset_index(drop=True)], axis=1)
        CM = df[df['ip_cluster'].isin([15,16,17])]
        CM_ids = CM.iloc[:, : 6]
        CM_stats = CM.filter(like='Weighted Score').filter(items=['Cluster 15 Weighted Score', 'Cluster 16 Weighted Score', 'Cluster 17 Weighted Score'])
        CM_stats = round(CM_stats.rank(pct=True) * 100, 0)
        CM_stats = pd.concat([CM_ids.reset_index(drop=True), CM_stats.reset_index(drop=True)], axis=1)
    if mode == "cluster":
        clus = pd.concat([df_ids.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
        clus = clus[clus['ip_cluster'] == cluster]
        clus_ids = clus.iloc[:, : 6]
        clus = round(clus[clus.filter(like='Weighted Score').columns].rank(pct = True)*100, 0)
        clus = clus[f'Cluster {cluster} Weighted Score']
        clus = pd.concat([clus_ids.reset_index(drop=True), clus.reset_index(drop=True)], axis=1)
        return clus
    if mode == "age":
        df = df[df['ageBracket'] == age]
        CB = df[df['ip_cluster'].isin([0,1,2])]
        CB_ids = CB.iloc[:, : 6]
        CB_stats = CB.filter(like='Weighted Score').filter(items=['Cluster 0 Weighted Score', 'Cluster 1 Weighted Score', 'Cluster 2 Weighted Score'])
        CB_stats = round(CB_stats.rank(pct=True) * 100, 0)
        CB_stats = pd.concat([CB_ids.reset_index(drop=True), CB_stats.reset_index(drop=True)], axis=1)
        AM = df[df['ip_cluster'].isin([3,4,5])]
        AM_ids = AM.iloc[:, : 6]
        AM_stats = AM.filter(like='Weighted Score').filter(items=['Cluster 3 Weighted Score', 'Cluster 4 Weighted Score', 'Cluster 5 Weighted Score'])
        AM_stats = round(AM_stats.rank(pct=True) * 100, 0)
        AM_stats = pd.concat([AM_ids.reset_index(drop=True), AM_stats.reset_index(drop=True)], axis=1)
        ST = df[df['ip_cluster'].isin([6,7,8])]
        ST_ids = ST.iloc[:, : 6]
        ST_stats = ST.filter(like='Weighted Score').filter(items=['Cluster 6 Weighted Score', 'Cluster 7 Weighted Score', 'Cluster 8 Weighted Score'])
        ST_stats = round(ST_stats.rank(pct=True) * 100, 0)
        ST_stats = pd.concat([ST_ids.reset_index(drop=True), ST_stats.reset_index(drop=True)], axis=1)
        FB = df[df['ip_cluster'].isin([9,10,11])]
        FB_ids = FB.iloc[:, : 6]
        FB_stats = FB.filter(like='Weighted Score').filter(items=['Cluster 9 Weighted Score', 'Cluster 10 Weighted Score', 'Cluster 11 Weighted Score'])
        FB_stats = round(FB_stats.rank(pct=True) * 100, 0)
        FB_stats = pd.concat([FB_ids.reset_index(drop=True), FB_stats.reset_index(drop=True)], axis=1)
        W = df[df['ip_cluster'].isin([12,13,14])]
        W_ids = W.iloc[:, : 6]
        W_stats = W.filter(like='Weighted Score').filter(items=['Cluster 12 Weighted Score', 'Cluster 13 Weighted Score', 'Cluster 14 Weighted Score'])
        W_stats = round(W_stats.rank(pct=True) * 100, 0)
        W_stats = pd.concat([W_ids.reset_index(drop=True), W_stats.reset_index(drop=True)], axis=1)
        CM = df[df['ip_cluster'].isin([15,16,17])]
        CM_ids = CM.iloc[:, : 6]
        CM_stats = CM.filter(like='Weighted Score').filter(items=['Cluster 15 Weighted Score', 'Cluster 16 Weighted Score', 'Cluster 17 Weighted Score'])
        CM_stats = round(CM_stats.rank(pct=True) * 100, 0)
        CM_stats = pd.concat([CM_ids.reset_index(drop=True), CM_stats.reset_index(drop=True)], axis=1)
        dfp = pd.concat([CB_stats, AM_stats, ST_stats, FB_stats,W_stats, CM_stats], axis=0, ignore_index=True)
        dfp = dfp.sort_values('playerId')
        dfp = dfp.reset_index(drop=True)
        return dfp

    dfp = pd.concat([df_ids.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    dfp = pd.concat([dfp.reset_index(drop=True), trend.reset_index(drop=True)], axis=1)

    if mode == "percentiles":
        dfp = pd.concat([CB_stats, AM_stats, ST_stats, FB_stats,W_stats, CM_stats], axis=0, ignore_index=True)
        dfp = dfp.sort_values('playerId')
        dfp = dfp.reset_index(drop=True)

    return dfp


def perf2(df, df2):
    df.drop(['Cluster -1 Weighted Score'], axis=1)
    df = df.drop(columns=df.columns[df.columns.get_loc('aerial_duel_tendency'):df.columns.get_loc('Zone 6 Actions_tendency') + 1])
    df.loc[:, ['Cluster 0 Weighted Score', 'Cluster 17 Weighted Score']] *= 100
    custom_scaler = MinMaxScaler(feature_range=(1, 100))
    for i in range(0, 18):
        min = df[df['ip_cluster'] == 1][f'Cluster {i} Weighted Score'].min()
        df.loc[df[f'Cluster {i} Weighted Score'] <= min, f'Cluster {i} Weighted Score'] = 1
        df.loc[(df[f'Cluster {i} Weighted Score'] >= min), f'Cluster {i} Weighted Score'] = custom_scaler.fit_transform(df.loc[(df[f'Cluster {i} Weighted Score'] >= min), [f'Cluster {i} Weighted Score']])

    dfp = pd.merge(df2, df, on='playerId')
    return dfp


def calculate_age_bracket(birth_year):
    today = datetime.today()
    age = today.year - birth_year

    if age <= 19:
        return "Youth"
    elif age <= 24:
        return "Rising to Prime"
    elif age <= 30:
        return "Prime"
    elif age <= 34:
        return "Falling from Prime"
    elif age >= 35:
        return "Veteran"
    else:
        "NA"


def perf_trend(df):
    # df = pd.merge(df2, df, on='playerId')
    # df = df.drop(['Cluster -1 Weighted Score'], axis=1)
    ssn_name = load_db_to_pd(sql_query="SELECT seasonId, startDate FROM Wyscout_Seasons", db_name='Scouting')
    df = pd.merge(df, ssn_name, on='seasonId')
    df['startDate'] = pd.to_datetime(df['startDate']).dt.year

    grouped = df.groupby(['playerId', 'shortName'])
    pct_change = {}
    for i in range(18):
        col_name = f'Cluster {i} Weighted Score'
        pct_change[col_name] = grouped.apply(lambda x: (x.loc[x['startDate'] == 2021, col_name].iloc[0] / x.loc[x['startDate'] == 2020, col_name].iloc[0] - 1) if (2020 in x['startDate'].values and 2021 in x['startDate'].values) else np.nan)

    def get_trend(pct_change):
        return np.where(pct_change >= 0.02, 'Increase',
                        np.where(pct_change <= -0.02, 'Decrease',
                                 np.where(abs(pct_change) < 0.02, 'Stable', '')))

    trend_dfs = []
    for i in range(18):
        col_name = f'Cluster {i} Weighted Score'
        trend_name = f'{col_name} Trend'
        trend = pct_change[col_name].groupby(['playerId', 'shortName']).apply(get_trend).reset_index(name=trend_name)
        trend.drop_duplicates(subset=['playerId', 'shortName'], inplace=True)
        trend['startDate'] = 2021
        trend_dfs.append(trend)

    trend_df = pd.concat(trend_dfs, axis=1)
    trend_df.drop_duplicates(subset=['playerId', 'shortName'], keep='last', inplace=True)
    trend_df = trend_df.loc[:, ~trend_df.columns.duplicated()]
    df = pd.merge(df, trend_df, on=['playerId', 'shortName', 'startDate'], how='left')
    # df.drop(df.filter(like='_tendency').columns, axis=1)
    df = df.sort_values('playerId')

    return df


def get_weighted_score(row, cluster_label):
    score_column = f"Cluster {cluster_label} Weighted Score"
    return row[score_column]
