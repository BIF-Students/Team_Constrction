# This file contains the functions used for the Cluster_viz.py file

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from helpers.metrics2 import *
import plotly.express as px
import numpy as np
import helperFunctions
from helpers.helperFunctions import *


# Produce averages for each cluster for specified values in arguemnt
def get_stat_values (data, metric):
    clusters = data.ip_cluster.unique()
    final_frame = pd.DataFrame()
    data = data[metric]
    for cluster in clusters:
        cluster_frame = data[data['ip_cluster'] == cluster]
        frame = cluster_frame.loc[:, cluster_frame.columns != 'ip_cluster']
        transposed = frame.T
        transposed['vals'] = transposed.mean(axis =1)
        transposed['ip_cluster'] = cluster
        vals_clusters = transposed[['vals', 'ip_cluster']]
        vals_clusters_labels = vals_clusters.rename_axis("labels").reset_index()
        vals_clusters_labels.columns.values[0] = "labels"
        final_frame = pd.concat([final_frame, vals_clusters_labels])
    return final_frame


# Sum scores per key in dict per player
def compute_sum_per_metric(data, dict):
    for key, val in dict.items():
        h = val
        h.remove('ip_cluster')
        data[key] = data[h].sum(axis=1)
    return data


# Get average cluster feature values
def get_avg(df):
    averages = pd.DataFrame(columns=['labels', 'vals'])
    labels = df.labels.unique()
    for label in labels:
        label_df = df[df['labels'] == label]
        avg_df = label_df['vals'].mean()
        data = [[label, avg_df]]
        df_done = pd.DataFrame(data, columns=['labels', 'vals'])
        averages = pd.concat([averages, df_done])
    return averages


# Create spiderweb diagram for specified features
def make_spider_web(raw_data, stat, title_att):
  stat_vals = get_stat_values(raw_data, stat)
  averages_found = get_avg(stat_vals)
  clusters = stat_vals.ip_cluster.unique()
  avg_cluster = max(clusters) +1
  averages_found['ip_cluster'] = avg_cluster
  clusters.sort()

  fig = go.Figure(layout=go.Layout(
      title=go.layout.Title(text='Comparison clusters - ' + title_att ),
      polar={'radialaxis': {'visible': True}},
      showlegend=True))

  fig.add_trace(
      go.Scatterpolar(r=averages_found.vals, fill='toself', opacity=0.4, theta=averages_found.labels, name="# AVG " + str(avg_cluster)),)

  for cluster in clusters:
      cluster_df = stat_vals[stat_vals['ip_cluster'] == cluster]
      frame = cluster_df.loc[:, cluster_df.columns != 'ip_cluster']
      fig.add_trace(
          go.Scatterpolar(r=frame.vals, fill='toself', opacity=0.4, theta=frame.labels, name="# Cluster " + str(cluster)),)

  fig.show()


# Create Pareto chart to visualize cluster characteristics for average feature scores
def pareto(df, cluster):
    df = df.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
    df = df.groupby(['ip_cluster'], as_index=False).mean()
    df = df[df.ip_cluster == cluster]
    df = df.drop(['ip_cluster'], axis=1)
    df = df.T
    df.rename(columns={df.columns[0]: "value"}, inplace=True)
    df = df.sort_values('value', ascending=False)

    fig = px.bar(df, x=df.index, y=df['value'])
    fig.add_hline(y=0.5)
    fig.show()


# Create dataframe to compare cluster feature means
def stat_comp(df):
    df = df.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
    df = df.groupby(['ip_cluster'], as_index=False).mean()
    return df


# Create dataframe with overview of which positions occupy clusters
def pos_dist(df, cluster):
    df = df.drop(['playerId', 'seasonId'], axis=1)
    df = df[df.ip_cluster == cluster]
    df = df.drop(['ip_cluster'], axis=1)
    df = df.groupby(['pos_group'], as_index=False).count()
    return df