# Import necessary modules
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from chemistry.chemistry_helpers import *



#Load possession event stream table
pos = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/df_pos.csv", decimal=",", sep=(';'))
#Load of table with evenstream and their VAEP scores
sd_table = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/sd_table.csv", decimal=",",   sep=(';'))

#Load of tables used to create components of the Chemistry expression
df_keepers, players_teams, matches_all, squad, def_suc_tot = get_data()

# Remove keepers from all used dataframes
keepers = df_keepers['playerId'].values.tolist()
squad = squad.query("playerId not in @keepers")
sd_table = sd_table.query("playerId not in @keepers")

# Impute zero values for sumVaep
sd_table['sumVaep'] = sd_table['sumVaep'].fillna(0)
pos['sumVaep'] = pos['sumVaep'].fillna(0)

#Insert ID of competition you wish to generate pairwise chemistry frome
# 564 is the Id realted to the Italian league
comp_id = 524
pairwise_chemistry_league, stats_league_21_22 = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, comp_id)

#The expected wins and losses of a team is extracted based on their xG per game
df_expected_league = compute_expected_outputs(stats_league_21_22)

#The team chemsitry is extracted
team_chem_league = get_team_chemistry(pairwise_chemistry_league)

#Merge expected wins of each team with their corresponding team chemistry
df_merged_league = team_chem_league.merge(df_expected_league, on=['teamId'])

#Print correlation vlue of league
print("{:.2f}".format((df_merged_league['chemistry'].corr(df_merged_league['expected_wins'], method='spearman'))))

#EXtract dataframe that contains names and teams of the players connected to their chemistries
df_overview_league = get_overview_frame(pairwise_chemistry_league, players_teams)


#Investigate teams and correlatoins in boxplot
teams_and_correlations = pd.read_csv("C:/Users/jhs/OneDrive - Brøndbyernes IF Fodbold/Skrivebord/corr.csv", sep=(';'), decimal=(','))
get_box(teams_and_correlations)

#Plot relationship between league and expected wins
label = "IT"
plot_relationship(df_merged_league, label)

'''
Scope study of Spanish league
Net values are extracted from https://www.transfermarkt.com/primera-division/startseite/wettbewerb/ES1
'''
#Get chemistries, expected wins and team chemistry
comp_id = 795
df_esp, df_process_21_22_esp = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, comp_id)
df_expected_esp = compute_expected_outputs(df_process_21_22_esp)
team_chem_esp = get_team_chemistry(df_esp)
df_merged_esp = team_chem_esp.merge(df_expected_esp, on=['teamId'])

#Extract Italian league figures to compare with the spanish league
comp_id = 524
df_it, df_process_21_22_it = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, comp_id)
df_expected_it = compute_expected_outputs(df_process_21_22_it)
team_chem_it = get_team_chemistry(df_it)
df_merged_it = team_chem_it.merge(df_expected_it, on=['teamId'])



#Attach team names
df_merged_esp_v2 = df_merged_esp.merge(players_teams[['teamId', 'name']], on = 'teamId')
#Construct dataframe with teams and their net worth
df_merged_esp_v2.name.unique()
teams_and_worth = [['Valencia', 225.65], ['Real Madrid', 860.80], ['Barcelona', 767.], ['Athletic Bilbao', 226.30],
       ['Atlético Madrid', 457.], ['Sevilla', 209], ['Osasuna', 127.70], ['Villarreal', 278.20],
       ['Real Betis', 252.50], ['Real Sociedad', 375.50], ['Celta de Vigo', 246.20], ['Levante', 0],
       ['Deportivo Alavés', 0], ['Getafe', 136.70], ['Cádiz', 62.40], ['Elche', 60.70], ['Granada', 0]]
net_worth_laliga = pd.DataFrame(teams_and_worth, columns = ['name', 'Total Value'])
#Merge net worth of team with dataframe contaning chemistry scores and expected wins
df_merged_esp_v2 = df_merged_esp_v2.rename(columns = {'expected_wins': 'Expected Wins' })
df_merged_esp_v3 = df_merged_esp_v2.merge(net_worth_laliga, on ='name')

#Inspect relationship between net worth and expected wins
plot_relationship_netVal(df_merged_esp_v3, 'Expected Wins', "Net Worth")
#Inspect expected wins agains chemistries
plot_relationship_chemistry(df_merged_esp_v3, 'Expected Wins', "Chemistry")


#Investigating correlation scores of offensive, defensive and overall chemistry for Italy and Spain
print('Spain')
print(df_merged_esp_v2['df_joi90'].corr(df_merged_esp_v2['Expected Wins']))
print(df_merged_esp_v2['df_jdi90'].corr(df_merged_esp_v2['Expected Wins']))
print(df_merged_esp_v2['chemistry'].corr(df_merged_esp_v2['Expected Wins']))
print('Italy')
print(df_merged_it['df_joi90'].corr(df_merged_it['expected_wins']))
print(df_merged_it['df_jdi90'].corr(df_merged_it['expected_wins']))
print(df_merged_it['chemistry'].corr(df_merged_it['expected_wins']))

#------------------------------------------------------------ ARCHIEVE ------------------------------------------------------------

#league level analysis an constructio of dataframe with all chemistry values saved in a csv file
'''
#Examinations made onm a league level
df_dk, df_process_21_22_dk = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 335)
df_eng, df_process_21_22_eng = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 364)
df_ger, df_process_21_22_ger = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 426)
df_fr, df_process_21_22_fr = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 412)
df_por, df_process_21_22_por = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 707)
df_ned, df_process_21_22_ned = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 635)
df_tur, df_process_21_22_tur = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 852)
df_bel, df_process_21_22_bel = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 198)
df_swi, df_process_21_22_swi = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 830)
df_gre, df_process_21_22_gre = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 448)
df_cro, df_process_21_22_cro = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 302)
df_cze, df_process_21_22_cze = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 323)
df_au, df_process_21_22_au = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 168)
df_ser, df_process_21_22_ser = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 905)
df_rus, df_process_21_22_rus = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 729)
df_sco, df_process_21_22_sco = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 750)
df_hun, df_process_21_22_hun = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 465)
df_ice, df_process_21_22_ice = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 480)
df_sk, df_process_21_22_sk = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 775)
df_nor, df_process_21_22_nor = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 669)
df_sl, df_process_21_22_sl = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 776)
df_swe, df_process_21_22_swe = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 808)
df_pol, df_process_21_22_pol = generate_chemistry(squad, pos, matches_all, def_suc_tot, players_teams, sd_table, 692)

df_expected_dk = compute_expected_outputs(df_process_21_22_dk)
df_expected_eng = compute_expected_outputs(df_process_21_22_eng)
df_expected_esp = compute_expected_outputs(df_process_21_22_esp)
df_expected_ger = compute_expected_outputs(df_process_21_22_ger)
df_expected_fr = compute_expected_outputs(df_process_21_22_fr)
df_expected_por = compute_expected_outputs(df_process_21_22_por)
df_expected_ned = compute_expected_outputs(df_process_21_22_ned)
df_expected_tur = compute_expected_outputs(df_process_21_22_tur)
df_expected_bel = compute_expected_outputs(df_process_21_22_bel)
df_expected_swi = compute_expected_outputs(df_process_21_22_swi)
df_expected_gre = compute_expected_outputs(df_process_21_22_gre)
df_expected_cro = compute_expected_outputs(df_process_21_22_cro)
df_expected_cze = compute_expected_outputs(df_process_21_22_cze)
df_expected_au = compute_expected_outputs(df_process_21_22_au)
df_expected_ser = compute_expected_outputs(df_process_21_22_ser)
df_expected_rus = compute_expected_outputs(df_process_21_22_rus)
df_expected_sco = compute_expected_outputs(df_process_21_22_sco)
df_expected_hun = compute_expected_outputs(df_process_21_22_hun)
df_expected_ice = compute_expected_outputs(df_process_21_22_ice)
df_expected_sk = compute_expected_outputs(df_process_21_22_sk)
df_expected_nor = compute_expected_outputs(df_process_21_22_nor)
df_expected_sl = compute_expected_outputs(df_process_21_22_sl)
df_expected_swe = compute_expected_outputs(df_process_21_22_swe)
df_expected_pol = compute_expected_outputs(df_process_21_22_pol)


team_chem_eng = get_team_chemistry(df_eng)
team_chem_fr = get_team_chemistry(df_fr)
team_chem_ger = get_team_chemistry(df_ger)
team_chem_esp = get_team_chemistry(df_esp)
team_chem_it = get_team_chemistry(df_it)
team_chem_ned = get_team_chemistry(df_ned)
team_chem_por = get_team_chemistry(df_por)
team_chem_bel = get_team_chemistry(df_bel)
team_chem_tur = get_team_chemistry(df_tur)
team_chem_au = get_team_chemistry(df_au)
team_chem_swi = get_team_chemistry(df_swi)
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


df_merged_eng = team_chem_eng.merge(df_expected_eng, on=['teamId'])
df_merged_it = team_chem_it.merge(df_expected_it, on=['teamId'])
df_merged_fr = team_chem_fr.merge(df_expected_fr, on=['teamId'])
df_merged_ger = team_chem_ger.merge(df_expected_ger, on=['teamId'])
df_merged_esp = team_chem_esp.merge(df_expected_esp, on=['teamId'])
df_merged_ned = team_chem_ned.merge(df_expected_ned, on=['teamId'])
df_merged_por = team_chem_por.merge(df_expected_por, on=['teamId'])
df_merged_bel = team_chem_bel.merge(df_expected_bel, on=['teamId'])
df_merged_tur = team_chem_tur.merge(df_expected_tur, on=['teamId'])
df_merged_au = team_chem_au.merge(df_expected_au, on=['teamId'])
df_merged_swi = team_chem_swi.merge(df_expected_swi, on=['teamId'])
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


f_all = pd.concat([df_merged_it, df_merged_eng, df_merged_dk, df_merged_fr,
                   df_merged_ger, df_merged_ned, df_merged_por,df_merged_esp,
                   df_merged_tur, df_merged_swi,df_merged_gre, df_merged_cro,
                   df_merged_rus, df_merged_ser, df_merged_hun, df_merged_ice,
                   df_merged_nor, df_merged_sl, df_merged_sco, df_merged_pol,
                   df_merged_au, df_merged_bel, df_merged_cze, df_merged_sk,
                   df_merged_swe])

df_expected_full = pd.concat([df_expected_dk, df_expected_eng, df_expected_fr, df_expected_it,
                              df_expected_ger, df_expected_por, df_expected_ned, df_expected_bel,
                              df_expected_esp, df_expected_tur, df_expected_au, df_expected_swi,
                               df_expected_ser, df_expected_gre, df_expected_cro,
                              df_expected_pol, df_expected_cze, df_expected_rus, df_expected_sco,
                              df_expected_hun, df_expected_sk, df_expected_swe,
                              df_expected_ice, df_expected_nor, df_expected_sl
                              ])



f_all = pd.concat([df_merged_it, df_merged_eng, df_merged_dk, df_merged_fr,
                   df_merged_ger, df_merged_ned, df_merged_por,df_merged_esp,
                   df_merged_tur, df_merged_swi,df_merged_gre, df_merged_cro,
                   df_merged_rus, df_merged_ser, df_merged_hun, df_merged_ice,
                   df_merged_nor, df_merged_sl, df_merged_sco, df_merged_pol,
                   df_merged_au, df_merged_bel, df_merged_cze, df_merged_sk,
                   df_merged_swe])

corr_pd = [(df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins'])), df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins']),
           (df_merged_ned['chemistry'].corr(df_merged_ned['expected_wins'])),  df_merged_por['chemistry'].corr(df_merged_por['expected_wins']),
            df_merged_ger['chemistry'].corr(df_merged_ger['expected_wins']), df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins']),
            df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins']), df_merged_swi['chemistry'].corr(df_merged_swi['expected_wins']),
           df_merged_gre['chemistry'].corr(df_merged_gre['expected_wins']), df_merged_cro['chemistry'].corr(df_merged_cro['expected_wins']),
           df_merged_rus['chemistry'].corr(df_merged_rus['expected_wins']), df_merged_ser['chemistry'].corr(df_merged_ser['expected_wins']),
           df_merged_nor['chemistry'].corr(df_merged_nor['expected_wins']), df_merged_sl['chemistry'].corr(df_merged_sl['expected_wins']),
           df_merged_au['chemistry'].corr(df_merged_au['expected_wins']), df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins']),
           df_merged_cze['chemistry'].corr(df_merged_cze['expected_wins']), df_merged_sk['chemistry'].corr(df_merged_sk['expected_wins']),
           df_merged_swe['chemistry'].corr(df_merged_swe['expected_wins'])
           ]

corr_pd = [
    (df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins']), 'DK'),
    (df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins']), 'ENG'),
    (df_merged_ned['chemistry'].corr(df_merged_ned['expected_wins']), 'NED'),
    (df_merged_por['chemistry'].corr(df_merged_por['expected_wins']), 'POR'),
    (df_merged_ger['chemistry'].corr(df_merged_ger['expected_wins']), 'GER'),
    (df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins']), 'FR'),
    (df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins']), 'TUR'),
    (df_merged_swi['chemistry'].corr(df_merged_swi['expected_wins']), 'SWI'),
    (df_merged_gre['chemistry'].corr(df_merged_gre['expected_wins']), 'GRE'),
    (df_merged_cro['chemistry'].corr(df_merged_cro['expected_wins']), 'CRO'),
    (df_merged_rus['chemistry'].corr(df_merged_rus['expected_wins']), 'RUS'),
    (df_merged_ser['chemistry'].corr(df_merged_ser['expected_wins']), 'SER'),
    (df_merged_nor['chemistry'].corr(df_merged_nor['expected_wins']), 'NOR'),
    (df_merged_sl['chemistry'].corr(df_merged_sl['expected_wins']), 'SL'),
    (df_merged_au['chemistry'].corr(df_merged_au['expected_wins']), 'AU'),
    (df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins']), 'BEL'),
    (df_merged_cze['chemistry'].corr(df_merged_cze['expected_wins']), 'CZE'),
    (df_merged_sk['chemistry'].corr(df_merged_sk['expected_wins']), 'SK'),
    (df_merged_swe['chemistry'].corr(df_merged_swe['expected_wins']), 'SWE'),
    (df_merged_ice['chemistry'].corr(df_merged_ice['expected_wins']), 'ICE')
]

plot_relationship_chemistry(df_merged_eng, 'ENG')
plot_relationship_chemistry(df_merged_it, 'IT')
plot_relationship_chemistry(df_merged_fr, 'FR')
plot_relationship_chemistry(df_merged_ger, 'GER')
plot_relationship_chemistry(df_merged_por, 'POR')
plot_relationship_chemistry(df_merged_ned, 'NED')
plot_relationship_chemistry(df_merged_bel, 'BEL')
plot_relationship_chemistry(df_merged_esp, 'expected_wins')
plot_relationship_chemistry(df_merged_tur, 'TUR')
plot_relationship_chemistry(f_all, 'expected_wins')

'''

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

#Leagues-wise examinations
'''
print(df_merged_dk['chemistry'].corr(df_merged_dk['expected_wins']))
print(df_merged_eng['chemistry'].corr(df_merged_eng['expected_wins']))
print(df_merged_ned['chemistry'].corr(df_merged_ned['expected_wins']))
print(df_merged_ger['chemistry'].corr(df_merged_ger['expected_wins']))
print(df_merged_fr['chemistry'].corr(df_merged_fr['expected_wins']))
print(df_merged_it['chemistry'].corr(df_merged_it['expected_wins']))
print(df_merged_bel['chemistry'].corr(df_merged_bel['expected_wins']))
print(df_merged_esp['chemistry'].corr(df_merged_esp['expected_wins']))
print(df_merged_tur['chemistry'].corr(df_merged_tur['expected_wins']))
#Out
#df_merged_au['chemistry'].corr(df_merged_au['expected_wins'])

print(df_merged_swi['chemistry'].corr(df_merged_swi['expected_wins']))
#Out
#print(df_merged_cyo['chemistry'].corr(df_merged_cyo['expected_wins']))

print(df_merged_ser['chemistry'].corr(df_merged_ser['expected_wins']))
print(df_merged_gre['chemistry'].corr(df_merged_gre['expected_wins']))

print(df_merged_cro['chemistry'].corr(df_merged_cro['expected_wins']))
print(df_merged_pol['chemistry'].corr(df_merged_pol['expected_wins']))

#print(df_merged_cze['chemistry'].corr(df_merged_cze['expected_wins']))

print(df_merged_rus['chemistry'].corr(df_merged_rus['expected_wins']))
print(df_merged_sco['chemistry'].corr(df_merged_sco['expected_wins']))
print(df_merged_hun['chemistry'].corr(df_merged_hun['expected_wins']))
print(df_merged_sk['chemistry'].corr(df_merged_sk['expected_wins']))

'''

#Experimental procedures for outlier removeal
'''
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
    print()'''