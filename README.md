## CS-109A Data Science Final project
**Harvard University** <br>
Summer 2018 <br>
Team #12: Bruno Janota and Hans Chacko  <br>


[Background](#background)<br>
[Project Overview](#project-overview)<br>
[Part 1: Baseline Model](#part-1-baseline-model)<br>
[Part 2: Developing Model with Elo Ranking](#elo-ranking-system)<br>
[Part 3: Ensemble Models](#part-3-ensemble-models)<br>
[Part 4: Predicting the 2018 World Cup](#part-4-predicting-the-2018-world-cup)<br>


# Background

The goal of this project will be to leverage various sources of team and player data in addition to 
historical match results to construct a 2018 FIFA World Cup prediction model and evaluate those models 
against the baseline of predictions from simply incorporating FIFA ranking as a measure of team strength. <br>

The FIFA World Cup is one of the most prominent sporting events in the world yet relative to
other major sports, football/futbol/soccer analytics generally lags the state of the art for sports
analytics. There are many reasons why national team soccer analytics trails other major sports
like baseball basketball and American football. For example, many national team matches are
against regional opponents as part of regional tournaments or world cup qualifiers and a
disproportional amount of the available data is from segmented sources therefore, compiling
something as simple as a relative ranking among all nations can be quite challenging.<br>

FIFA’s response to this problem was the introduction of an official FIFA ranking for each
member nation beginning in December 1992. The initial rankings were met with criticism and
significant changes were implemented in January 1999 and July 2006. The initial ranking
comprised of a team receiving one point for a draw or three for a victory in FIFA-recognized
matches like traditional league scoring, but the method proved to be overly simplistic for
international comparisons. The 1999 ranking system update scaled the point ranking by a factor
of 10 and considered the number of goals scored/conceded, whether the match was home or
away, the importance of the match and regional strength which enabled match losers to earn
points and a fixed number of points were no longer necessarily awarded for a victory or a draw.
The rankings were again met with criticism and in 2006, updates were introduced to reduce the
evaluation period to 4 years vs. 8 years, revise the match importance parameters, and ignore the
goals scored and home or away advantage.<br>

Additionally, it was recently announced that following the 2018 World Cup, the FIFA world
rankings would adopt an Elo based ranking system which considers many factors including the
teams previous rating, the status (in order of importance: World Cup/Olympic games, continental
championship and intercontinental tournaments, world cup qualifiers, all other tournaments, and
lastly friendlies), goal difference, result, and expected result of the match. Since this ranking has 
not yet been introduced the basis of this analysis will compare the results to the pre-2018
rankings.<br>


# Project Overview
The goal of this project will be to leverage various sources of team/player data, historical international match results, historical FIFA rankings, and betting odds to construct a 2018 FIFA World Cup prediction model. The model will predict a win, loss, or draw match outcome for historical international matches from 1993 to 2018 divided into a training and testing set and evaluate those predictions against the baseline of predictions from simply incorporating FIFA ranking as a naïve measure of team strength (i.e. the team with a higher ranking should win). In addition to predicting historical international matches, the classification model will also be used to predict the outcome of the 2018 FIFA World Cup.<br><br>

The project will be split into three parts:<br><br>
Part 1 : Baseline Model <br>
Combine the international match outcome and FIFA rankings data sets for matches between 1993 and 2018. Perform some basic feature engineering to add additional features. Lastly, perform a 70/30% train/test set split and evaluate the performance of a Random Forest model on the overall classification accuracy on the test set. This will be the baseline model.<br><br>
Part 2 : Developing Model with Elo Ranking <br><br>
Replace the FIFA rankings in Part 1 with an Elo based scoring model and assess improvement, if any, on the overall classification accuracy on the same test set used in part 1. <br><br>
Part 3 : Ensemble Models <br>
The optimal combination of features from parts 1 and 2 (may include FIFA rank, Elo score, or both) will be used to train a variety of classification models (Random Forest, xgboost, LDA, QDA, KNN, etc.). The probabilistic results for each match (win, tie, loss) will be blended with the results of a Poisson Distribution model that uses the complete player ranking data scraped from sofifa.com for the FIFA 2018 video game to predict the group stages of the 2018 FIFA World Cup. <br><br>
Part 4 : Predicting the 2018 World Cup<br>
The knockout stages of the world cup will be simulated 1000 times to determine the probability of each team winning the tournament. Matches that result in a tie during the knockout stages will take into account the average penalty rating of the top 5 penalty shooters for each time from the sofifa.com data set to break the tie. The final result will be the probability of each of the 32 teams that qualified for the 2018 World Cup to win the tournament.<br><br>

# Part 1: Baseline Model
## Load, Clean and Merge Data

### Comments
1. Load historical FIFA rankings (1993-present) dataset into pandas dataframe
```python
# load historical FIFA rankings (1993-present) dataset into pandas dataframe
fifaRank_df = pd.read_csv('FifaRanking1993to2018_Tadhg Fitzgerald.csv')
fifaRank_df = fifaRank_df.loc[:,['rank', 'country_full', 'country_abrv', 'rank_date', 'total_points', 'previous_points',
                                 'cur_year_avg_weighted', 'two_year_ago_weighted', 'three_year_ago_weighted']]
fifaRank_df = fifaRank_df.replace({"IR Iran": "Iran"})
fifaRank_df['rank_date'] = pd.to_datetime(fifaRank_df['rank_date'])
fifaRank_df.head()
```
![Historical Rankings](/Images/HistoricalRankings.PNG)

### Comments
1. Load international match results (1872-2018) dataset into pandas dataframe


```python
results_df = pd.read_csv('InternationalResultsFrom1993to2018.csv')
results_df =  results_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
results_df['date'] = pd.to_datetime(results_df['date'])
results_df.head()

results_df = pd.read_csv('InternationalResultsFrom1993to2018.csv')
results_df =  results_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
results_df['date'] = pd.to_datetime(results_df['date'])
```

![Historical Results](/Images/HistoricalResults.PNG)

### Comments
1. Load country region and income group data into pandas dataframe
2. Convert categorical variables using one-hot encoding

```python
country_df = pd.read_csv('WorldCountryData.csv')
country_df = pd.get_dummies(country_df, columns=['Region','IncomeGroup'], drop_first=True)
country_df.head()
```

![World Country Data](/Images/WorldCountryData.PNG)

### Comments
1. Get ranks for every day
2. Join the ranks
3. Join region and income group data

```python
fifaRank_df = fifaRank_df.set_index(['rank_date'])\
                        .groupby(['country_full'], group_keys=False)\
                        .resample('D').first()\
                        .fillna(method='ffill')\
                        .reset_index()

results_df = results_df.merge(fifaRank_df, 
                        left_on=['date', 'home_team'], 
                        right_on=['rank_date', 'country_full'])
results_df = results_df.merge(fifaRank_df, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))

results_df = results_df.merge(country_df, 
                        left_on=['home_team'], 
                        right_on=['ShortName'])
results_df = results_df.merge(country_df, 
                        left_on=['away_team'], 
                        right_on=['ShortName'], 
                        suffixes=('_home', '_away'))

```

### Comments
Generate additional features

```python
results_df['rank_difference'] = results_df['rank_home'] - results_df['rank_away']
results_df['average_rank'] = (results_df['rank_home'] + results_df['rank_away'])/2
results_df['score_difference'] = results_df['home_score'] - results_df['away_score']
results_df['is_stake'] = results_df['tournament'] != 'Friendly'
results_df['total_goals'] = results_df['home_score'] + results_df['away_score'] 
results_df['year'] = results_df['date'].dt.year

winner = []
result = []
for i in range (len(results_df['home_team'])):
    if results_df['home_score'][i] > results_df['away_score'][i]:
        winner.append(results_df['home_team'][i])
        result.append(1.0)
    elif results_df['home_score'][i] < results_df ['away_score'][i]:
        winner.append(results_df['away_team'][i])
        result.append(0.0)
    else:
        winner.append('Draw')
        result.append(0.5)

results_df['winning_team'] = winner
results_df['result'] = result

```

### Comments
1. Plot International Matches played between 1993-2018: 14997
2. Total International Matches played by 2018 World Cup teams between 1993-2018: 14997

```python
plt.figure(figsize=(12, 8), dpi= 80)
sns.set(style='darkgrid')
sns.countplot(x='result', data = results_df)
plt.xticks(range(3), ['Loss','Tie','Win'])
plt.title('Home Team International Match Outcomes', fontsize=14)
plt.show()
```

![Home Team International Matches Outcomes](/Images/HomeTeamInternationalMatchOutcomes.png)

### Comments
Plot Number of International Matches per Year

```python
games_per_year = results_df.groupby(['year'])['year'].count()
years = ['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003',
       '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018']
plt.figure(figsize=(12, 8), dpi= 80)
plt.bar(range(len(games_per_year)), games_per_year)
plt.xticks(range(len(games_per_year)), years, rotation = 90)
plt.title('Number of International Matches per Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('# of Matches', fontsize=12)
plt.show()
```

![International Matches Per Year](/Images/IntlMatchesYear.png)


## Naive Model based on FIFA Ranking Only

### Comments
1. Accuracy of FIFA Ranking at Predicting Winner: 21.88%<br>

```python
results_df['fifa_correct_withDraws'] = ((results_df.home_score >= results_df.away_score) & (results_df.rank_home > results_df.rank_away)) | ((results_df.away_score >= results_df.home_score) & (results_df.rank_away > results_df.rank_home))
results_df['fifa_correct'] = ((results_df.home_score > results_df.away_score) & (results_df.rank_home > results_df.rank_away)) | ((results_df.away_score > results_df.home_score) & (results_df.rank_away > results_df.rank_home))

print('Accuracy of FIFA Ranking at Predicting Winner: {}%'
      .format(round(results_df.fifa_correct.sum()/results_df.fifa_correct.count()*100, 2)))
```

### Comments
1. Accuracy of FIFA Ranking at Predicting Winner (ignoring draws): 28.9%

```python
no_Draws_df = results_df.loc[results_df['result'] != 0.5].copy()
no_Draws_df['fifa_correct_withDraws'] = ((no_Draws_df.home_score >= no_Draws_df.away_score) & (no_Draws_df.rank_home > no_Draws_df.rank_away)) | ((no_Draws_df.away_score >= no_Draws_df.home_score) & (no_Draws_df.rank_away > no_Draws_df.rank_home))
no_Draws_df['fifa_correct'] = ((no_Draws_df.home_score > no_Draws_df.away_score) & (no_Draws_df.rank_home > no_Draws_df.rank_away)) | ((no_Draws_df.away_score > no_Draws_df.home_score) & (no_Draws_df.rank_away > no_Draws_df.rank_home))
print('Accuracy of FIFA Ranking at Predicting Winner (ignoring draws): {}%'
      .format(round(no_Draws_df.fifa_correct.sum()/no_Draws_df.fifa_correct.count()*100, 2)))
```

## Visualize Subset of Data for Portugal Matches Only
1. Let's work with a subset of the data that includes games played by Portugal
2. Visualize Portugal game outcomes

### Comments
```python
portugal = results_df[(results_df['home_team'] == 'Portugal') | (results_df['away_team'] == 'Portugal')]

wins = []
for row in portugal['winning_team']:
    if row != 'Portugal' and row != 'Draw':
        wins.append('Loss')
    elif row == 'Draw':
        wins.append('Draw')
    else:
        wins.append('Win')
        
winsdf= pd.DataFrame(wins, columns=['Portugal Results'])

plt.figure(figsize=(12, 8), dpi= 80)
sns.set(style='darkgrid')
sns.countplot(x='Portugal Results', data=winsdf)
```
![Portugal Matches](/Images/Portugal.png))

## Evaluate Baseline Decision Tree Model 

### Comments
1. Remove features without predictive value or not known prior to game start
2. Split training set in features/labels
3. Split the data into training and testing sets
4. Show the number of observations for the test and training dataframes

```python
no_draw_train_df = results_df[results_df['result'] != 0.5]

train_df = no_draw_train_df[['rank_home','rank_away','neutral','previous_points_home','cur_year_avg_weighted_home',
                     'two_year_ago_weighted_home','three_year_ago_weighted_home','previous_points_away',
                     'cur_year_avg_weighted_away','two_year_ago_weighted_away','three_year_ago_weighted_away',
                     'rank_difference','average_rank','is_stake','result']]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, 
                                                                            random_state = 42)
print('Number of observations in the training data:', len(train_labels))
print('Number of observations in the test data:',len(test_labels))
```

Number of observations in the training data: 8513
Number of observations in the test data: 2838

### Comments
1. Determine best tree depth  
2. Plot classification accuracy vs tree depth

```python
tree_depth = list(range(1,25))

cv_scores_mean = []
cv_scores_std = []
for depth in tree_depth:
    baselineModel = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(baselineModel, train_features, train_labels, cv=5, scoring='accuracy')
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())
    
# determining best tree depth
optimal_depth = tree_depth[cv_scores_mean.index(max(cv_scores_mean))]
print('The optimal tree depth is: {}'.format(optimal_depth))

# plot classification accuracy vs tree depth
plt.rcParams['figure.figsize'] = (10,8)
plt.plot(tree_depth, cv_scores_mean)
plt.fill_between(tree_depth, (np.array(cv_scores_mean) - 2*np.array(cv_scores_std)), 
                 (np.array(cv_scores_mean) + 2*np.array(cv_scores_std)), color = [0.7, 0.7, 0.7], alpha = 0.5)
plt.title('Decision Tree Classification Accuracy vs. Tree Depth (with +/- 2 sd band)')
plt.xlabel('Tree Depth')
plt.ylabel('Classification Accuracy')
plt.show()
```
The optimal tree depth is: 5

![Tree Depth](/Images/TreeDepth.png)

### Comments
1. Use cross-validation to build baseline model

```python
baselineModel = DecisionTreeClassifier(max_depth=optimal_depth).fit(train_features, train_labels)

print('Decision Tree Classifier (FIFA rank only):')
print('Classification Accuracy on training set: {}%'.format(round(baselineModel.score(train_features, train_labels,)*100, 2)))
print('Classification Accuracy on testing set: {}%\n'.format(round(baselineModel.score(test_features, test_labels,)*100, 2)))
```

Decision Tree Classifier (FIFA rank only):
Classification Accuracy on training set: 75.17%
Classification Accuracy on testing set: 74.21%

### Comments
1. View the decision tree for baseline model

```python
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import os
import sys

def conda_fix(graph):
    path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
    paths = ("dot", "twopi", "neato", "circo", "fdp")
    paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
    graph.set_graphviz_executables(paths)

dot_data = StringIO()
export_graphviz(baselineModel, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
conda_fix(graph)

# View the decision tree for baseline model
Image(graph.create_png())
```

![Baseline Tree](/Images/BaselineTree.png)

# Part 2: Developing Model with Elo Ranking

## Elo Ranking System
The Elo rating system is a method for calculating the relative skill levels of teams (or players) in zero-sum games. From Wikipedia, a team's Elo rating as applied to our data set is represented by a number which increases or decreases depending on the outcome of matches between international teams. After every game, the winning team takes points from the losing one. The difference between the ratings of the winner and loser determines the total number of points gained or lost after a game. In a series of games between a high-rated team and a low-rated team, the high-rated team is expected to score more wins. If the high-rated team wins, then only a few rating points will be taken from the low-rated team. However, if the lower rated team scores an upset win, many rating points will be transferred. The lower rated team will also gain a few points from the higher rated team in the event of a draw. This means that this rating system is self-correcting. A team whose rating is too low should, in the long run, do better than the rating system predicts, and thus gain rating points until the rating reflects their true playing strength. The elo function that we used resembles the Elo World Ranking which is defined below:

![Elo1](/Images/elo1.PNG)

![Elo2](/Images/elo2.PNG)

![Elo3](/Images/elo3.PNG)

## Comments
1. Set some constants
2. Define function to Update ELO
3. Define function to get Expected result

```python
mean_elo = 1500
elo_width = 400

def update_elo(home_elo, away_elo, match_result, match_type, home_score, away_score):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    if(match_type == 'FIFA World Cup'):
        k_factor = 60.0
    elif(match_type == 'Friendly'):
        k_factor = 20.0
    else:
        k_factor = 40.0    
    
    if(home_score == away_score):
        goal_factor = 1.0
    elif(np.abs(home_score - away_score) == 2):
        goal_factor = 1.5
    else:
        goal_factor = (11 + np.abs(home_score - away_score))/8        
    
    expected_win = expected_result(home_elo, away_elo)
    change_in_elo = k_factor * goal_factor * (match_result - expected_win)
    home_elo += change_in_elo
    away_elo -= change_in_elo
    
    return home_elo, away_elo

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

```

## Compute Elo Rating
The loop where it happens
- We go through each row in the DataFrame.
- We look up the current Elo rating of both teams.
- We calculate the expected wins for the team that actually won.
- Write Elo before and after the game in the Data Frame.
- Update the Elo rating for both teams in the "current_elos" list.

```python
current_season = elo_games_df.date[:0].dt.year
for row in elo_games_df.itertuples():
       
    idx = row.Index
    home_team = row.home_team
    away_team = row.away_team
  
    # Get pre-match ratings
    teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
    teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

    # Update on game results
    home_elo_after, away_elo_after = update_elo(teamA_elo, teamB_elo, row.result, row.tournament, 
                                                row.home_score, row.away_score)
        
    # Save updated elos
    elo_games_df.at[idx, 'home_elo_before_game'] = teamA_elo
    elo_games_df.at[idx, 'away_elo_before_game'] = teamB_elo
    elo_games_df.at[idx, 'home_elo_after_game'] = home_elo_after
    elo_games_df.at[idx, 'away_elo_after_game'] = away_elo_after
    
    # update current elos
    elo_teams_df.set_value(elo_teams_df.loc[elo_teams_df.team == home_team].index.values[0], 'elo', home_elo_after)
    elo_teams_df.set_value(elo_teams_df.loc[elo_teams_df.team == away_team].index.values[0], 'elo', away_elo_after)

elo_games_df.head()

elo_teams_df.sort_values(['elo'], ascending=False).head(10)

```
### Comments
1. Capture top 5 teams and plot elo change with time for each team

```python
elo_top5 = elo_teams_df.sort_values(['elo'], ascending=False).head(5)['team'].values.tolist()

fig, axs = plt.subplots(5,1, figsize=(12, 26), facecolor='w', edgecolor='k')
axs = axs.ravel()

for i, team in enumerate(elo_top5):
    team_df = elo_games_df[(elo_games_df['home_team'] == team) | (elo_games_df['away_team'] == team)] 
    
    elo_over_time = []
    for row in team_df.itertuples():
        if(row.home_team == team):
            elo_over_time.append(row.home_elo_before_game)
        else:
            elo_over_time.append(row.away_elo_before_game)

    axs[i].plot(elo_over_time)
    axs[i].set_ylabel('Elo Points', fontsize=12)
    axs[i].set_title('{} Elo (1993-2018)'.format(team))
    axs[i].set_xlabel('Match Number', fontsize=12) 

plt.show()
```

The elo ratings for the top 5 rated teams going into the 2018 World Cup are as follows:
![Elo4](/Images/elo4.PNG)
![Elo5](/Images/elo5.PNG)

## Comments
1. Join Elo Scores with World Cup games dataset
2. Plot Fifa Rank Difference vs Match Outcome

```python
results_df['index'] = range(len(results_df))
train_results_df = results_df.merge(elo_games_df, 
                        left_on='index', 
                        right_on='index')
                        
plt.rcParams['figure.figsize'] = (15,10)
train_results_df.plot.scatter('index','rank_difference', c='result_x', colormap='jet', alpha=0.5,
                             title = 'FIFA Rank Difference vs. Match Outcome')
plt.show()
                        
```
![Elo6](/Images/elo6.PNG)

### Comments
1. Number of Upsets based on FIFA Rank: 53.7%<br>
2. Number of Upsets based on Elo Score: 20.3%<br>

```python
train_results_df['fifa_upsets'] = ((train_results_df.home_score_x > train_results_df.away_score_x) & \
                              (train_results_df.rank_home < train_results_df.rank_away)) | \
                              ((train_results_df.away_score_x > train_results_df.home_score_x) & \
                               (train_results_df.rank_away < train_results_df.rank_home))
    
train_results_df['elo_upsets'] = ((train_results_df.home_score_x > train_results_df.away_score_x) & \
                              (train_results_df.home_elo_before_game < train_results_df.away_elo_before_game)) | \
                              ((train_results_df.away_score_x > train_results_df.home_score_x) & \
                               (train_results_df.away_elo_before_game < train_results_df.home_elo_before_game))
        
print('Number of Upsets based on FIFA Rank: {}%'
      .format(round(train_results_df.fifa_upsets.sum()/train_results_df.fifa_upsets.count()*100, 1)))

print('Number of Upsets based on Elo Score: {}%'
      .format(round(train_results_df.elo_upsets.sum()/train_results_df.elo_upsets.count()*100, 1)))

```


Number of Upsets based on FIFA Rank: 53.7%
Number of Upsets based on Elo Score: 20.3%

## Comment
1. Generate Additional Features
2. Remove features without predictive value or not known prior to game start
3. Split training set in features/labels
4. Using Skicit-learn to split data into training and testing sets
5. Split the data into training and testing sets
6. Build model and capture training and test accuracies

```python
# generate additional features
train_results_df['elo_difference'] = train_results_df['home_elo_before_game'] - train_results_df['away_elo_before_game']
train_results_df['average_elo'] = (train_results_df['home_elo_before_game'] + train_results_df['away_elo_before_game'])/2

no_draw_train_df = train_results_df[train_results_df['result_x'] != 0.5]

# remove features without predictive value or not known prior to game start
train_elo_df = no_draw_train_df[['neutral','home_elo_before_game','away_elo_before_game','is_stake','elo_difference',
                                 'average_elo','Region_Europe & Central Asia_home','Region_Latin America & Caribbean_home',
                                 'Region_Middle East & North Africa_home','Region_North America_home','Region_South Asia_home',
                                 'Region_Sub-Saharan Africa_home','IncomeGroup_High income: nonOECD_home',
                                 'IncomeGroup_Low income_home','IncomeGroup_Lower middle income_home',
                                 'IncomeGroup_Upper middle income_home','Region_Europe & Central Asia_away',
                                 'Region_Latin America & Caribbean_away','Region_Middle East & North Africa_away',
                                 'Region_North America_away','Region_South Asia_away','Region_Sub-Saharan Africa_away',
                                 'IncomeGroup_High income: nonOECD_away','IncomeGroup_Low income_away',
                                 'IncomeGroup_Lower middle income_away','IncomeGroup_Upper middle income_away','result_x']]

# split training set in features/labels
features = train_elo_df.drop(['result_x'], axis = 1)
labels = np.asarray(train_elo_df['result_x'], dtype="|S6")

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, 
                                                                            random_state = 42)
                                                                            
                                                                            baselineModel = DecisionTreeClassifier(max_depth=5).fit(train_features, train_labels)

print('Decision Tree Classifier:')
print('Classification Accuracy on training set: {}%'.format(round(baselineModel.score(train_features, train_labels,)*100, 2)))
print('Classification Accuracy on testing set: {}%\n'.format(round(baselineModel.score(test_features, test_labels,)*100, 2)))

```

Decision Tree Classifier:<br>
Classification Accuracy on training set: 74.12%<br>
Classification Accuracy on testing set: 72.69%<br>

# Part 3: Ensemble Models

## Ensemble Learners

The optimal combination of features from parts 1 and 2 (may include FIFA rank, Elo score, or both) will be used to train a variety of classification models (Random Forest, xgboost, LDA, QDA, KNN, etc.). The probabilistic results for each match (win, tie, loss) will be blended with the results of a Poisson Distribution model that uses the complete player ranking data scraped from sofifa.com for the FIFA 2018 video game to predict the group stages of the 2018 FIFA World Cup. The knockout stages of the world cup will be simulated 1000 times to determine the probability of each team winning the tournament. Matches that result in a tie during the knockout stages will take into account the average penalty rating of the top 5 penalty shooters for each time from the sofifa.com data set to break the tie. The final result will be the probability of each of the 32 teams that qualified for the 2018 World Cup to win the tournament.

## RandomForestClassifier

### Comments

```python
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 500 decision trees
rf = RandomForestClassifier(n_estimators = 500, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest predict method on the test data
predictions = rf.predict(test_features)
# View the predicted probabilities of the first 10 observations
rf.predict_proba(test_features)[0:10]
# Create confusion matrix
pd.crosstab(test_labels, predictions, rownames=['Actual Outcome'], colnames=['Predicted Outcome']).apply(lambda r: r/r.sum(), axis=1)
from sklearn.metrics import accuracy_score
```

![Random Forest Accuracy Score](/Images/model1.PNG)

### Comments
1. Test Accuracy

```python
# Test Accuracy
print('Test Accuracy: {}'.format(accuracy_score(test_labels, predictions)))
```
Test Accuracy: 0.7258632840028189

### Important Features

```python
# get feature names
feature_list = list(features.columns)

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances];
```

Variable: elo_difference                 Importance: 0.26<br>
Variable: away_elo_before_game           Importance: 0.19<br>
Variable: home_elo_before_game           Importance: 0.18<br>
Variable: average_elo                    Importance: 0.16<br>
Variable: neutral                        Importance: 0.02<br>
Variable: is_stake                       Importance: 0.02<br>
Variable: IncomeGroup_Upper middle income_home Importance: 0.02<br>
Variable: IncomeGroup_Upper middle income_away Importance: 0.02<br>
Variable: Region_Europe & Central Asia_home Importance: 0.01<br>
Variable: Region_Latin America & Caribbean_home Importance: 0.01<br>
Variable: Region_Middle East & North Africa_home Importance: 0.01<br>
Variable: Region_Sub-Saharan Africa_home Importance: 0.01<br>
Variable: IncomeGroup_High income: nonOECD_home Importance: 0.01<br>
Variable: IncomeGroup_Low income_home    Importance: 0.01<br>
Variable: IncomeGroup_Lower middle income_home Importance: 0.01<br>
Variable: Region_Europe & Central Asia_away Importance: 0.01<br>
Variable: Region_Latin America & Caribbean_away Importance: 0.01<br>
Variable: Region_Middle East & North Africa_away Importance: 0.01<br>
Variable: Region_Sub-Saharan Africa_away Importance: 0.01<br>
Variable: IncomeGroup_High income: nonOECD_away Importance: 0.01<br>
Variable: IncomeGroup_Low income_away    Importance: 0.01<br>
Variable: IncomeGroup_Lower middle income_away Importance: 0.01<br>
Variable: Region_North America_home      Importance: 0.0<br>
Variable: Region_South Asia_home         Importance: 0.0<br>
Variable: Region_North America_away      Importance: 0.0<br>
Variable: Region_South Asia_away         Importance: 0.0<br>

### PCA

```python
from sklearn.decomposition import PCA

pca_5_transformer = PCA(5).fit(train_features)
np.cumsum(pca_5_transformer.explained_variance_ratio_)
```

array([0.56661497, 0.99997483, 0.99997933, 0.99998319, 0.99998541])

### Comments
1. Plot boundaries based on PCA transformation

```python
pca_transformer = PCA(2).fit(train_features)
x_train_2d = pca_transformer.transform(train_features)
x_test_2d = pca_transformer.transform(test_features)

# lists to track each group's plotting color and label
colors = ['r', 'g', 'b']
label_text = ['Loss', 'Draw', 'Win']

plt.rcParams['figure.figsize'] = (10,8)

# loop over the different groups
for result in [0.0,0.5,1]:
    result_df = x_train_2d[train_labels.astype(np.float) == result]
    plt.scatter(result_df[:,0], result_df[:,1], c = colors[int(2*result)], label=label_text[int(2*result)])
    
# add labels
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimention 2")
plt.legend()
plt.show()
```
![PCA](/Images/model2.PNG)

### KNN

### Comments
1. Find optimum number of neighbours

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

neighbors = list(range(1,41))
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_features, train_labels, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# determining best k
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print('The optimal number of neighbors is: {}'.format(optimal_k))

# plot classification accuracy vs k
plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Classification Accuracy')
plt.show()
```
The optimal number of neighbors is: 38

![KNN Neighbours](/Images/model3.PNG)

### XGBoost

### Comments
1. Find Optimum Depth
2. Plot depths

```python
from xgboost import XGBClassifier

depth = list(range(1,11))
cv_scores = []
for d in depth:
    xgb = XGBClassifier(max_depth=d)
    scores = cross_val_score(xgb, train_features, train_labels, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# determining best depth
optimal_depth = depth[cv_scores.index(max(cv_scores))]
print('The optimal max depth for XGB model is: {}'.format(optimal_depth))

# plot classification accuracy vs depth
plt.plot(depth, cv_scores)
plt.xlabel('Max Tree Depth')
plt.ylabel('Classification Accuracy')
plt.show()
```

The optimal max depth for XGB model is: 3

![XGBoost](/Images/model4.PNG)

### Build Ensemble

### Comments
1. Build Models : LDA, QDA, KNN, XGB
2. Get accuracy of each

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

lda = LinearDiscriminantAnalysis().fit(train_features, train_labels)
qda = QuadraticDiscriminantAnalysis().fit(train_features, train_labels)
knn = KNeighborsClassifier(n_neighbors=23).fit(train_features, train_labels)
xgb = XGBClassifier(max_depth=2).fit(train_features, train_labels)

print('LDA Test Accuracy: {}%'.format(round(lda.score(test_features, test_labels)*100,2)))
print('QDA Test Accuracy: {}%'.format(round(qda.score(test_features, test_labels)*100,2)))
print('KNN Test Accuracy: {}%'.format(round(knn.score(test_features, test_labels)*100,2)))
print('XGB Test Accuracy: {}%'.format(round(xgb.score(test_features, test_labels)*100,2)))
```
LDA Test Accuracy: 74.14%<br>
QDA Test Accuracy: 55.39%<br>
KNN Test Accuracy: 72.13%<br>
XGB Test Accuracy: 74.28%<br>

### Comments
1. Assemble model predictions to train and test model dataframes
2. Augment training and test dataframes to corresponding prediction model dataframe

```python
model_names = ['lda', 'qda', 'knn', 'xgb']
models = [lda, qda, knn, xgb]

ensemble_tune = []
ensemble_test = []

for i in models:
    ensemble_tune.append(i.predict(x_train_2d))
    ensemble_test.append(i.predict(x_test_2d))

ensemble_tune = np.array(ensemble_tune).reshape(4, len(x_train_2d)).T
ensemble_test = np.array(ensemble_test).reshape(4, len(x_test_2d)).T

# Convert ensemble tune/test to dataframes to concatenate
ensemble_tune_df = pd.DataFrame(np.vstack(ensemble_tune), columns = model_names)
ensemble_test_df = pd.DataFrame(np.vstack(ensemble_test), columns = model_names)
x_tune = pd.DataFrame(x_train_2d, columns = ['PCA1', 'PCA2'])
x_test = pd.DataFrame(x_test_2d, columns = ['PCA1', 'PCA2'])

# Concatenate x_tune/x_test with ensemble_tune/test
augmented_tune = pd.concat([x_tune, ensemble_tune_df], axis=1, join_axes=[x_tune.index])
augmented_test = pd.concat([x_test, ensemble_test_df], axis=1, join_axes=[x_test.index])

augmentedModel_dt = DecisionTreeClassifier(max_depth=4).fit(augmented_tune, train_labels)

print('Augmented Decision Meta-Tree Classifier:')
print('Classification Accuracy on test set: {}%\n'.format(round(augmentedModel_dt.score(augmented_test, test_labels)*100, 2)))
```

Augmented Decision Meta-Tree Classifier:<br>
Classification Accuracy on test set: 72.45%<br>

### Comments
1. Rebuild model this time without QDA
2. Augment training and test dataframes to corresponding prediction model dataframe

```python
model_names = ['lda', 'knn', 'xgb']
models = [lda, knn, xgb]

ensemble_tune = []
ensemble_test = []

for i in models:
    ensemble_tune.append(i.predict(train_features))
    ensemble_test.append(i.predict(test_features))

ensemble_tune = np.array(ensemble_tune).reshape(len(models), len(train_features)).T
ensemble_test = np.array(ensemble_test).reshape(len(models), len(test_features)).T

# Convert ensemble tune/test to dataframes to concatenate
ensemble_tune_df = pd.DataFrame(np.vstack(ensemble_tune), columns = model_names)
ensemble_test_df = pd.DataFrame(np.vstack(ensemble_test), columns = model_names)

# Concatenate x_tune/x_test with ensemble_tune/test
train_features = train_features.reset_index(drop = True)
test_features = test_features.reset_index(drop = True)

augmented_tune = pd.concat([train_features, ensemble_tune_df], axis=1, join_axes=[train_features.index])
augmented_test = pd.concat([test_features, ensemble_test_df], axis=1, join_axes=[test_features.index])

augmentedModel_dt = DecisionTreeClassifier(max_depth=4).fit(augmented_tune, train_labels)

print('Augmented Decision Meta-Tree Classifier:')
print('Classification Accuracy on test set: {}%\n'.format(round(augmentedModel_dt.score(augmented_test, test_labels)*100, 2)))
```

Augmented Decision Meta-Tree Classifier:
Classification Accuracy on test set: 73.47%

# Part 4: Predicting the 2018 World Cup 

## Load 2018 World Cup Schedule with Results

### Comments
1. Load World Cup Schedule with Results into a dataframe
```python
wc2018_df = pd.read_csv('2018WorldCupGroupSchedule-withResults.csv')
wc2018_df['Date'] = pd.to_datetime(wc2018_df['Date'])
wc2018_df.head()
```

![World Cup Results](/Images/wc1.PNG)

## Feature Engineering

### Comments
1. Prepare Dataframes for Elo Rankings
2. Update Elos
3. Join region and income group data
4. Add additional features to match the test set
5. Capture Classification Accuracy of 2018 Group Stage

```python
# prepare dataframes for Elo Rankings
wc2018_elo_df = wc2018_df[['home_team','away_team','home_score','away_score','tournament','neutral', \
                           'is_stake','result']].reset_index()
wc2018_elo_df['home_elo_before_game'] = 0
wc2018_elo_df['home_elo_after_game'] = 0
wc2018_elo_df['away_elo_before_game'] = 0
wc2018_elo_df['away_elo_after_game'] = 0
wc2018_elo_df.head()

for row in wc2018_elo_df.itertuples():
       
    idx = row.Index
    home_team = row.home_team
    away_team = row.away_team
  
    # Get pre-match ratings
    teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
    teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

    # Update on game results
    home_elo_after, away_elo_after = update_elo(teamA_elo, teamB_elo, row.result, row.tournament, 
                                                row.home_score, row.away_score)
        
    # Save updated elos
    wc2018_elo_df.at[idx, 'home_elo_before_game'] = teamA_elo
    wc2018_elo_df.at[idx, 'away_elo_before_game'] = teamB_elo
    wc2018_elo_df.at[idx, 'home_elo_after_game'] = home_elo_after
    wc2018_elo_df.at[idx, 'away_elo_after_game'] = away_elo_after
    
    # update current elos
    elo_teams_df.set_value(elo_teams_df.loc[elo_teams_df.team == home_team].index.values[0], 'elo', home_elo_after)
    elo_teams_df.set_value(elo_teams_df.loc[elo_teams_df.team == away_team].index.values[0], 'elo', away_elo_after)

wc2018_elo_df.head()

# join region and income group data
wc2018_elo_df = wc2018_elo_df.merge(country_df, 
                        left_on=['home_team'], 
                        right_on=['ShortName'])
wc2018_elo_df = wc2018_elo_df.merge(country_df, 
                        left_on=['away_team'], 
                        right_on=['ShortName'], 
                        suffixes=('_home', '_away'))
wc2018_elo_df.sort_values(by = 'index', inplace=True)

# Add additional features to match the test set
wc2018_elo_df['elo_difference'] = wc2018_elo_df['home_elo_before_game'] - wc2018_elo_df['away_elo_before_game']
wc2018_elo_df['average_elo'] = (wc2018_elo_df['home_elo_before_game'] + wc2018_elo_df['away_elo_before_game'])/2


wc2018_test_df = wc2018_elo_df[['neutral','home_elo_before_game','away_elo_before_game','is_stake','elo_difference',
                                 'average_elo','Region_Europe & Central Asia_home','Region_Latin America & Caribbean_home',
                                 'Region_Middle East & North Africa_home','Region_North America_home','Region_South Asia_home',
                                 'Region_Sub-Saharan Africa_home','IncomeGroup_High income: nonOECD_home',
                                 'IncomeGroup_Low income_home','IncomeGroup_Lower middle income_home',
                                 'IncomeGroup_Upper middle income_home','Region_Europe & Central Asia_away',
                                 'Region_Latin America & Caribbean_away','Region_Middle East & North Africa_away',
                                 'Region_North America_away','Region_South Asia_away','Region_Sub-Saharan Africa_away',
                                 'IncomeGroup_High income: nonOECD_away','IncomeGroup_Low income_away',
                                 'IncomeGroup_Lower middle income_away','IncomeGroup_Upper middle income_away','result']]

# Select features in the same order as used for training
wc2018_features = wc2018_test_df.drop(['result'], axis = 1)
wc2018_labels = np.asarray(wc2018_elo_df['result'], dtype="|S6")

import statsmodels.api as sm 

X = wc2018_test_df.drop(['result'], axis = 1)
X = sm.add_constant(X, has_constant='add')
X = np.asarray(X, dtype='float')

prediction_home_score = np.round(model_home.predict(X),0)
prediction_away_score = np.round(model_away.predict(X),0)

wc2018_features['pred_home_score'] = prediction_home_score
wc2018_features['pred_away_score'] = prediction_away_score

print('Classification Accuracy on 2018 Group Stage: {}%'.format(round(lda.score(wc2018_features, \
                                                                                          wc2018_labels)*100, 2)))
                                                                                          
```
Classification Accuracy on 2018 Group Stage: 52.08%

The LDA model performed about as well on the 2018 Group Stage matches as it did in the out-of-sample test set during Part 3.


### Comments
1. Compute scores for the Group Stage to determine the group winners going to Knockout rounds

```python
wc2018_elo_df['model_pred'] = np.asarray(baselineModel.predict(wc2018_features), dtype='float')
wc2018_elo_df['pred_home_score'] = prediction_home_score
wc2018_elo_df['pred_away_score'] = prediction_away_score
np.asarray(baselineModel.predict(wc2018_features), dtype='float')

home_team_points = []
away_team_points = []
    
for index, row in wc2018_elo_df.iterrows():
    win = 3
    loss = 0
    draw = 1
    if row['model_pred'] == 0.5:
        home_team_points.append(draw)
        away_team_points.append(draw)
    elif row['model_pred'] == 1.0:
        home_team_points.append(win)
        away_team_points.append(loss)
    elif row['model_pred'] == 0.0:
        home_team_points.append(loss)
        away_team_points.append(win)
        
wc2018_elo_df['home_points'] = pd.Series(home_team_points)
wc2018_elo_df['away_points'] = pd.Series(away_team_points)

cols = {'home_team': 'Team', 'pred_home_score': 'Goals', 'home_points': 'Points',
        'away_team': 'Team', 'pred_away_score': 'Goals', 'away_points': 'Points'}

home_team_group_results_df = wc2018_elo_df[['home_team', 'pred_home_score', 'home_points']].rename(columns=cols)
away_team_group_results_df = wc2018_elo_df[['away_team', 'pred_away_score', 'away_points']].rename(columns=cols)

scoreboard_df = pd.concat([home_team_group_results_df, away_team_group_results_df]).reset_index(drop=True)
scoreboard_df = scoreboard_df.merge(pd.read_csv('worldcup_country_groups.csv'), on='Team', how='left')
scoreboard_df = scoreboard_df.groupby(['Group', 'Team']).sum().reset_index()
scoreboard_df = scoreboard_df.sort_values(['Group', 'Points', 'Goals'], ascending=[True, False, False]).reset_index(drop=True)
scoreboard_df

```

![World Cup Results](/Images/wc2.PNG)


### Comments
1. Compute Round of 16
2. Predict Round of 16

```python
round_of_16_games = [(0, 5), (8, 13),
                     (16, 21), (24, 29),
                     (4, 1), (12, 9),
                     (20, 17), (28, 25)]

round_of_8_games = [(0, 1), (2, 3),
                     (4, 5), (6, 7)]

semifinal_games = [(0, 1), (2, 3)]

final_game = [(0, 1)]
players_df = pd.read_csv('CompleteFIFA2018PlayerDataset.csv', low_memory=False)

def get_team_penalties(team):
    mask = players_df['Nationality'] == team
    return pd.to_numeric(players_df[mask]['Penalties'].head(5), errors='coerce').mean()

def simulate_penalties(home_team, away_team):
    home_pen = get_team_penalties(home_team)
    away_pen = get_team_penalties(away_team)
    
    if home_pen == away_pen:
        winner = home_team
    elif home_pen > away_pen:
        winner = home_team
    elif home_pen < away_pen:
        winner = away_team
    
    return winner
winners = []

for match in round_of_16_games:
    
    home_team = scoreboard_df.iloc[match[0]]['Team']
    away_team = scoreboard_df.iloc[match[1]]['Team']
    
    # Get pre-match ratings
    teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
    teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

    expected_win = expected_result(teamA_elo, teamB_elo)
    
    print('{} vs. {}'.format(home_team, away_team))
    
    if expected_win > 0.7:
        print('{} wins in regulation time\n'.format(home_team))
        winners.append(home_team)
    elif expected_win < 0.3:
        print('{} wins in regulation time\n'.format(away_team))
        winners.append(away_team)
    else:
        winning_team = simulate_penalties(home_team, away_team)
        print('{} wins in OT/PK\n'.format(winning_team))
        winners.append(winning_team)
```
Egypt vs. Portugal<br>
Egypt wins in OT/PK<br>

France vs. Argentina<br>
Argentina wins in OT/PK<br>

Costa Rica vs. Sweden<br>
Sweden wins in regulation time<br>

Belgium vs. Japan<br>
Belgium wins in regulation time<br>

Spain vs. Russia<br>
Spain wins in OT/PK<br>

Iceland vs. Peru<br>
Iceland wins in OT/PK<br>

Germany vs. Switzerland<br>
Germany wins in OT/PK<br>

Colombia vs. Panama<br>
Colombia wins in regulation time<br>

### Comments
1. Compute Round of 8
2. Predict Round of 8

```python
round_of_8_games = np.array(winners).reshape(4,2)

winners = []

for match in round_of_8_games:
    
    home_team = match[0]
    away_team = match[1]
    
    # Get pre-match ratings
    teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
    teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

    expected_win = expected_result(teamA_elo, teamB_elo)
    
    print('{} vs. {}'.format(home_team, away_team))
    
    if expected_win > 0.7:
        print('{} wins in regulation time\n'.format(home_team))
        winners.append(home_team)
    elif expected_win < 0.3:
        print('{} wins in regulation time\n'.format(away_team))
        winners.append(away_team)
    else:
        winning_team = simulate_penalties(home_team, away_team)
        print('{} wins in OT/PK\n'.format(winning_team))
        winners.append(winning_team)

```

Egypt vs. Argentina<br>
Argentina wins in OT/PK<br>

Sweden vs. Belgium<br>
Sweden wins in OT/PK<br>

Spain vs. Iceland<br>
Spain wins in regulation time<br>

Germany vs. Colombia<br>
Colombia wins in OT/PK<br>

### Comments
1. Compute SemiFinal
2. Predict SemiFinal Outcome

```python
semifinal_games = np.array(winners).reshape(2,2)

winners = []

for match in semifinal_games:
    
    home_team = match[0]
    away_team = match[1]
    
    # Get pre-match ratings
    teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
    teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

    expected_win = expected_result(teamA_elo, teamB_elo)
    
    print('{} vs. {}'.format(home_team, away_team))
    
    if expected_win > 0.7:
        print('{} wins in regulation time\n'.format(home_team))
        winners.append(home_team)
    elif expected_win < 0.3:
        print('{} wins in regulation time\n'.format(away_team))
        winners.append(away_team)
    else:
        winning_team = simulate_penalties(home_team, away_team)
        print('{} wins in OT/PK\n'.format(winning_team))
        winners.append(winning_team)

```
Argentina vs. Sweden
Argentina wins in OT/PK

Spain vs. Colombia
Colombia wins in OT/PK



### Comments
1. Compute Finals
2. Predict Finals Outcome

```python
home_team = winners[0]
away_team = winners[1]
   
# Get pre-match ratings
teamA_elo = elo_teams_df.loc[elo_teams_df['team'] == home_team, 'elo'].values[0]
teamB_elo = elo_teams_df.loc[elo_teams_df['team'] == away_team, 'elo'].values[0]

expected_win = expected_result(teamA_elo, teamB_elo)
    
print('2018 World Cup Final\n'.format(home_team, away_team))
print('{} vs. {}'.format(home_team, away_team))
    
if expected_win > 0.7:
    print('{} wins in regulation time\n'.format(home_team))
    winners.append(home_team)
elif expected_win < 0.3:
    print('{} wins in regulation time\n'.format(away_team))
    winners.append(away_team)
else:
    winning_team = simulate_penalties(home_team, away_team)
    print('{} wins in OT/PK\n'.format(winning_team))
    winners.append(winning_team)
```

2018 World Cup Final<br>

Argentina vs. Colombia<br><br>
**Argentina wins in OT/PK**<br>


# Literature Review

1.	Zeileis A, Leitner C, Hornik K (2018). "Probabilistic Forecasts for the 2018 FIFA World Cup Based on the Bookmaker Consensus Model", Working Paper 2018-09, Working Papers in Economics and Statistics, Research Platform Empirical and Experimental Economics, UniversitÃd’t Innsbruck. https://www2.uibk.ac.at/downloads/c4041030/wpaper/2018-09.pdf
2.	Goldman-Sachs Global Investment Research (2014). “The World Cup and Economics 2014.” Accessed 2018-07-11, http://www.goldmansachs.com/our-thinking/outlook/ world-cup-and-economics-2014-folder/world-cup-economics-report.pdf
3.	Elo, A. E. (1978). The rating of chessplayers, past and present. Arco Pub., New York.
4.	Lorenz A Gilch and Sebastian MÃijller. On Elo based prediction models for the FIFA Worldcup 2018. https://arxiv.org/abs/1806.01930
5.	Strenk, Mike, Modeling the World Cup 2018, (2018), GitHub Repository, https://github.com/MikeStrenk/Modeling-the-World-Cup-2018

