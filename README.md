## CS-109A Data Science Final project
Harvard University <br>
Summer 2018 <br>
Team: Bruno Janota and Hans Chacko #12 <br>


[Project Statement and Background](#project-statement-and-background)<br>
[Project Overview](#project-overview)<br>
[Part1: Baseline Model](#part1:-baseline-model)<br>
[Part 2: Developing Model with Elo Ranking](#part-2:-developing-model-with-elo-ranking)<br>
[Part 3: Ensemble Models](#part-3:-ensemble-models)<br>
[Part 4: Predicting the 2018 World Cup](#part-4:predicting-the-2018-world-cup)<br>


Part 2: Developing Model with Elo Ranking

# Project Statement and Background

The goal of this project will be to leverage various sources of team and player data in addition to historical match results to construct a 2018 FIFA World Cup prediction model and evaluate those models against the baseline of predictions from simply incorporating FIFA ranking as a measure of team strength. <br>

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
1. Accuracy of FIFA Ranking at Predicting Winner: 21.88%
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

![Baseline Tree](/Images/TreeDepth.png)

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

## Comment
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

## Comment
1. Number of Upsets based on FIFA Rank: 53.7%
2. Number of Upsets based on Elo Score: 20.3%
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

Decision Tree Classifier:
Classification Accuracy on training set: 74.12%
Classification Accuracy on testing set: 72.69%

# Part 3: Ensemble Models
## Ensemble Learners
The optimal combination of features from parts 1 and 2 (may include FIFA rank, Elo score, or both) will be used to train a variety of classification models (Random Forest, xgboost, LDA, QDA, KNN, etc.). The probabilistic results for each match (win, tie, loss) will be blended with the results of a Poisson Distribution model that uses the complete player ranking data scraped from sofifa.com for the FIFA 2018 video game to predict the group stages of the 2018 FIFA World Cup. The knockout stages of the world cup will be simulated 1000 times to determine the probability of each team winning the tournament. Matches that result in a tie during the knockout stages will take into account the average penalty rating of the top 5 penalty shooters for each time from the sofifa.com data set to break the tie. The final result will be the probability of each of the 32 teams that qualified for the 2018 World Cup to win the tournament.

# Part 4: Predicting the 2018 World Cup 


## Computing the Probabilities of Wins for the 32 Qualifying Teams
![Alt Text](url)
## Performance Comparison to a non-ML Approach
![Alt Text](url)
### Cleaning the Data
![Alt Text](url)
### Aggregating Team Stats
![Alt Text](url)
### Building a dataset for training a model
![Alt Text](url)
### Finalizing Training Dataset Format
![Alt Text](url)
### Introducing Poisson Distribution 
![Alt Text](url)

# Literature Review

1. Zeileis A, Leitner C, Hornik K (2018). "Probabilistic Forecasts for the 2018 FIFA World
Cup Based on the Bookmaker Consensus Model", Working Paper 2018-09, Working
Papers in Economics and Statistics, Research Platform Empirical and Experimental
Economics, UniversitÃd’t Innsbruck.
https://www2.uibk.ac.at/downloads/c4041030/wpaper/2018-09.pdf
2. Goldman-Sachs Global Investment Research (2014). “The World Cup and Economics
2014.” Accessed 2018-07-11, http://www.goldmansachs.com/our-thinking/outlook/
world-cup-and-economics-2014-folder/world-cup-economics-report.pdf
3. Elo, A. E. (1978). The rating of chessplayers, past and present. Arco Pub., New York.
4. Lorenz A Gilch and Sebastian MÃijller. On Elo based prediction models for the FIFA
Worldcup 2018. https://arxiv.org/abs/1806.01930
