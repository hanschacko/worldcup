## CS-109A Data Science Final project
Bruno Janota and Hans Chacko #12


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
The goal of this project will be to leverage various sources of team/player data, historical international match results, historical FIFA rankings, and betting odds to construct a 2018 FIFA World Cup prediction model. The model will predict a win, loss, or draw match outcome for historical international matches from 1993 to 2018 divided into a training and testing set and evaluate those predictions against the baseline of predictions from simply incorporating FIFA ranking as a naïve measure of team strength (i.e. the team with a higher ranking should win). In addition to predicting historical international matches, the classification model will also be used to predict the outcome of the 2018 FIFA World Cup.

The project will be split into three parts:
Part 1 : Baseline Model
Combine the international match outcome and FIFA rankings data sets for matches between 1993 and 2018. Perform some basic feature engineering to add additional features. Lastly, perform a 70/30% train/test set split and evaluate the performance of a Random Forest model on the overall classification accuracy on the test set. This will be the baseline model.
Part 2 : Incorporating ELO and Other Feature Engineering
Replace the FIFA rankings in Part 1 with an Elo based scoring model and assess improvement, if any, on the overall classification accuracy on the same test set used in part 1.
The optimal combination of features from parts 1 and 2 (may include FIFA rank, Elo score, or both) will be used to train a variety of classification models (Random Forest, xgboost, LDA, QDA, KNN, etc.). The probabilistic results for each match (win, tie, loss) will be blended with the results of a Poisson Distribution model that uses the complete player ranking data scraped from sofifa.com for the FIFA 2018 video game to predict the group stages of the 2018 FIFA World Cup. 
Part 3 : Precting the 2018 World Cup
The knockout stages of the world cup will be simulated 1000 times to determine the probability of each team winning the tournament. Matches that result in a tie during the knockout stages will take into account the average penalty rating of the top 5 penalty shooters for each time from the sofifa.com data set to break the tie. The final result will be the probability of each of the 32 teams that qualified for the 2018 World Cup to win the tournament.

# Part 1 : Baseline Model
## Load, Clean and Merge Data

### Comments
Load historical FIFA rankings (1993-present) dataset into pandas dataframe
```python
# load historical FIFA rankings (1993-present) dataset into pandas dataframe
fifaRank_df = pd.read_csv('FifaRanking1993to2018_Tadhg Fitzgerald.csv')
fifaRank_df = fifaRank_df.loc[:,['rank', 'country_full', 'country_abrv', 'rank_date', 'total_points', 'previous_points',
                                 'cur_year_avg_weighted', 'two_year_ago_weighted', 'three_year_ago_weighted']]
fifaRank_df = fifaRank_df.replace({"IR Iran": "Iran"})
fifaRank_df['rank_date'] = pd.to_datetime(fifaRank_df['rank_date'])
```

### Comments
1. Load international match results (1872-2018) dataset into pandas dataframe
2. Load country region and income group data into pandas dataframe
3. Convert categorical variables using one-hot encoding
```python
results_df = pd.read_csv('InternationalResultsFrom1993to2018.csv')
results_df =  results_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
results_df['date'] = pd.to_datetime(results_df['date'])

results_df = pd.read_csv('InternationalResultsFrom1993to2018.csv')
results_df =  results_df.replace({'Germany DR': 'Germany', 'China': 'China PR'})
results_df['date'] = pd.to_datetime(results_df['date'])

country_df = pd.read_csv('WorldCountryData.csv')

country_df = pd.get_dummies(country_df, columns=['Region','IncomeGroup'], drop_first=True)
```

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
![Alt Text](url)

## Visualize Subset of Data for Portugal Matches Only
![Portugal Matches](/Images/Portugal.png))

## Evaluate Baseline Decision Tree Model 
![Baseline Tree](/Images/BaselineTree.png)

# Part 2 : Developing Model with Elo Rating
## Introduction to Elo Rating
The Elo rating system is a method for calculating the relative skill levels of teams (or players) in zero-sum games. From Wikipedia, a team's Elo rating as applied to our data set is represented by a number which increases or decreases depending on the outcome of matches between international teams. After every game, the winning team takes points from the losing one. The difference between the ratings of the winner and loser determines the total number of points gained or lost after a game. In a series of games between a high-rated team and a low-rated team, the high-rated team is expected to score more wins. If the high-rated team wins, then only a few rating points will be taken from the low-rated team. However, if the lower rated team scores an upset win, many rating points will be transferred. The lower rated team will also gain a few points from the higher rated team in the event of a draw. This means that this rating system is self-correcting. A team whose rating is too low should, in the long run, do better than the rating system predicts, and thus gain rating points until the rating reflects their true playing strength. The elo function that we used resembles the Elo World Ranking which is defined in the images below.

## Compute Elo Rating
The loop where it happens
- We go through each row in the DataFrame.
- We look up the current Elo rating of both teams.
- We calculate the expected wins for the team that actually won.
- Write Elo before and after the game in the Data Frame.
- Update the Elo rating for both teams in the "current_elos" list.

# Part 3 : Predicting the 2018 World Cup 

## Ensemble Learners
The optimal combination of features from parts 1 and 2 (may include FIFA rank, Elo score, or both) will be used to train a variety of classification models (Random Forest, xgboost, LDA, QDA, KNN, etc.). The probabilistic results for each match (win, tie, loss) will be blended with the results of a Poisson Distribution model that uses the complete player ranking data scraped from sofifa.com for the FIFA 2018 video game to predict the group stages of the 2018 FIFA World Cup. The knockout stages of the world cup will be simulated 1000 times to determine the probability of each team winning the tournament. Matches that result in a tie during the knockout stages will take into account the average penalty rating of the top 5 penalty shooters for each time from the sofifa.com data set to break the tie. The final result will be the probability of each of the 32 teams that qualified for the 2018 World Cup to win the tournament.

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
