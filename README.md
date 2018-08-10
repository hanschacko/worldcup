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

History of FIFA rankings<br>

Original Rankings in 1993 | 1999 Ranking Updates | 2006 Ranking Updates
--------------------------|----------------------|---------------------
3 points for a win <br> 1 point for a draw | Point scale factored 10x <br> Considered # of goals scored <br> and conceded,home/away, importance <br>of match, and regional strength |
Reduced evaluation period <br>from 8 to 4 years


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
![Alt Text](url)
![International Matches Per Year](/Images/IntlMatchesYear.png)
## Native Model based on FIFA Ranking Only
![Alt Text](url)
## Visualize Subset of Data for Portugal Matches Only
![Portugal Matches](/Images/IntlMatchesYear.png))
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
CS-109A Data Science Final project
Bruno Janota and Hans Chacko #12


# Project Statement and Background

The FIFA World Cup is one of the most prominent sporting events in the world yet relative to
other major sports, football/futbol/soccer analytics generally lags the state of the art for sports
analytics. There are many reasons why national team soccer analytics trails other major sports
like baseball basketball and American football. For example, many national team matches are
against regional opponents as part of regional tournaments or world cup qualifiers and a
disproportional amount of the available data is from segmented sources therefore, compiling
something as simple as a relative ranking among all nations can be quite challenging.
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
goals scored and home or away advantage.
Additionally, it was recently announced that following the 2018 World Cup, the FIFA world
rankings would adopt an Elo based ranking system which considers many factors including the
teams previous rating, the status (in order of importance: World Cup/Olympic games, continental
championship and intercontinental tournaments, world cup qualifiers, all other tournaments, and
lastly friendlies), goal difference, result, and expected result of the match. Since this ranking has 
not yet been introduced the basis of this analysis will compare the results to the pre-2018
rankings.

History of FIFA rankings

Original Rankings in 1993 | 1999 Ranking Updates | 2006 Ranking Updates
--------------------------|----------------------|---------------------
3 points for a win <br> 1 point for a draw | Point scale factored 10x <br> Considered # of goals scored <br> and conceded,home/away, importance <br>of match, and regional strength |
Reduced evaluation period <br>from 8 to 4 years



# Project Goal
The goal of this project will be to leverage various sources of team and player data in addition to
historical match results to construct a 2018 FIFA World Cup prediction model and evaluate those
models against the baseline of predictions from simply incorporating FIFA ranking as a measure
of team strength.

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
## Native Model based on FIFA Ranking Only
## Visualize Subset of Data for Portugal Matches Only
## Evaluate Baseline Decision Tree Model 

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
## Performance Compariron to a non-ML Approach
### Cleaning the Data
### Aggregating Team Stats
### Building a dataset for training a model
### Finalizing Training Dataset Format
### Introducing Poisson Distribution 

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
