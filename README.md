## World Cup

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
