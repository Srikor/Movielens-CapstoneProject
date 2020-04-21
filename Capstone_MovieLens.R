################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes
# Install the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download the Movilens 10M data set zip file from grouplens website into an temporary object. This zip file
# contains two .dat files, one containing the list of movies and the other containing the ratings 
# for the corresponding movies in the previous file.
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# The "ratings" object below will contain the ratings from the ratings.dat file. The data in both
# the files are delimited with a double colon "::". Hence, extract the data to seperate columns using
# this delimiter. Set column names for clarity and programming purposes.
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Extract the movies data from movies.dat file similar to the ratings. Set appropriate column names.
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Ensure the data in the columns are of appropriate data type.
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

# Combine the data from both the ratings and movies datasets into a single dataset.
movielens <- left_join(ratings, movies, by = "movieId")

# Set seed to ensure the code outputs same results regardless of the machine it is being executed.
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1, sample.kind="Rounding")

# Create the training and testing sets. The training set (edx) will contain 90% of the records in the 
# movielens data frame. The test set (validation) will contain the balance 10% of records.
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set. Unless both User Id and Movie Id
# are there in both the data sets, predicting the rating for a particular user, for a particular movie
# may not be accurate.
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# The below is a convinience function to calculate the RMSE once the model has been trained
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#########################
# ### Model training ####
#########################
# The average rating of a movie and the number of Users rating the movie predominantely decides its
# fate in the box office. Hence, the average rating of the movie and the user bias towards a movie 
# will be used in predicting the user rating.

# The edx dataset has approxmiately 9 million ratings. This can be split into train and test datasets
# for training and testing the model recursively prior to predicting the ratings in the validation
# dataset.
split_idx <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-split_idx,]
test_temp <- edx[split_idx,]

# Similar to the steps done with the edx and validation dataset make sure userId and movieId in 
# test set are also in train set.
test <- test_temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

test_removed <- anti_join(test_temp, test)
train <- rbind(train, test_removed)

# Calculate the average movie ratings for all the movies in train datast.
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(mu_m = mean(rating))

# Calculate the User bias; Each User may rate a movie differently based on their perception. 
# Hence, find out how much does a User rating differ from the average rating for the movie.
user_bias <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_m))

# Predict the ratings in the test dataset using the movie averages and user bias.
prediction_test <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  mutate(predicted_rating = mu_m + b_u)

# Calculate the RMSE for the above prediction.
RMSE(prediction_test$predicted_rating, test$rating) 

# The RMSE based on the above model is 0.8646843. The above model did not include any L2 regularization 
# parameter lambda. Identify the optimal lambda that minimizes the RMSE. Lambda values from 1 to 10
# with increments of 1 can be tried on the user bias to identify the optimal lambda value.
lambda_seq <- seq(1,10,1)

# Use sapply to apply the potential lambda values on the user bias
rmse_val <- sapply(lambda_seq, function(x){
  user_bias <- train %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_m)/(n() + x))
  
  prediction_test <- test %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_bias, by='userId') %>%
    mutate(predicted_rating = mu_m + b_u)
  
  RMSE(prediction_test$predicted_rating, test$rating)
})

lambda_optimal <- data.frame(lambda = lambda_seq, rmse= rmse_val) %>% 
  filter(rmse == min(rmse))

# The above code resulted in optimal lambda value of 5 with a RMSE of 0.864253. This lambda will
# be used in the prediction of ratings in the validation set.
lambda <- lambda_optimal$lambda

#####################
# ### Prediction ####
#####################
# Update the movie averages and user bias objects to use the edx dataset for training
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(mu_m = mean(rating))

user_bias <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_m)/(n() + lambda))

# Use the model finalized in the training section with the lambda value to predict the movie ratings
prediction <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  mutate(predicted_rating = mu_m + b_u)

# Find out the Root Mean Square Error between the predicted rating and the rating in the validation
# data set
RMSE(prediction$predicted_rating, validation$rating) 

# The RMSE for above prediction is 0.8649708

########################
### Results Analysis ###
########################
# The below code shows there is not a single predicted rating that is equal to the average rating. This is 
# expected since the actual ratings are in multiples of 0.5 while the predicted ratings are not rounded 
# of to multiples of 0.5.
prediction %>% 
  filter(predicted_rating == rating) %>% head()