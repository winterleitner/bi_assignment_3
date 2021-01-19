#https://beta.rstudioconnect.com/content/1518/notebook-classification.html

library(sparklyr)
library(dplyr)
library(tidyr)
#library(titanic)
library(ggplot2)
#library(purrr)

# Connect to local spark cluster and load data
# Convert titanic_train data into parquet format and output to disk
# https://github.com/rstudio/sparkDemos/blob/master/dev/cloudera/spark_ml_classification_titanic.Rmd
sc <- spark_connect(master = "local", version = "2.0.0")

filelocation <- paste(getwd(),"/W14/titanic.csv",sep="")


titanic <- as.data.frame(read.csv(filelocation,sep="\t"))

titanicdf <- read.csv("W14/titanic.csv", sep="\t")
parquet_path <- "./titanic-parquet"
parquet_table <- "titanic"
if(!dir.exists(parquet_path)){

  #copy_to(sc, titanic_train, parquet_table, overwrite = TRUE)
  copy_to(sc, titanic, parquet_table, overwrite = TRUE)
  tbl(sc, parquet_table) %>% spark_write_parquet(path = parquet_path)
}

spark_read_parquet(sc, name = "titanic", path = "titanic-parquet")

db_drop_table(sc, "titanic")
titanic_tbl <- copy_to(sc, titanic,"titanic")

# Transform features with Spark ML API
titanic2_tbl <- titanic_tbl %>% 
  mutate(Family_Size = SibSp + Parch + 1L) %>% 
  mutate(Pclass = as.character(Pclass)) %>%
  filter(!is.na(Embarked)) %>%
  mutate(Age = if_else(is.na(Age), mean(Age), Age)) %>%
  sdf_register("titanic2")

# Transform family size with Spark ML API
titanic_final_tbl <- titanic2_tbl %>%
  mutate(Family_Size = as.numeric(Family_size)) %>%
  ft_bucketizer(input_col = "Family_Size",
                output_col = "Family_Sizes",
                splits = c(1,2,5,12)) %>%
  mutate(Family_Sizes = as.character(as.integer(Family_Sizes)))  %>%
  sdf_register("titanic_final")

# Partition the data
partition <- titanic_final_tbl %>% 
  mutate(Survived = as.numeric(Survived), SibSp = as.numeric(SibSp), Parch = as.numeric(Parch)) %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Family_Sizes) %>%
  sdf_random_split(train = 0.75, test = 0.25, seed = 8585)

# Create table references
train_tbl <- partition$train
test_tbl <- partition$test

# Model survival as a function of several predictors
ml_formula <- formula(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Family_Sizes)

# Train a logistic regression model
ml_log <- ml_logistic_regression(train_tbl, ml_formula)

## Random Forest
ml_rf <- ml_random_forest(train_tbl, ml_formula)

# Bundle the modelss into a single list object
ml_models <- list(
  "Logistic" = ml_log,
  "Random Forest" = ml_rf
)

# Create a function for scoring
score_test_data <- function(model, data=test_tbl){
  pred <- ml_predict(model, data)
  select(pred, Survived, prediction)
}

# Score all the models
ml_score <- lapply(ml_models, score_test_data)

# Function for calculating accuracy
calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_multiclass_classification_evaluator("prediction", "Survived", "accuracy")
}

# Calculate AUC and accuracy
perf_metrics <- data.frame(
  model = names(ml_score),
  AUC = 100 * sapply(ml_score, ml_multiclass_classification_evaluator, "Survived", "prediction"),
  Accuracy = 100 * sapply(ml_score, calc_accuracy),
  row.names = NULL, stringsAsFactors = FALSE)

# Plot results
gather(perf_metrics, metric, value, AUC, Accuracy) %>%
  ggplot(aes(reorder(model, value), value, fill = metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() +
  xlab("") +
  ylab("Percent") +
  ggtitle("Performance Metrics")
