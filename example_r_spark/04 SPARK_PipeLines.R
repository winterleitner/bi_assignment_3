## https://spark.rstudio.com/guides/pipelines/

library(sparklyr)
library(dplyr)
library(tidyr)
library(DBI)
library(nycflights13)

# Connect to local spark cluster and load data
# Convert titanic_train data into parquet format and output to disk
# https://github.com/rstudio/sparkDemos/blob/master/dev/cloudera/spark_ml_classification_titanic.Rmd
sc <- spark_connect(master = "local", version = "2.0.0")

# first R pipeline
# empty pipeline
r_pipeline <-  . %>%lm(am ~ cyl + mpg, data = .)
r_pipeline


# use the pipeline on a dataframe
r_model <- r_pipeline(mtcars)
r_model



# to prevent from errors futher down the coding , reformat sched_dep_time
flights$sched_dep_time <- as.double(flights$sched_dep_time)

spark_flights <- sdf_copy_to(sc, flights,overwrite = TRUE)


flightsSQL <- dbGetQuery(sc, 'SELECT * 
                              FROM flights 
                              WHERE DEP_DELAY <= 0')
flightsSQL %>% collect()


flightsSQL <- dbGetQuery(sc, 'SELECT carrier,avg(dep_delay) AS AvgDelay
                              FROM flights 
                              GROUP BY carrier
                              ORDER BY AvgDelay DESC')
flightsSQL %>% collect()


flightsSQL <- dbGetQuery(sc, 'SELECT carrier,max(dep_delay)/60 AS MaxDelay
                              FROM flights 
                              GROUP BY carrier
                              ORDER BY MaxDelay DESC')
flightsSQL %>% collect()

#  only instructions nothing is done yet on the sprakcluster
df <- spark_flights %>%
  filter(!is.na(dep_delay)) %>%
  mutate(
    month = paste0("m", month),
    day = paste0("d", day)
  ) %>%
  select(dep_delay, sched_dep_time, month, day, distance) 


ft_dplyr_transformer(sc, df)
ft_dplyr_transformer(sc, df) %>%
  ml_param("statement")

flights_pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(
    tbl = df
  ) %>%
  ft_binarizer(
    input_col = "dep_delay",
    output_col = "delayed",
    threshold = 15
  ) %>%
  ft_bucketizer(
    input_col = "sched_dep_time",
    output_col = "hours",
    splits = c(400, 800, 1200, 1600, 2000, 2400)
  )  %>%
  ft_r_formula(delayed ~ month + day + hours + distance) %>% 
  ml_logistic_regression()

flights_pipeline

# split the date in training and testing
partitioned_flights <- sdf_random_split(
  spark_flights,
  training = 0.01,
  testing = 0.01,
  rest = 0.98
)

# build a model on SPARK
fitted_pipeline <- ml_fit(
  flights_pipeline,
  partitioned_flights$training
)
fitted_pipeline

# do some predictions
predictions <- ml_transform(
  fitted_pipeline,
  partitioned_flights$testing
)

# show confusion matrix
predictions %>%
  group_by(delayed, prediction) %>%
  tally()

# save SPARK df to disk
ml_save(
  flights_pipeline,
  "flights_pipeline",
  overwrite = TRUE
)

# save SPARK model to disk
ml_save(
  fitted_pipeline,
  "flights_model",
  overwrite = TRUE
)

# load model again
reloaded_model <- ml_load(sc, "flights_model")

# do some new predictions
new_df <- spark_flights %>%
  filter(
    month == 7 #,
    #day == 5
  )

newPredictions <- ml_transform(reloaded_model, new_df)

newPredictions %>%
  group_by(delayed, prediction) %>%
  tally()
