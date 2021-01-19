## http://spark.rstudio.com/
## https://www.rstudio.com/resources/cheatsheets/

# install.packages("sparklyr", dependencies=TRUE)
# .libPaths()
# myPaths <- .libPaths() # get the paths
# myPaths <- c(myPaths[2], myPaths[1])  # switch them
# .libPaths(myPaths)
# .libPaths()


#install.packages(c("nycflights13", "Lahman"))
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home")
library(sparklyr)
library(dplyr)
library(ggplot2)
library(DBI)

#spark_install(version = "2.0.0")

sc <- spark_connect(master = "local", version="2.0.0")

#db_drop_table(sc, "iris")
# bring R datafrma to Spark
iris_tbl <- copy_to(sc, iris)

src_tbls(sc)




iris_preview <- dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview


# copy mtcars into spark
mtcars_tbl <- copy_to(sc, mtcars)
mtcars_tbl

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  mutate(cyl8 = cyl == 8) %>%
  sdf_random_split(training = 0.5, test = 0.5, seed = 1099)

# fit a linear model to the training dataset
model <- partitions$training %>%
  ml_linear_regression(response = "mpg", features = c("wt", "cyl"))

model
summary(model)

# Score the data
pred <- ml_predict(model, partitions$test) %>%
  collect

# Plot the predicted versus actual mpg
ggplot(pred, aes(x = mpg, y = prediction)) +
  geom_abline(lty = "dashed", col = "red") +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Actual Fuel Consumption",
    y = "Predicted Fuel Consumption",
    title = "Predicted vs. Actual Fuel Consumption"
  )

spark_disconnect(sc)
