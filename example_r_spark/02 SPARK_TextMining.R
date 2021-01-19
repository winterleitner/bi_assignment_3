.libPaths()
# http://rstudio.github.io/sparklyr/articles/guides-textmining.html

library(sparklyr)
library(dplyr)
library(dbplyr)

#spark_install(version = "2.0.0")

sc <- spark_connect(master = "local", version="2.0.0")
#setwd("C:/Users/poden/surfdrive/10 HvA/01 Data Engineer Data Scientist/01 Weekly material/10 Spark/demo R/")
getwd()

DonQuichote_path <- paste0(getwd(),"/W14/DonQuichote.txt")
DonQuichote <-  spark_read_text(sc, "DonQuichote", DonQuichote_path) 



all_words <- DonQuichote%>%
  mutate(line = regexp_replace(line, "[_\"\'():;,.!?\\-]", " ")) 

all_words <- all_words %>%
  ft_tokenizer(input_col = "line",
               output_col = "word_list")

head(all_words, 4)

all_words <- all_words %>%
  ft_stop_words_remover(input_col = "word_list",
                        output_col = "wo_stop_words")


head(all_words, 4)

all_words <- all_words %>%
  mutate(word = explode(word_list)) %>%
  select(word) %>%
  filter(nchar(word) > 4)

head(all_words, 4)



head(all_words, 4)

all_words <- all_words %>%
  compute("all_words")


word_count <- all_words %>%
  group_by(word) %>%
  tally() %>%
  arrange(desc(n)) 

word_count

word_count %>%
  head(100) %>%
  collect() %>%
  with(wordcloud::wordcloud(
    word, 
    n,
    colors = c("#999999", "#E69F00", "#56B4E9","#56B4E9")))
