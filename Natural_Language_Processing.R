library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(rstudioapi)

path<-dirname(getSourceEditorContext()$path)
setwd(path)

email <- fread('emails.csv')

email %>% class()

email$text <- email$text %>% str_replace_all("Subject: ","")

email %>% view()


email<- email %>% mutate(id=row_number())

email<- email %>% select(id, text, spam) %>% view()


email %>% inspectdf::inspect_na()


#Splitting data
set.seed(123)
split <- email$spam %>% sample.split(SplitRatio = 0.8)
train <- email %>% subset(split == T)
test <- email %>% subset(split == F)


#Tokenizing 
train_tokens <- gsub('[[:punct:]0-9]', ' ',train$text %>% tolower()) %>% word_tokenizer()

it_train <- train_tokens %>% 
  itoken(progressbar = F)



#exclude words 
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are",'subject','a','s','re',
                'hi')

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


#remove the least and the most words in data 

pruned_vocab <- vocab %>%          
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()



# N Grams ----  # word combinations to strength model
vocab <- it_train %>% create_vocabulary(ngram = c(2L, 3L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5)

bigram_vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(bigram_vectorizer)
dtm_train %>% dim()





#----------------Modelling-----------------


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training



glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")   # on train set




it_test <- test$text %>% tolower() %>% word_tokenizer()  #on test set 

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)


dtm_test <- it_test %>% create_dtm(bigram_vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)





