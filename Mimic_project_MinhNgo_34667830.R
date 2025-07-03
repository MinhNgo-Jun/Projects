# Name: Minh Chung Ngo
# Student ID: 34667830

## Load packages:

library(MVN)
library(tourr)
library(tidyverse)
library(tidymodels)
library(conflicted)
library(colorspace)
library(patchwork)
library(discrim)
library(MASS)
library(randomForest)
library(gridExtra)
library(GGally)
library(geozoo)
library(mulgar)
library(rpart)
library(fpp3)
library(xgboost)
library(keras3)
library(reticulate)
library(caret)
library(tm)
library(SnowballC)
library(tidytext)
library(stringr)
library(tokenizers)
library(stopwords)
library(textclean)
library(ggcorrplot)

## Load Data:

mimic_train_x <- read.csv('mimic_data_kaggle/mimic_train_X.csv')
mimic_test_x <- read.csv('mimic_data_kaggle/mimic_test_X.csv')
mimic_train_y <- read.csv('mimic_data_kaggle/mimic_train_y.csv')
diagnose <- read.csv('mimic_data_kaggle/MIMIC_metadata_diagnose.csv')
extra_diagnose <- read.csv('mimic_data_kaggle/extra_data/MIMIC_diagnoses.csv')

## Adjust data format to be appropriate for machine learning:

mimic_train <- mimic_train_x %>% 
  left_join(mimic_train_y, by = 'icustay_id') %>%
  left_join(diagnose, by = c("ICD9_diagnosis" = "ICD9_CODE")) %>%
  select(-c(X.x, X.y)) %>%
  mutate(HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG),
         Age_admit=as.numeric(difftime(as_date(ADMITTIME),as_date(DOB),units='days'))/365.25,
         ADMISSION_TYPE = factor(mimic_train_x$ADMISSION_TYPE),
         INSURANCE = factor(mimic_train_x$INSURANCE),
         ICD9_diagnosis = factor(mimic_train_x$ICD9_diagnosis),
         FIRST_CAREUNIT = factor(mimic_train_x$FIRST_CAREUNIT),
         GENDER = factor(mimic_train_x$GENDER)) %>%
  mutate(across(c(SHORT_DIAGNOSE,LONG_DIAGNOSE), ~replace_na(., 'None'))) %>%
  select(-c(Diff,DOB,RELIGION,ETHNICITY,MARITAL_STATUS,ADMITTIME)) 

mimic_test <- mimic_test_x %>%
  left_join(diagnose, by = c("ICD9_diagnosis" = "ICD9_CODE")) %>%
  mutate(Age_admit=as.numeric(difftime(as_date(ADMITTIME),as_date(DOB),units='days'))/365.25,
         ADMISSION_TYPE = factor(mimic_test_x$ADMISSION_TYPE),
         INSURANCE = factor(mimic_test_x$INSURANCE),
         ICD9_diagnosis = factor(mimic_test_x$ICD9_diagnosis),
         FIRST_CAREUNIT = factor(mimic_test_x$FIRST_CAREUNIT),
         GENDER = factor(mimic_test_x$GENDER)) %>%
  mutate(across(c(SHORT_DIAGNOSE,LONG_DIAGNOSE), ~replace_na(., 'None'))) %>%
  select(-c(Diff,DOB,RELIGION,ETHNICITY,MARITAL_STATUS,ADMITTIME))

## Heat-map 

### Select only numeric columns except IDs
mimic_corr_data <- mimic_train %>%
  select(where(is.numeric)) %>%
  select(-subject_id, -hadm_id, -icustay_id)

### Compute correlation matrix
corr_matrix <- cor(mimic_corr_data, use = "complete.obs")

### Convert to long format
corr_long <- as.data.frame(as.table(corr_matrix))

### Plot heatmap
ggplot(corr_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue", high = "red", mid = "white",
    midpoint = 0, limit = c(-1, 1), space = "Lab",
    name = "Correlation"
  ) +
  theme_minimal(base_size = 10) +
  labs(title = "Correlation Heatmap of Numeric Features") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

## Fit linear model to see which one is significantly statistical. 
var_test_data <- mimic_train %>%
  select(where(is.numeric),HOSPITAL_EXPIRE_FLAG) %>%
  select(-subject_id, -hadm_id, -icustay_id)

test <- glm(HOSPITAL_EXPIRE_FLAG ~ .,data=var_test_data, family = binomial)

summary(test) 

### Variables should be kept when fitting model (when they cannot handle many variables)

vars_num_keep <- c('SysBP_Min','DiasBP_Mean',"MeanBP_Mean", "HeartRate_Mean", "RespRate_Mean",
                   "SpO2_Min", "TempC_Mean", "Glucose_Mean", 'Age_admit', 'GENDER', 'ADMISSION_TYPE', 
                   'INSURANCE','FIRST_CAREUNIT','diag_scores','sdiag_scores','ldiag_scores')

## Text Analysis:

### Tokenize and remove stopwords
stop_words_df <- tibble(word = stopwords::stopwords("en", source = "smart"))

tokenize_text <- function(data, text_col) {
  data %>%
    select(HOSPITAL_EXPIRE_FLAG, text = {{ text_col }}) %>%
    mutate(text = replace_non_ascii(text)) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words_df)
}

### Count word frequency per class
word_counts <- function(tokens_df) {
  tokens_df %>%
    count(HOSPITAL_EXPIRE_FLAG, word) %>%
    pivot_wider(names_from = HOSPITAL_EXPIRE_FLAG, values_from = n, values_fill = 0) %>%
    rename(k = `1`, l = `0`)
}

### LLR function based on Crucial Nursing Description Extractor (CNDE)
calc_llr <- function(k, l, m, n) {
  total <- k + l + m + n
  N_hr <- k + m
  N_not_hr <- l + n
  N_total <- total
  
  ### Marginal and conditional probabilities
  pw <- (k + l) / total
  pwh <- k / (k + m)
  pwnh <- l / (l + n)
  
  ### Avoid log(0) or negative input to log by bounding probabilities
  eps <- 1e-10
  bound <- function(x) pmax(pmin(x, 1 - eps), eps)
  
  pw     <- bound(pw)
  pwh    <- bound(pwh)
  pwnh   <- bound(pwnh)
  one_pw <- bound(1 - pw)
  one_pwh <- bound(1 - pwh)
  one_pwnh <- bound(1 - pwnh)
  
  ### Numerator and denominator (as log terms)
  term1 <- k * log(pw / pwh)
  term2 <- m * log(one_pw / one_pwh)
  term3 <- l * log(pw / pwnh)
  term4 <- n * log(one_pw / one_pwnh)
  
  llr <- -2 * (term1 + term2 + term3 + term4)
  
  ifelse(is.nan(llr) | is.infinite(llr), 0, llr)
}

### Compute and normalize LLR
compute_llr <- function(counts_df, total_k, total_l) {
  counts_df %>%
    mutate(
      m = total_k - k,
      n = total_l - l,
      llr = calc_llr(k, l, m, n)
    ) %>%
    mutate(llr_scaled = (llr - min(llr)) / (max(llr) - min(llr))) %>%
    select(word, llr_scaled)
}

### Score each row by summing llr_scaled of matched words
score_text <- function(df, text_col, llr_table) {
  df %>%
    mutate(row_id = row_number()) %>%
    select(row_id, text = {{ text_col }}) %>%
    unnest_tokens(word, text) %>%
    inner_join(llr_table, by = "word") %>%
    group_by(row_id) %>%
    summarise(score = sum(llr_scaled), .groups = "drop")
}

### Totals
total_k <- sum(mimic_train$HOSPITAL_EXPIRE_FLAG == 1)
total_l <- sum(mimic_train$HOSPITAL_EXPIRE_FLAG == 0)

### DIAGNOSIS
diagnosis_tokens <- tokenize_text(mimic_train, DIAGNOSIS)
diagnosis_counts <- word_counts(diagnosis_tokens)
diagnosis_llr <- compute_llr(diagnosis_counts, total_k, total_l)
diagnosis_scores <- score_text(mimic_train, DIAGNOSIS, diagnosis_llr) %>%
  rename(diag_scores = score)

### SHORT_DIAGNOSE
short_tokens <- tokenize_text(mimic_train, SHORT_DIAGNOSE)
short_counts <- word_counts(short_tokens)
short_llr <- compute_llr(short_counts, total_k, total_l)
short_scores <- score_text(mimic_train, SHORT_DIAGNOSE, short_llr) %>%
  rename(sdiag_scores = score)

### LONG_DIAGNOSE
long_tokens <- tokenize_text(mimic_train, LONG_DIAGNOSE)
long_counts <- word_counts(long_tokens)
long_llr <- compute_llr(long_counts, total_k, total_l)
long_scores <- score_text(mimic_train, LONG_DIAGNOSE, long_llr) %>%
  rename(ldiag_scores = score)

### Merge scores back
mimic_train <- mimic_train %>%
  mutate(row_id = row_number()) %>%
  left_join(diagnosis_scores, by = "row_id") %>%
  left_join(short_scores, by = "row_id") %>%
  left_join(long_scores, by = "row_id") %>%
  mutate(across(c(diag_scores, sdiag_scores, ldiag_scores), ~replace_na(., 0))) %>%
  select(-c(DIAGNOSIS,SHORT_DIAGNOSE,LONG_DIAGNOSE))

### Score mimic_test using LLRs from mimic_train
mimic_test <- mimic_test %>%
  mutate(row_id = row_number())

diagnosis_test_scores <- score_text(mimic_test, DIAGNOSIS, diagnosis_llr) %>%
  rename(diag_scores = score)
short_test_scores <- score_text(mimic_test, SHORT_DIAGNOSE, short_llr) %>%
  rename(sdiag_scores = score)
long_test_scores <- score_text(mimic_test, LONG_DIAGNOSE, long_llr) %>%
  rename(ldiag_scores = score)

mimic_test <- mimic_test %>%
  left_join(diagnosis_test_scores, by = "row_id") %>%
  left_join(short_test_scores, by = "row_id") %>%
  left_join(long_test_scores, by = "row_id") %>%
  mutate(across(c(diag_scores, sdiag_scores, ldiag_scores), ~replace_na(., 0))) %>%
  select(-c(DIAGNOSIS,SHORT_DIAGNOSE,LONG_DIAGNOSE))

## Diagnosis analysis: 

### Get diagnosis appear more than 100 times
frequent_codes <- extra_diagnose %>%
  count(ICD9_CODE) %>%
  filter(n > 20) %>%
  pull(ICD9_CODE)

### Create diagnosis count matrix
diag_freq <- extra_diagnose %>%
  filter(ICD9_CODE %in% frequent_codes) %>%
  group_by(SUBJECT_ID, HADM_ID, ICD9_CODE) %>%
  summarise(value = n(), .groups = "drop")

### Pivot and join diagnosis data
diag_matrix <- pivot_wider(
  data = diag_freq,
  id_cols = c("SUBJECT_ID", "HADM_ID"),
  names_from = "ICD9_CODE",
  names_prefix = "ICD9_",
  values_from = "value",
  values_fill = list(value = 0)  # Explicit filling of NAs
)

train_diag <- mimic_train %>% 
  select(-ICD9_diagnosis) %>%
  rename(SUBJECT_ID = subject_id, HADM_ID = hadm_id) %>%
  left_join(diag_matrix, by = c("SUBJECT_ID", "HADM_ID"))

test_diag <- mimic_test %>% 
  select(-ICD9_diagnosis) %>%
  rename(SUBJECT_ID = subject_id, HADM_ID = hadm_id) %>%
  left_join(diag_matrix, by = c("SUBJECT_ID", "HADM_ID"))

train_diag[is.na(train_diag)] <- 0
test_diag[is.na(test_diag)] <- 0

y <- train_diag$HOSPITAL_EXPIRE_FLAG

train_diag <-train_diag %>% select(-HOSPITAL_EXPIRE_FLAG)

### One-hot encode
dummy <- dummyVars(~ ., data = train_diag, fullRank = TRUE)
X_mimic_train <- predict(dummy, newdata = train_diag) %>% as.data.frame()
X_mimic_test <- predict(dummy, newdata = test_diag) %>% as.data.frame()

### Align test features with training
missing_cols <- setdiff(names(X_mimic_train), names(X_mimic_test))
X_mimic_test[missing_cols] <- 0
X_mimic_test <- X_mimic_test[, names(X_mimic_train)]

### Scale numeric values
var_cols <- c(grep("^ADMISSION_TYPE", names(X_mimic_train), value = TRUE),
              grep("^INSURANCE", names(X_mimic_train), value = TRUE),
              grep("^FIRST_CAREUNIT", names(X_mimic_train), value = TRUE),
              grep("^ICD9", names(X_mimic_train), value = TRUE),
              grep("^GENDER", names(X_mimic_train), value = TRUE))
scale_cols <- setdiff(names(X_mimic_train), c("SUBJECT_ID", "HADM_ID","icustay_id", var_cols))
pp <- preProcess(X_mimic_train[, scale_cols], method = c("center", "scale"))
X_mimic_train <- predict(pp, X_mimic_train)
X_mimic_test <- predict(pp, X_mimic_test)

## Split data to cross-validation

set.seed(34667830)
mimic_train_clean <- mimic_train %>% select(all_of(vars_num_keep),ICD9_diagnosis,HOSPITAL_EXPIRE_FLAG)
mimic_test_clean <- mimic_test %>% select(all_of(vars_num_keep),ICD9_diagnosis)

mimic_train_no_ICD9 <- mimic_train_clean %>% select(all_of(vars_num_keep),HOSPITAL_EXPIRE_FLAG)
mimic_test_no_ICD9 <- mimic_test_clean %>% select(all_of(vars_num_keep))

mimic_folds <- vfold_cv(mimic_train_clean, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

mimic_folds_no_ICD9 <- vfold_cv(mimic_train_no_ICD9, v = 5, strata = HOSPITAL_EXPIRE_FLAG)


## Logistic regression:

### In sample evaluation
mimic_log_full <- glm(HOSPITAL_EXPIRE_FLAG ~ ., data = mimic_train_no_ICD9, family = binomial)

in_sample_preds <- mimic_train_no_ICD9 %>%
  mutate(prob = predict(mimic_log_full, type = "response"),
         pred_class = factor(ifelse(prob > 0.5, 1, 0), levels = c(0, 1)))

accuracy(in_sample_preds, truth = HOSPITAL_EXPIRE_FLAG, estimate = pred_class)
roc_auc(in_sample_preds, HOSPITAL_EXPIRE_FLAG, prob)

### Out sample evaluation
log_acc <- NULL
log_auc <- NULL
for (i in 1:5) {
  test_fold <- assessment(mimic_folds_no_ICD9$splits[[i]])
  train_fold <- analysis(mimic_folds_no_ICD9$splits[[i]])
  
  # Apply log model
  mimic_log <- glm(HOSPITAL_EXPIRE_FLAG ~ ., data = train_fold, family = binomial)
  test_fold <- test_fold |> 
    mutate(prob = predict(mimic_log, test_fold, type="response"),
           pred_class = factor(ifelse(prob > 0.5, 1, 0), levels = c(0, 1)))
  log_acc <- c(log_acc, accuracy(test_fold, HOSPITAL_EXPIRE_FLAG, pred_class)$.estimate)
  log_auc <- c(log_auc, roc_auc(test_fold, HOSPITAL_EXPIRE_FLAG, prob)$.estimate)
}

mean(log_acc)
mean(log_auc)

### Prediction 
log_prediction <- mimic_test %>%
  mutate(HOSPITAL_EXPIRE_FLAG = predict(mimic_log_full, newdata = mimic_test_no_ICD9, type = "response")) %>%
  select(ID = icustay_id, HOSPITAL_EXPIRE_FLAG)

write.csv(log_prediction, "logistic_regression.csv", row.names = FALSE)

## Decision tree:
set.seed(34667830)

tree_spec <- decision_tree(
  cost_complexity = tune(), 
  min_n = tune(),
  tree_depth = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

### Create workflow
tree_workflow <- workflow() %>%
  add_model(tree_spec) %>%
  add_formula(HOSPITAL_EXPIRE_FLAG ~.)

### Define grid
tree_grid <- grid_regular(
  cost_complexity(),
  min_n(),
  tree_depth(),
  levels = 3
)

### Tune the model
tune_dt_results <- tune_grid(
  tree_workflow,
  resamples = mimic_folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc, accuracy)
)

### Get best by AUC or Accuracy (Out-sample)
tune_dt_results %>% show_best(metric = "roc_auc")

best_tree <- select_best(tune_dt_results, metric = "roc_auc")

tree_result_acc_auc <-tune_dt_results |>
  collect_metrics() |>
  filter(.config == best_tree$.config)

tree_result_acc_auc

### Fit model
final_tree_workflow <- finalize_workflow(tree_workflow, best_tree)
final_tree_fit <- fit(final_tree_workflow, data = mimic_train_clean)

### In-sample evaluation
in_sample_preds <- mimic_train_clean %>%
  mutate(prob = predict(final_tree_fit, new_data = mimic_train_clean, type = "prob")$.pred_1,
         pred_class = predict(final_tree_fit, new_data = mimic_train_clean, type = "class")$.pred_class)

accuracy(in_sample_preds, truth = HOSPITAL_EXPIRE_FLAG, estimate = pred_class)
roc_auc(data = in_sample_preds,
        truth = HOSPITAL_EXPIRE_FLAG,prob,
        event_level = "second")

### Prediction
decision_tree_pred <- mimic_test %>%
  mutate(HOSPITAL_EXPIRE_FLAG = predict(final_tree_fit, new_data = mimic_test_clean, type = "prob")$.pred_1) %>%
  select(ID = icustay_id, HOSPITAL_EXPIRE_FLAG)
write.csv(decision_tree_pred, "decision_tree.csv", row.names = FALSE)

## Random Forest:
set.seed(34667830)
rf_spec <- rand_forest(mtry = 4, trees = 500) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_formula(HOSPITAL_EXPIRE_FLAG ~.)

rf_results <- fit_resamples(
  rf_workflow,
  resamples = mimic_folds_no_ICD9,
  metrics = metric_set(roc_auc, accuracy)
)

rf_result_acc_auc <- collect_metrics(rf_results)

rf_result_acc_auc

### Fit model
final_rf_fit <- fit(rf_workflow, data = mimic_train_no_ICD9)

### In-sample evaluation
in_sample_preds <- mimic_train_no_ICD9 %>%
  mutate(prob = predict(final_rf_fit, new_data = mimic_train_no_ICD9, type = "prob")$.pred_1,
         pred_class = predict(final_rf_fit, new_data = mimic_train_no_ICD9, type = "class")$`.pred_class`)

accuracy(in_sample_preds, truth = HOSPITAL_EXPIRE_FLAG, estimate = pred_class)
roc_auc(data = in_sample_preds,
        truth = HOSPITAL_EXPIRE_FLAG,prob,
        event_level = "second")

### Prediction
final_rf_fit <- fit(rf_workflow, data = mimic_train_no_ICD9)
rf_pred <- mimic_test %>%
  mutate(HOSPITAL_EXPIRE_FLAG = predict(final_rf_fit, new_data = mimic_test_no_ICD9, type = "prob")$.pred_1) %>%
  select(ID = icustay_id, HOSPITAL_EXPIRE_FLAG)

write.csv(rf_pred, "random_forest.csv", row.names = FALSE)

## Boosted tree:
set.seed(34667830)
bt_spec <- boost_tree(
  trees = 300,
  learn_rate = 0.05,
  tree_depth = 5
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

bt_workflow <- workflow() %>%
  add_model(bt_spec) %>%
  add_formula(HOSPITAL_EXPIRE_FLAG ~.)

bt_results <- fit_resamples(
  bt_workflow,
  resamples = mimic_folds_no_ICD9,
  metrics = metric_set(roc_auc, accuracy)
)

### Out-sample evaluation
bt_result_acc_auc <- collect_metrics(bt_results)

bt_result_acc_auc

### Fit model
final_bt_fit <- fit(bt_workflow, data = mimic_train_no_ICD9)

### In-sample evaluation
in_sample_preds <- mimic_train_no_ICD9 %>%
  mutate(prob = predict(final_bt_fit, new_data = mimic_train_no_ICD9, type = "prob")$.pred_1,
         pred_class = predict(final_bt_fit, new_data = mimic_train_no_ICD9, type = "class")$`.pred_class`)

accuracy(in_sample_preds, truth = HOSPITAL_EXPIRE_FLAG, estimate = pred_class)
roc_auc(data = in_sample_preds,
        truth = HOSPITAL_EXPIRE_FLAG,prob,
        event_level = "second")

### Prediction
boosted_tree_pred <- mimic_test %>%
  mutate(HOSPITAL_EXPIRE_FLAG = predict(final_bt_fit, new_data = mimic_test_no_ICD9, type = "prob")$.pred_1) %>%
  select(ID = icustay_id, HOSPITAL_EXPIRE_FLAG)
write.csv(boosted_tree_pred, "boosted_tree.csv", row.names = FALSE)

## Neural Network:

### Convert data to approraite format (matrix)
X_train_mat <- X_mimic_train %>% select(-c("SUBJECT_ID", "HADM_ID","icustay_id","row_id")) %>% as.matrix()
X_test_mat  <- X_mimic_test %>% select(-c("SUBJECT_ID", "HADM_ID","icustay_id","row_id")) %>% as.matrix()
y_vec <- as.numeric(y) -1

### Create model
network_model <- keras_model_sequential() |>
  layer_dense(units = 128, 
              input_shape = ncol(X_train_mat),
              kernel_regularizer = regularizer_l2(0.001)) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  layer_dropout(0.2) |>
  
  layer_dense(units = 64) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  
  layer_dense(units = 64) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  
  layer_dense(units = 32) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  
  layer_dense(units = 32) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  
  layer_dense(units = 1, activation = "sigmoid")

network_model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.0005), 
  loss = "binary_crossentropy",
  metrics = c("accuracy", "auc")
)

### Train model
set.seed(34667830)

history <- network_model |> fit(
  x = X_train_mat, 
  y = y_vec,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(monitor = "val_auc", patience = 10, restore_best_weights = TRUE)
  ),
  verbose = 1
)

### Prediction
pred_probs <- as.numeric(predict(network_model, X_test_mat))
pred_class <- factor(ifelse(pred_probs > 0.5, 1, 0), levels = c(0, 1))

neural_network <- mimic_test %>%
  select(ID = icustay_id) %>%
  mutate(HOSPITAL_EXPIRE_FLAG = pred_probs)

write.csv(neural_network, "neural_network.csv", row.names = FALSE)
