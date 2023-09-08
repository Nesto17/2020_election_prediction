# Load required libraries
library(tidyverse)
library(tidymodels)
library(stacks)

library(xgboost)
library(lightgbm)
library(bonsai)
library(dbarts)
library(ranger)
library(glmnet)
library(kernlab)
library(LiblineaR)
library(Cubist)
library(rules)

# Reading the datasets
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Removing unnecessary columns
train <- train %>% select(-c(id, name)) # Name is not a predictor.

# Removing features which contain the amount of zeroes > 25% of the whole dataset
treshold <- 0.25
cols_to_remove <- colnames(train)[colSums(train == 0) / nrow(train) > treshold]
cols_to_remove <- cols_to_remove[!is.na(cols_to_remove)]
train <- train[, !(names(train) %in% cols_to_remove)]

# ------------------------------------------------------------

# Creating 10 cross-validation folds
set.seed(2020)
folds <- vfold_cv(train, v = 10, strata = percent_dem)

# Preprocessing recipe
rec <- recipe(percent_dem ~ ., data = train) %>% 
  step_impute_mean(all_numeric_predictors()) %>% # Impute NAs with mean 
  step_mutate(x2013_code = factor(x2013_code)) %>% # Mutate categorical features
  step_corr(all_numeric_predictors(), threshold = 0.999) %>% # Remove predictors that are extremely correlated
  step_nzv(all_numeric_predictors()) %>% # Remove predictors that are highly sparse
  step_log(all_numeric_predictors(), offset = 1) %>% # Log transform numerical predictors 
  step_normalize(all_numeric_predictors()) %>% # Scale numerical predictors
  step_dummy(all_nominal()) # Encode categorical features

# Define metric set
metric <- metric_set(rmse)

# ------------------------------------------------------------

# Model specifications
# Model 1 - XGBoost
xgboost_spec <- boost_tree(
  trees = 888,
  min_n = 39,
  tree_depth = 12,
  learn_rate = 0.09
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Model 2 - Light GBM
lightgbm_spec <- parsnip::boost_tree(
  trees = 1000,
  tree_depth = 13, 
  min_n = 20
) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

# Model 3 - BART
bart_spec <- parsnip::bart(
  trees = 333,
  prior_terminal_node_coef = 0.978,
  prior_terminal_node_expo = 2.28
) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

# Model 4 - Random Forest
randforest_spec <- parsnip::rand_forest(
  trees = 1510,
  min_n = 2
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Model 5 - GLM Net
glmnet_spec <- parsnip::linear_reg(
  penalty = 0.01, 
  mixture = 0.9
) %>%
  set_engine("glmnet") %>% 
  set_mode("regression")

# Model 6 - SVM with Radial Kernel
svmrbf_spec <- parsnip::svm_rbf(
  cost = 40,
  rbf_sigma = 0.005,
  margin = 0.015
) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

# Model 7 - Linear SVM
svmlin_spec <- parsnip::svm_linear(
  cost = 5,
  margin = 0.14
) %>% 
  set_engine("LiblineaR") %>% 
  set_mode("regression")

# Model 8 - Cubist rule-based Regression
cubist_spec <- parsnip::cubist_rules(
  committees = 83,
  neighbors = 0,
  max_rules = 101
) %>% 
  set_engine("Cubist") %>% 
  set_mode("regression")

model_specs <- list(xgboost_spec, lightgbm_spec, bart_spec, randforest_spec,
                    glmnet_spec, svmrbf_spec, svmlin_spec, cubist_spec)
names(model_specs) <- c("xgboost", "lightgbm", "bart", "randforest",
                        "glmnet", "svmrbf", "svmlin", "cubist")

# ------------------------------------------------------------

# Individually create workflows
wfls <- list()
for (spec in names(model_specs)) {
  wfl <- workflow() %>% 
    add_model(model_specs[[spec]]) %>% 
    add_recipe(rec)
  
  wfl_name <- paste(c(spec, "_wfl"), collapse = "")
  wfls[[wfl_name]] <- wfl
}

# Fit validation folds to each workflow
results <- list()
for (wfl in names(wfls)) {
  res <- wfls[[wfl]] %>% fit_resamples(
    resamples = folds,
    metrics = metric,
    control = control_stack_resamples()
  )
  
  res_name <- paste(c(wfl, "res"), collapse = "")
  results[[res_name]] <- res
}

# ------------------------------------------------------------

# Constructing data stack
xgboost <- results$xgboost_wflres
lightgbm <- results$lightgbm_wflres
bart <- results$bart_wflres
randforest <- results$randforest_wflres
glmnet <- results$glmnet_wflres
svm_rbf <- results$svmrbf_wflres
svm_lin <- results$svmlin_wflres
cubist <- results$cubist_wflres

data_stack <- stacks() %>% 
  add_candidates(xgboost) %>% 
  add_candidates(lightgbm) %>% 
  add_candidates(bart) %>% 
  add_candidates(randforest) %>% 
  add_candidates(glmnet) %>% 
  add_candidates(svm_rbf) %>% 
  add_candidates(svm_lin) %>% 
  add_candidates(cubist)

# Constructing final model stack with non-zero stacking coefficients
set.seed(2023)
model_stack <- data_stack %>%
  blend_predictions() %>% 
  fit_members()

# ------------------------------------------------------------

# Making predictions
pred <- model_stack %>% 
  predict(test)

# Creating prediction tibble and export
pred <- test %>% 
  select(id) %>% 
  bind_cols(pred)

write_csv(pred, "pred.csv")




