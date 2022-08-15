# Title: pt02_modeling
# Author: Justin Kringler
# Date: 9/14/2022

# Purpose: create random forest model

# Warnings: NA

# Building Model - Split data
vars<-c("type_number","height_m","weight_kg",
        "abilities_number", "hp","attack","defense","sp_attack","sp_defense",
        "speed","legendary")

trees_df <- data %>%
  select(vars)

set.seed(123)
trees_split <- initial_split(trees_df, strata = legendary)
trees_train <- training(trees_split)
trees_test <- testing(trees_split)


# Recipe on Training
tree_rec <- recipe(legendary ~ ., data = trees_train) %>%
  step_log(weight_kg) %>%
  step_log(height_m) %>%
  step_normalize(all_numeric_predictors()) %>%
  themis::step_downsample(legendary, under_ratio = 3) %>%
  themis::step_upsample(legendary, over_ratio = .8) 


tree_prep <- prep(tree_rec)
juiced <- juice(tree_prep)



# Tuning Specs
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")


# Tuning Workflow
```{r}
tune_wf <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tune_spec)
```

# Cross Validation
set.seed(234)
trees_folds <- vfold_cv(trees_train)

# Tune
doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 20
)
# Results AUC
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

# Select Best Model
best_auc <- select_best(tune_res, "roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_auc)

# Most impactful
final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(legendary ~ .,
      data = juice(tree_prep)
  ) %>%
  vip(geom = "point")