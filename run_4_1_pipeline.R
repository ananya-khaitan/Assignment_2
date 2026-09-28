###############################################################################
# SERB SURE Objectives 1, 2, 3 - R Pipeline (Optimized for Speed)
# Models: 10 ML Classifiers (kNN and ANN removed)
# Runs: IVs Only vs IVs + Controls
###############################################################################

library(haven)
library(glmnet)
library(randomForest)
library(caret)
library(xgboost)
library(rpart)
library(e1071)
library(pROC)
library(MLmetrics)
library(h2o)
library(ggplot2)

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-23")
h2o.init(nthreads = -1, max_mem_size = "12G")
h2o.no_progress() # Disable progress bars for clean logs

base_dir <- "C:/Users/ANANYA/Desktop/BITS/3-2/RC/Assignment 2/4-1"
setwd(base_dir)

dta_path <- file.path(base_dir, "HarmonizedGEM_2015_22Final", "HarmonizedGEM_2015_22Final.dta")
cat("Loading dataset (this may take a minute)...\n")
full_data <- read_dta(dta_path)
cat("Dataset loaded.\n")

# Define features based on Objectives document
base_sct <- c("suskill_2015_2022", "opportunity_perception", "fearfail_2015_2022",
              "knowing_entrepreneur", "entrepreneurial_intention", "media_coverage",
              "discent", "busang", "established_business_owner", "good_career_choice",
              "ease_start_business", "high_status", "social_enterprise")
legatum <- c("prosperity_index_score", "safety__security", "personal_freedom",
             "governance", "social_capital", "investment_environment",
             "enterprise_conditions", "market_access__infrastructure",
             "economic_quality", "living_conditions", "health", "education", "natural_environment")
efw <- c("EFW", "EFW_quartile", "EFW_government", "EFW_legal", "EFW_money",
         "EFW_trade", "EFW_regulation")
controls <- c("age", "gender", "education_attainment", "household_income",
              "hhsize", "occupation_status", "gdp_per_capita_ppp",
              "population_totalthousands", "gdp_growth")

# Define Independent Variables for each objective
obj_ivs <- list(
  "Obj1" = base_sct,
  "Obj2" = c(base_sct, legatum),
  "Obj3" = c(base_sct, efw)
)

specs   <- c("IV_Only", "IV_and_Controls")
targets <- c("TEA", "OME", "NME")
levels  <- c("Global", "India")

###############################################################################
# Helper: Evaluate Model
###############################################################################
evaluate_model <- function(model_name, y_true, y_prob, y_pred, out_dir, varimp_df = NULL) {
  metrics_csv   <- file.path(out_dir, "metrics", "Model_Metrics.csv")
  confusion_csv <- file.path(out_dir, "confusion_matrices", "Confusion_Tables.csv")
  
  # Calculate Metrics
  acc       <- Accuracy(y_pred, y_true)
  recall    <- Sensitivity(y_true, y_pred, positive = "Yes")
  precision <- Precision(y_true, y_pred, positive = "Yes")
  f1        <- F1_Score(y_true, y_pred, positive = "Yes")
  roc_obj   <- roc(y_true, y_prob, quiet = TRUE, levels=c("No", "Yes"), direction="<")
  auc_val   <- as.numeric(auc(roc_obj))
  
  met_row <- data.frame(Model = model_name, Accuracy = acc, Recall = recall,
                        Precision = precision, F1_Score = f1, AUC = auc_val)
  
  if (file.exists(metrics_csv)) {
    write.table(met_row, metrics_csv, sep = ",", append = TRUE, col.names = FALSE, row.names = FALSE)
  } else {
    write.table(met_row, metrics_csv, sep = ",", append = FALSE, col.names = TRUE, row.names = FALSE)
  }
  
  # Confusion Matrix
  cm <- table(Predicted = y_pred, Actual = y_true)
  tp <- ifelse("Yes" %in% rownames(cm) & "Yes" %in% colnames(cm), cm["Yes", "Yes"], 0)
  fp <- ifelse("Yes" %in% rownames(cm) & "No" %in% colnames(cm), cm["Yes", "No"], 0)
  tn <- ifelse("No" %in% rownames(cm) & "No" %in% colnames(cm), cm["No", "No"], 0)
  fn <- ifelse("No" %in% rownames(cm) & "Yes" %in% colnames(cm), cm["No", "Yes"], 0)
  
  cm_row <- data.frame(Model = model_name, TP = tp, FP = fp, TN = tn, FN = fn)
  if (file.exists(confusion_csv)) {
    write.table(cm_row, confusion_csv, sep = ",", append = TRUE, col.names = FALSE, row.names = FALSE)
  } else {
    write.table(cm_row, confusion_csv, sep = ",", append = FALSE, col.names = TRUE, row.names = FALSE)
  }
  
  # Save Variable Importance
  if (!is.null(varimp_df) && nrow(varimp_df) > 0) {
    varimp_df$Importance <- as.numeric(varimp_df$Importance) # Fix for order() xtfrm error
    csv_name <- file.path(out_dir, "variable_importance", "csv", paste0("VarImp_", gsub(" ", "_", model_name), ".csv"))
    write.csv(varimp_df, csv_name, row.names = FALSE, fileEncoding = "UTF-16LE")
    
    # Plot top 20
    plot_df <- head(varimp_df[order(varimp_df$Importance, decreasing = TRUE), ], 20)
    plot_df <- plot_df[plot_df$Importance > 0, ]
    if(nrow(plot_df) > 0) {
      plot_df$Feature <- factor(plot_df$Feature, levels = rev(plot_df$Feature))
      p <- ggplot(plot_df, aes(x = Feature, y = Importance, fill = Importance)) +
        geom_bar(stat = "identity", width = 0.7) +
        coord_flip() + theme_minimal() + labs(title = paste("VarImp -", model_name), x = NULL) +
        theme(legend.position = "none")
      ggsave(file.path(out_dir, "variable_importance", "plots", paste0("VarImp_", gsub(" ", "_", model_name), ".png")), p, width = 6, height = 5, dpi = 150)
    }
  }
}

###############################################################################
# MAIN PIPELINE LOOP
###############################################################################
for (obj_name in names(obj_ivs)) {
  for (spec in specs) {
    for (lvl in levels) {
      for (tgt in targets) {
        cat("\n=======================================================\n")
        cat(sprintf("Running: %s | Spec: %s | Level: %s | Target: %s\n", obj_name, spec, lvl, tgt))
        cat("=======================================================\n")
        
        out_dir <- file.path(base_dir, "outputs", obj_name, spec, lvl, tgt)
        dir.create(file.path(out_dir, "metrics"), recursive = TRUE, showWarnings = FALSE)
        dir.create(file.path(out_dir, "confusion_matrices"), recursive = TRUE, showWarnings = FALSE)
        dir.create(file.path(out_dir, "variable_importance", "csv"), recursive = TRUE, showWarnings = FALSE)
        dir.create(file.path(out_dir, "variable_importance", "plots"), recursive = TRUE, showWarnings = FALSE)
        
        # Determine features based on spec
        if (spec == "IV_Only") {
          current_features <- obj_ivs[[obj_name]]
        } else {
          current_features <- c(obj_ivs[[obj_name]], controls)
        }
        
        req_vars <- c(tgt, current_features)
        found_vars <- req_vars[req_vars %in% colnames(full_data)]
        
        # Level filter
        if (lvl == "India") {
          if ("country_name" %in% colnames(full_data)) {
            df <- full_data[full_data$country_name == "India", found_vars, drop=FALSE]
          } else {
            cat("Country column not found, skipping India.\n"); next
          }
        } else {
          df <- full_data[, found_vars, drop=FALSE]
        }
        
        # Remove target NAs
        df <- df[!is.na(df[[tgt]]), ]
        if (nrow(df) < 50) { cat("Not enough data, skipping.\n"); next }
        
        # Target transform
        tgt_vals <- as.numeric(df[[tgt]])
        df[[tgt]] <- factor(ifelse(tgt_vals > 0, "Yes", "No"), levels=c("No","Yes"))
        
        # Convert all to numeric (excluding target)
        for(v in colnames(df)) {
          if(v != tgt) df[[v]] <- as.numeric(df[[v]])
        }
        
        # Remove zero-variance columns (Very common in single-country slices)
        nzv <- nearZeroVar(df[, names(df) != tgt])
        if (length(nzv) > 0) {
          cols_to_drop <- names(df[, names(df) != tgt])[nzv]
          cat("Dropping zero-variance columns:", paste(cols_to_drop, collapse=", "), "\n")
          df <- df[, !colnames(df) %in% cols_to_drop]
        }
        
        # Remove cols with >15% NAs
        na_props <- colMeans(is.na(df))
        bad_cols <- names(na_props[na_props > 0.15])
        if (length(bad_cols) > 0) {
          cat("Dropping >15% NA columns:", paste(bad_cols, collapse=", "), "\n")
          df <- df[, !colnames(df) %in% bad_cols]
        }
        
        # Check if enough predictors are left
        if (ncol(df) < 2) { cat("Not enough predictors left, skipping.\n"); next }
        
        # Impute remaining
        for (v in colnames(df)) {
          if (v != tgt && any(is.na(df[[v]]))) {
            df[[v]][is.na(df[[v]])] <- median(df[[v]], na.rm = TRUE)
          }
        }
        
        # Split 70/30
        set.seed(123)
        pd <- sample(2, nrow(df), replace = TRUE, prob = c(0.7, 0.3))
        train <- df[pd == 1, ]
        test  <- df[pd == 2, ]
        
        y_test_true <- test[[tgt]]
        
        # Matrices
        train_x <- model.matrix(as.formula(paste(tgt, "~ . - 1")), data = train)
        train_y <- ifelse(train[[tgt]] == "Yes", 1, 0)
        test_x  <- model.matrix(as.formula(paste(tgt, "~ . - 1")), data = test)
        
        # H2O Data
        train_h2o <- as.h2o(train)
        test_h2o  <- as.h2o(test)
        x_h2o <- setdiff(colnames(train), tgt)
        
        ################ 1-4. GLM (Lasso, Ridge, EN, LR) via glmnet ################
        cat("Training GLM models...\n")
        # Weights
        w_yes <- sum(train_y == 0) / sum(train_y == 1)
        w_vec <- ifelse(train_y == 1, w_yes, 1.0)
        
        alphas <- c(Lasso = 1, Ridge = 0, Elastic_Net = 0.5)
        for (mdl in names(alphas)) {
          cv_fit <- cv.glmnet(train_x, train_y, alpha = alphas[[mdl]], family = "binomial", weights = w_vec)
          prob <- predict(cv_fit, newx = test_x, s = "lambda.min", type = "response")[, 1]
          pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
          
          coefs <- as.matrix(coef(cv_fit, s = "lambda.min"))
          varimp <- data.frame(Feature = rownames(coefs)[-1], Importance = abs(coefs[-1, 1]))
          evaluate_model(gsub("_", " ", mdl), y_test_true, prob, pred, out_dir, varimp)
        }
        
        # LR
        lr_fit <- glm(as.formula(paste(tgt, "~ .")), data=train, family="binomial", weights=w_vec)
        prob <- predict(lr_fit, newdata=test, type="response")
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi <- data.frame(Feature=names(coef(lr_fit))[-1], Importance=abs(coef(lr_fit)[-1]))
        vi[is.na(vi)] <- 0
        evaluate_model("Logistic Regression", y_test_true, prob, pred, out_dir, vi)
        
        ################ 5. Decision Tree ################
        cat("Training Decision Tree...\n")
        dt_fit <- rpart(as.formula(paste(tgt, "~ .")), data=train, method="class", weights=w_vec, control=rpart.control(cp=0.001))
        prob <- predict(dt_fit, newdata=test, type="prob")[,"Yes"]
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi <- data.frame(Feature=names(dt_fit$variable.importance), Importance=dt_fit$variable.importance)
        evaluate_model("Decision Tree", y_test_true, prob, pred, out_dir, vi)
        
        ################ 6. Naive Bayes ################
        cat("Training Naive Bayes...\n")
        nb_fit <- naiveBayes(as.formula(paste(tgt, "~ .")), data=train)
        prob <- predict(nb_fit, newdata=test, type="raw")[,"Yes"]
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        evaluate_model("Naive Bayes", y_test_true, prob, pred, out_dir, data.frame(Feature="None", Importance=0))
        
        ################ 7. Random Forest ################
        cat("Training Random Forest...\n")
        if(nrow(train) > 50000) {
          rf_fit <- randomForest(as.formula(paste(tgt, "~ .")), data=train[sample(nrow(train), 50000),], ntree=100)
        } else {
          rf_fit <- randomForest(as.formula(paste(tgt, "~ .")), data=train, ntree=100)
        }
        prob <- predict(rf_fit, newdata=test, type="prob")[,"Yes"]
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi <- data.frame(Feature=rownames(rf_fit$importance), Importance=rf_fit$importance[,1])
        evaluate_model("Random Forest", y_test_true, prob, pred, out_dir, vi)
        
        ################ 8. XGBoost (Fast CPU Hist) ################
        cat("Training XGBoost...\n")
        dtrain <- xgb.DMatrix(data = train_x, label = train_y, weight = w_vec)
        dtest  <- xgb.DMatrix(data = test_x)
        xgb_fit <- xgb.train(params = list(objective="binary:logistic", tree_method="hist", max_depth=6), 
                             data = dtrain, nrounds = 100, verbose=0)
        prob <- predict(xgb_fit, dtest)
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi_xgb <- xgb.importance(model = xgb_fit)
        vi <- data.frame(Feature=vi_xgb$Feature, Importance=vi_xgb$Gain)
        evaluate_model("XGBoost", y_test_true, prob, pred, out_dir, vi)
        
        ################ 9. H2O Random Forest ################
        cat("Training H2O Random Forest...\n")
        hrf_fit <- h2o.randomForest(x = x_h2o, y = tgt, training_frame = train_h2o,
                                    balance_classes = TRUE, ntrees = 100, max_depth = 10)
        prob <- as.data.frame(h2o.predict(hrf_fit, test_h2o))$Yes
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi_hrf <- as.data.frame(h2o.varimp(hrf_fit))
        vi <- data.frame(Feature=vi_hrf$variable, Importance=vi_hrf$scaled_importance)
        evaluate_model("H2O Random Forest", y_test_true, prob, pred, out_dir, vi)
        
        ################ 10. H2O GBM ################
        cat("Training H2O GBM...\n")
        hgbm_fit <- h2o.gbm(x = x_h2o, y = tgt, training_frame = train_h2o,
                            balance_classes = TRUE, ntrees = 100, max_depth = 6)
        prob <- as.data.frame(h2o.predict(hgbm_fit, test_h2o))$Yes
        pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels=c("No","Yes"))
        vi_hgbm <- as.data.frame(h2o.varimp(hgbm_fit))
        vi <- data.frame(Feature=vi_hgbm$variable, Importance=vi_hgbm$scaled_importance)
        evaluate_model("H2O GBM", y_test_true, prob, pred, out_dir, vi)
        
        cat(sprintf("=> Finished %s %s %s %s\n", obj_name, spec, lvl, tgt))
      }
    }
  }
}

cat("\nPIPELINE COMPLETE.\n")