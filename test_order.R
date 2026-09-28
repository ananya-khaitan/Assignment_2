varimp_df <- data.frame(Feature = c("A", "B"), Importance = as.matrix(c(1, 2)))
varimp_df[order(varimp_df, decreasing = TRUE), ]
