library(stm)
library(topicmodels)
library(ldatuning)
library(lsa)
library(data.table)
library(jsonlite)
# devtools::install_github("etam4260/kneedle")
library(kneedle)

# params to change for each simulated data
setwd("D:/bj/documents/munk/analysis/LDA_simul/src")

# maximum distance
find_elbow <- function(k_values, metric_values, minimize = FALSE) {
  # Normalize to 0-1 scale
  x <- (k_values - min(k_values)) / (max(k_values) - min(k_values))
  y <- (metric_values - min(metric_values)) / (max(metric_values) - min(metric_values))
  
  # invert for minimization
  if(minimize) { y <- 1 - y }
  
  x1 <- x[1]
  y1 <- y[1]
  x2 <- x[length(x)]
  y2 <- y[length(y)]
  # perpendicular distance from he line
  distances <- abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1) / sqrt((y2 - y1)^2 + (x2 - x1)^2)
  
  # k value with the maximum distance
  return(k_values[which.max(distances)])
}

find_elbow_kneedle <- function(k_values, metric_values, decreasing = FALSE) {
  tryCatch({
    elbow_k <- kneedle(k_values, metric_values, decreasing = decreasing)[1]
    return(as.numeric(elbow_k))
  }, error = function(msg) {
    message(paste("kneedle algorithm unsuccessful:", msg$message))
    return(NA)
  })
}

plot_elbow <- function(k_values, metric_values, minimize = FALSE, metric_name = "") {
  if (metric_name == "") {
    metric_name <- "metric"
  }
  k <- kneedle(k_values, metric_values, decreasing = minimize)[1]
  plot(k_values, metric_values, type = "l",
       xlab="k", ylab=metric_name)
  lines(k_values[c(1, length(k_values))], metric_values[c(1, length(k_values))],
        lty=2)
  points(k, metric_values[k_values==k], col="red", pch=16)
}

# mode if exists, otherwise median
get_consensus_k <- function(x) {
  x <- na.omit(x)
  if (length(x) == 0) return(NA)
  
  freq_table <- table(x)
  max_freq <- max(freq_table)
  modes <- as.numeric(names(freq_table)[freq_table == max_freq])

  if (length(modes) == 1) {
    return(modes)
  } else {
    return(floor(median(x)))  # fallback to median
  }
}

run_full_eval <- function(simul_name,
                          searchk_range = 5,
                          n_runs = 5) {
  
  path_prefix <- paste0("../simul_data/", simul_name)
  path_prefix_save <- paste0(path_prefix, "/model_fits/")
  if (!dir.exists(path_prefix_save)) {
    dir.create(path_prefix_save, recursive = TRUE)
  }
  ##############
  #
  # loading data
  dtm_df <- fread(paste0(path_prefix, "_dtm.csv"))
  meta_df <- fread(paste0(path_prefix, "_meta.csv"))
  config <- read_json(paste0(path_prefix, "_config.json"))
  
  dtm_matrix <- as.matrix(dtm_df)
  vocab <- colnames(dtm_matrix)
  
  # removing empty docs if any
  keep_docs <- rowSums(dtm_matrix) > 0
  if(sum(!keep_docs) > 0) {
    dtm_matrix <- dtm_matrix[keep_docs, ]
    meta_df <- meta_df[keep_docs, ]
  }
  
  meta_df$prev_covar <- as.factor(meta_df$prev_covar)
  meta_df$content_covar <- as.factor(meta_df$content_covar)
  
  stm_data <- readCorpus(dtm_matrix, type = "Matrix")
  stm_data$vocab <- vocab
  
  # Extract k
  if (is.list(config$num_topics) || length(config$num_topics) > 1) {
    true_k <- max(unlist(config$num_topics))
  } else if (is.character(config$num_topics)) {
    true_k <- max(as.numeric(unlist(regmatches(config$num_topics, gregexpr("[0-9]+", config$num_topics)))))
  } else {
    true_k <- as.numeric(config$num_topics)
  }
  
  ############
  ##
  ##  search K  # TODO: move inside simulation to have multiple results and see variance
  ##
  k_range <- seq(max(2, true_k - searchk_range), true_k + searchk_range, by = 1)
  run_id <- paste0("sim_", simul_name, "_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  
  cat("\n--- Running ldatuning --- \n")
  # Griffiths2004: maximizing log-likelihood (//perpl)
  # CaoJuan2009: inverse cosine similarity of topics
  # Arun2010: SVD based KL divergence of the topic-word and topic-document distributions
  # Deveaud2014: maximizes topic distance (Shannon divergence based)
  k_results_lda <- FindTopicsNumber(
    dtm_matrix, topics = k_range,
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
    method = "Gibbs", verbose = T
  )
  cat("\n--- Running searchK for STM --- \n")
  k_results_stm <- searchK(
    stm_data$documents, stm_data$vocab, K = k_range,
    prevalence =~ prev_covar, data = meta_df, verbose = T
  )
  
  lda_clean <- as.data.frame(lapply(k_results_lda, as.numeric))
  stm_clean <- data.frame(
    topics = as.numeric(unlist(k_results_stm$results$K)),
    semcoh = as.numeric(unlist(k_results_stm$results$semcoh)),
    heldout = as.numeric(unlist(k_results_stm$results$heldout)),
    residual = as.numeric(unlist(k_results_stm$results$residual))
  )
  
  k_lda_gr <- find_elbow_kneedle(lda_clean$topics, lda_clean$Griffiths2004, decreasing = FALSE)
  k_lda_dev <- find_elbow_kneedle(lda_clean$topics, lda_clean$Deveaud2014, decreasing = FALSE)
  k_lda_cao <- find_elbow_kneedle(lda_clean$topics, lda_clean$CaoJuan2009, decreasing = TRUE)
  k_lda_arun <- find_elbow_kneedle(lda_clean$topics, lda_clean$Arun2010, decreasing = TRUE)
  
  k_stm_semcoh <- find_elbow_kneedle(stm_clean$topics, stm_clean$semcoh, decreasing = FALSE)
  k_stm_held <- find_elbow_kneedle(stm_clean$topics, stm_clean$heldout, decreasing = FALSE)
  k_stm_res <- find_elbow_kneedle(stm_clean$topics, stm_clean$residual, decreasing = TRUE)
  
  # consensus
  best_k_lda <- get_consensus_k(c(k_lda_gr, k_lda_dev, k_lda_cao, k_lda_arun))
  best_k_stm <- get_consensus_k(c(k_stm_semcoh, k_stm_held, k_stm_res))
  # fallback in case kneedle fails entirely
  if(is.na(best_k_lda)) {
    print("kneedle algorithm failed for LDA")
    best_k_lda <- true_k
  }
  if(is.na(best_k_stm)) {
    print("kneedle algorithm failed for STM")
    best_k_stm <- true_k
  }
  
  ##########
  #
  # log
  # clean the config list
  clean_config <- lapply(config, function(x) {
    # Catch NULLs or empty lists and replace with NA
    if (is.null(x) || length(x) == 0) {
      return(NA)
    } 
    # Collapse multi-item lists into a single string
    if (length(x) > 1) {
      return(paste(x, collapse=","))
    } 
    # Otherwise, return the item
    return(x)
  })
  config_df <- as.data.frame(clean_config, stringsAsFactors = FALSE)
  
  # Pivot to wide format
  lda_long <- melt(as.data.table(lda_clean), id.vars = "topics", variable.name = "Metric", value.name = "Value")
  stm_long <- melt(as.data.table(stm_clean), id.vars = "topics", variable.name = "Metric", value.name = "Value")
  # setnames(stm_long, "K", "topics")
  all_metrics_long <- rbind(lda_long, stm_long)
  # cretae cols MetricName_K
  wide_metrics <- dcast(all_metrics_long, . ~ Metric + topics, value.var = "Value")
  wide_metrics$. <- NULL
  
  # Combine into one row
  master_log_row <- cbind(
    data.frame(Run_ID = run_id, True_K = true_k),
    config_df,
    wide_metrics,
    data.frame(
      Elbow_Griffiths = k_lda_gr, Elbow_Deveaud = k_lda_dev, Elbow_CaoJuan = k_lda_cao, Elbow_Arun = k_lda_arun,
      Elbow_SemCoh = k_stm_semcoh, Elbow_Heldout = k_stm_held, Elbow_Residual = k_stm_res,
      Final_Consensus_K_LDA = best_k_lda, Final_Consensus_K_STM = best_k_stm
    )
  )
  
  write.csv(master_log_row, paste0(path_prefix_save, "simulation_master_log.csv"), row.names = FALSE)
  
  #####################
  ##
  ##  fitting models
  
  for (run in 1:n_runs) {
    cat(paste("\n--- Starting Model Fits: Run", run, "---\n"))
    run_dir <- paste0(path_prefix_save, "run_", run)
    dir.create(run_dir, showWarnings = FALSE)
    
    # LDA
    m_lda <- LDA(dtm_matrix, k = best_k_lda, method = "Gibbs")
    write.csv(posterior(m_lda)$topics, paste0(run_dir, "/lda_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_lda)$terms, paste0(run_dir, "/lda_beta_overall.csv"), row.names = FALSE)
    
    # CTM
    m_ctm <- CTM(dtm_matrix, k = best_k_lda)
    write.csv(posterior(m_ctm)$topics, paste0(run_dir, "/ctm_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_ctm)$terms, paste0(run_dir, "/ctm_beta_overall.csv"), row.names = FALSE)
    
    # STM
    m_stm <- stm(
      documents = stm_data$documents, vocab = stm_data$vocab, K = best_k_stm,
      prevalence =~ prev_covar, content =~ content_covar, data = meta_df,
      init.type = "Spectral", verbose = T
    )
    write.csv(m_stm$theta, paste0(run_dir, "/stm_theta.csv"), row.names = FALSE)
    
    # Extract specific beta matrices for each content group
    content_levels <- m_stm$settings$covariates$yvarlevels
    for (i in seq_along(content_levels)) {
      beta_matrix <- exp(m_stm$beta$logbeta[[i]])
      colnames(beta_matrix) <- stm_data$vocab
      write.csv(beta_matrix, paste0(run_dir, "/stm_beta_group_", content_levels[i], ".csv"), row.names = FALSE)
    }
  }
  # LSI
  m_lsa <- lsa(t(dtm_matrix), dims = best_k_lda)
  write.csv(m_lsa$dk, paste0(path_prefix_save, "lsi_theta.csv"), row.names = FALSE)
  write.csv(t(m_lsa$tk), paste0(path_prefix_save, "lsi_beta_overall.csv"), row.names = FALSE)
  
  cat("\n--- Simulations completed, results saved to", path_prefix_save , "---\n")
}

run_full_eval(simul_name = "trial01", searchk_range = 5, n_runs = 4)
