library(stm)
library(topicmodels)
library(ldatuning)
library(lsa)
library(data.table)
library(jsonlite)
# devtools::install_github("etam4260/kneedle")
library(kneedle)

# Default values for command line parameters
base_data_path <- "simul_data"
searchk_range <- 5
n_runs <- 4
search_k_permodelfit <- FALSE

# Handle command line arguments dynamically
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  # If the first argument does not start with --, treat it as the base path for backwards compatibility
  if (!startsWith(args[1], "--")) {
    base_data_path <- args[1]
  }
  
  # Parse named options
  for (i in seq_along(args)) {
    if (args[i] == "--base_path" && i < length(args)) {
      base_data_path <- args[i+1]
    }
    if (args[i] == "--searchk_range" && i < length(args)) {
      searchk_range <- as.numeric(args[i+1])
    }
    if (args[i] == "--n_runs" && i < length(args)) {
      n_runs <- as.numeric(args[i+1])
    }
    if (args[i] == "--search_k_permodelfit" && i < length(args)) {
      val <- toupper(args[i+1])
      search_k_permodelfit <- (val == "TRUE" || val == "T")
    }
  }
}

# Ensure the path is relative to the current working directory or absolute as provided
if (!dir.exists(base_data_path)) {
  stop(paste("Base data directory does not exist:", base_data_path))
}

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
                          n_runs = 5,
                          base_path = "simul_data",
                          search_k_permodelfit = FALSE) {
  
  path_prefix <- file.path(base_path, simul_name)
  path_prefix_save <- file.path(path_prefix, "model_fits")
  if (!dir.exists(path_prefix_save)) {
    dir.create(path_prefix_save, recursive = TRUE)
  }
  ##############
  #
  # loading data
  corpus_json_path <- paste0(path_prefix, "/_corpus.json")
  if (!file.exists(corpus_json_path)) {
    stop(paste("Unified corpus JSON file not found:", corpus_json_path))
  }
  
  corpus <- jsonlite::fromJSON(corpus_json_path, simplifyVector = TRUE)
  config <- corpus$config
  vocab <- corpus$vocab
  meta_df <- as.data.table(corpus$metadata)
  
  # Reconstruct dtm_matrix from sparse JSON DTM
  n_docs <- length(corpus$dtm$indices)
  v_size <- length(vocab)
  dtm_matrix <- matrix(0, nrow = n_docs, ncol = v_size)
  colnames(dtm_matrix) <- vocab
  
  for (i in seq_len(n_docs)) {
    indices <- corpus$dtm$indices[[i]]
    counts <- corpus$dtm$counts[[i]]
    if (length(indices) > 0) {
      dtm_matrix[i, indices + 1] <- counts
    }
  }
  
  # removing empty docs if any
  keep_docs <- rowSums(dtm_matrix) > 0
  if(sum(!keep_docs) > 0) {
    dtm_matrix <- dtm_matrix[keep_docs, ]
    meta_df <- meta_df[keep_docs, ]
  }
  
  # Remove words that do not appear in any document (columns with all zeros)
  # This is critical to prevent STM's "Word indices must be sequential integers starting with 1" error
  keep_vocab <- colSums(dtm_matrix) > 0
  if (sum(!keep_vocab) > 0) {
    dtm_matrix <- dtm_matrix[, keep_vocab, drop = FALSE]
    vocab <- vocab[keep_vocab]
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
  ##  search K
  ##
  k_range <- seq(max(2, true_k - searchk_range), true_k + searchk_range, by = 1)
  run_id <- paste0("sim_", simul_name, "_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  
  # clean the config list for logging
  clean_config <- lapply(config, function(x) {
    if (is.null(x) || length(x) == 0) return(NA)
    if (length(x) > 1) return(paste(x, collapse=","))
    return(x)
  })
  config_df <- as.data.frame(clean_config, stringsAsFactors = FALSE)

  # Global parameters outside the loop
  best_k_stm_global <- true_k
  best_k_lda_global <- true_k
  best_k_lda_run1 <- true_k

  if (!search_k_permodelfit) {
    # 1. Run deterministic/global STM searchK once outside the loop
    cat("\n--- Running searchK for STM --- \n")
    k_results_stm <- searchK(
      stm_data$documents, stm_data$vocab, K = k_range,
      prevalence =~ prev_covar, data = meta_df, verbose = T
    )
    stm_clean <- data.frame(
      topics = as.numeric(unlist(k_results_stm$results$K)),
      semcoh = as.numeric(unlist(k_results_stm$results$semcoh)),
      heldout = as.numeric(unlist(k_results_stm$results$heldout)),
      residual = as.numeric(unlist(k_results_stm$results$residual))
    )
    k_stm_semcoh <- find_elbow_kneedle(stm_clean$topics, stm_clean$semcoh, decreasing = FALSE)
    k_stm_held <- find_elbow_kneedle(stm_clean$topics, stm_clean$heldout, decreasing = FALSE)
    k_stm_res <- find_elbow_kneedle(stm_clean$topics, stm_clean$residual, decreasing = TRUE)
    best_k_stm_global <- get_consensus_k(c(k_stm_semcoh, k_stm_held, k_stm_res))
    if(is.na(best_k_stm_global)) best_k_stm_global <- true_k

    # Fit STM once globally (Deterministic: Moved out of stochastic loop)
    cat("\n--- Fitting Deterministic STM --- \n")
    m_stm <- stm(
      documents = stm_data$documents, vocab = stm_data$vocab, K = best_k_stm_global,
      prevalence =~ prev_covar, content =~ content_covar, data = meta_df,
      init.type = "Spectral", verbose = T
    )
    write.csv(m_stm$theta, paste0(path_prefix_save, "/stm_theta.csv"), row.names = FALSE)
    content_levels <- m_stm$settings$covariates$yvarlevels
    for (i in seq_along(content_levels)) {
      beta_matrix <- exp(m_stm$beta$logbeta[[i]])
      colnames(beta_matrix) <- stm_data$vocab
      write.csv(beta_matrix, paste0(path_prefix_save, "/stm_beta_group_", content_levels[i], ".csv"), row.names = FALSE)
    }

    # 2. Run traditional global Search K for LDA
    cat("\n--- Running ldatuning --- \n")
    k_results_lda <- FindTopicsNumber(
      dtm_matrix, topics = k_range,
      metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
      method = "Gibbs", verbose = T
    )
    lda_clean <- as.data.frame(lapply(k_results_lda, as.numeric))
    
    k_lda_gr <- find_elbow_kneedle(lda_clean$topics, lda_clean$Griffiths2004, decreasing = FALSE)
    k_lda_dev <- find_elbow_kneedle(lda_clean$topics, lda_clean$Deveaud2014, decreasing = FALSE)
    k_lda_cao <- find_elbow_kneedle(lda_clean$topics, lda_clean$CaoJuan2009, decreasing = TRUE)
    k_lda_arun <- find_elbow_kneedle(lda_clean$topics, lda_clean$Arun2010, decreasing = TRUE)
    
    best_k_lda_global <- get_consensus_k(c(k_lda_gr, k_lda_dev, k_lda_cao, k_lda_arun))
    if(is.na(best_k_lda_global)) best_k_lda_global <- true_k
    best_k_lda_run1 <- best_k_lda_global

    # Log master metrics once globally
    lda_long <- melt(as.data.table(lda_clean), id.vars = "topics", variable.name = "Metric", value.name = "Value")
    stm_long <- melt(as.data.table(stm_clean), id.vars = "topics", variable.name = "Metric", value.name = "Value")
    all_metrics_long <- rbind(lda_long, stm_long)
    wide_metrics <- dcast(all_metrics_long, . ~ Metric + topics, value.var = "Value")
    wide_metrics$. <- NULL
    
    master_log_row <- cbind(
      data.frame(Run_ID = run_id, True_K = true_k),
      config_df,
      wide_metrics,
      data.frame(
        Elbow_Griffiths = k_lda_gr, Elbow_Deveaud = k_lda_dev, Elbow_CaoJuan = k_lda_cao, Elbow_Arun = k_lda_arun,
        Elbow_SemCoh = k_stm_semcoh, Elbow_Heldout = k_stm_held, Elbow_Residual = k_stm_res,
        Final_Consensus_K_LDA = best_k_lda_global, Final_Consensus_K_STM = best_k_stm_global
      )
    )
    write.csv(master_log_row, paste0(path_prefix_save, "/simulation_master_log.csv"), row.names = FALSE)
  }

  #####################
  ##  stochastic fitting loop
  for (run in 1:n_runs) {
    cat(paste("\n--- Starting Model Fits: Run", run, "---\n"))
    run_dir <- file.path(path_prefix_save, paste0("run_", run))
    dir.create(run_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Local run variables
    current_best_k_lda <- best_k_lda_global
    
    if (search_k_permodelfit) {
      # Stochastic Search K for both LDA and STM (sampling is involved)
      # STM searchK is stochastic due to random held-out words partitioning
      cat(paste("\n--- Running local searchK (STM) for Run", run, "--- \n"))
      k_results_stm <- searchK(
        stm_data$documents, stm_data$vocab, K = k_range,
        prevalence =~ prev_covar, data = meta_df, verbose = T
      )
      stm_clean <- data.frame(
        topics = as.numeric(unlist(k_results_stm$results$K)),
        semcoh = as.numeric(unlist(k_results_stm$results$semcoh)),
        heldout = as.numeric(unlist(k_results_stm$results$heldout)),
        residual = as.numeric(unlist(k_results_stm$results$residual))
      )
      k_stm_semcoh <- find_elbow_kneedle(stm_clean$topics, stm_clean$semcoh, decreasing = FALSE)
      k_stm_held <- find_elbow_kneedle(stm_clean$topics, stm_clean$heldout, decreasing = FALSE)
      k_stm_res <- find_elbow_kneedle(stm_clean$topics, stm_clean$residual, decreasing = TRUE)
      current_best_k_stm <- get_consensus_k(c(k_stm_semcoh, k_stm_held, k_stm_res))
      if(is.na(current_best_k_stm)) current_best_k_stm <- true_k

      # Fit STM locally for this run (since its selected K is stochastic)
      cat(paste("\n--- Fitting STM for Run", run, "--- \n"))
      m_stm <- stm(
        documents = stm_data$documents, vocab = stm_data$vocab, K = current_best_k_stm,
        prevalence =~ prev_covar, content =~ content_covar, data = meta_df,
        init.type = "Spectral", verbose = T
      )
      write.csv(m_stm$theta, paste0(run_dir, "/stm_theta.csv"), row.names = FALSE)
      content_levels <- m_stm$settings$covariates$yvarlevels
      for (i in seq_along(content_levels)) {
        beta_matrix <- exp(m_stm$beta$logbeta[[i]])
        colnames(beta_matrix) <- stm_data$vocab
        write.csv(beta_matrix, paste0(run_dir, "/stm_beta_group_", content_levels[i], ".csv"), row.names = FALSE)
      }

      # Stochastic search K for LDA
      cat(paste("\n--- Running local ldatuning for Run", run, "--- \n"))
      k_results_lda <- FindTopicsNumber(
        dtm_matrix, topics = k_range,
        metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
        method = "Gibbs", verbose = T
      )
      lda_clean <- as.data.frame(lapply(k_results_lda, as.numeric))
      
      k_lda_gr <- find_elbow_kneedle(lda_clean$topics, lda_clean$Griffiths2004, decreasing = FALSE)
      k_lda_dev <- find_elbow_kneedle(lda_clean$topics, lda_clean$Deveaud2014, decreasing = FALSE)
      k_lda_cao <- find_elbow_kneedle(lda_clean$topics, lda_clean$CaoJuan2009, decreasing = TRUE)
      k_lda_arun <- find_elbow_kneedle(lda_clean$topics, lda_clean$Arun2010, decreasing = TRUE)
      
      current_best_k_lda <- get_consensus_k(c(k_lda_gr, k_lda_dev, k_lda_cao, k_lda_arun))
      if(is.na(current_best_k_lda)) current_best_k_lda <- true_k
      
      if (run == 1) {
        best_k_lda_run1 <- current_best_k_lda
      }

      # Log run-specific SearchK metrics locally
      run_metrics <- list(
        Elbow_Griffiths = k_lda_gr,
        Elbow_Deveaud = k_lda_dev,
        Elbow_CaoJuan = k_lda_cao,
        Elbow_Arun = k_lda_arun,
        Elbow_SemCoh = k_stm_semcoh,
        Elbow_Heldout = k_stm_held,
        Elbow_Residual = k_stm_res,
        Final_Consensus_K_LDA = current_best_k_lda,
        Final_Consensus_K_STM = current_best_k_stm
      )
      jsonlite::write_json(run_metrics, file.path(run_dir, "searchK_metrics.json"), auto_unbox = TRUE)
    }
    
    # LDA
    m_lda <- LDA(dtm_matrix, k = current_best_k_lda, method = "Gibbs")
    write.csv(posterior(m_lda)$topics, paste0(run_dir, "/lda_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_lda)$terms, paste0(run_dir, "/lda_beta_overall.csv"), row.names = FALSE)
    
    # CTM
    m_ctm <- CTM(dtm_matrix, k = current_best_k_lda)
    write.csv(posterior(m_ctm)$topics, paste0(run_dir, "/ctm_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_ctm)$terms, paste0(run_dir, "/ctm_beta_overall.csv"), row.names = FALSE)
  }

  # LSI (Deterministic: Moved out of stochastic loop)
  cat("\n--- Fitting Deterministic LSI --- \n")
  m_lsa <- lsa(t(dtm_matrix), dims = best_k_lda_run1)
  write.csv(m_lsa$dk, paste0(path_prefix_save, "/lsi_theta.csv"), row.names = FALSE)
  write.csv(t(m_lsa$tk), paste0(path_prefix_save, "/lsi_beta_overall.csv"), row.names = FALSE)
  
  cat("\n--- Simulations and fits completed, results saved to", path_prefix_save , "---\n")
}

# Recursively find all corpus directories containing _corpus.json under base_data_path
corpus_files <- list.files(base_data_path, pattern = "_corpus.json$", recursive = TRUE, full.names = FALSE)

for (corpus_file in corpus_files) {
  sim_name <- dirname(corpus_file)
  if (sim_name != ".") {
    cat(paste("\n=========================================\n"))
    cat(paste("Running model fits for simulation:", sim_name, "\n"))
    cat(paste("=========================================\n"))
    run_full_eval(
      simul_name = sim_name,
      searchk_range = searchk_range,
      n_runs = n_runs,
      base_path = base_data_path,
      search_k_permodelfit = search_k_permodelfit
    )
  }
}
