# thread controls to prevent system crash when using multiple threads: Disable BLAS, Lapack, and OpenMP multithreading in parent R session - only needed if num_cores will be set to 2+
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
Sys.setenv(NUMEXPR_NUM_THREADS = "1")
Sys.setenv(R_PARALLEL_PORT = "random")

library(stm)
library(topicmodels)
library(ldatuning)
library(lsa)
library(data.table)
library(jsonlite)
# devtools::install_github("etam4260/kneedle")
library(kneedle)
library(parallel)
library(doParallel)
library(foreach)

# data.table is globally restricted to single-threaded operations - only needed if num_cores will be set to 2+
if (exists("setDTthreads")) {
  setDTthreads(1)
}
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(1)
  RhpcBLASctl::omp_set_num_threads(1)
}

# Safe sequential implementation of ldatuning::FindTopicsNumber to avoid process colision when running multiple cores
# This replacement evaluates the LDA models sequentially via lapply and calculates the exact same metrics.
safe_Arun2010 <- function(models, dtm) {
  len <- slam::row_sums(dtm)
  sapply(models, function(model) {
    if (is.null(model)) return(NA_real_)
    tryCatch({
      m1 <- exp(model@beta)
      m1.svd <- svd(m1)
      cm1 <- as.matrix(m1.svd$d)
      m2 <- model@gamma
      cm2 <- len %*% m2
      norm <- norm(as.matrix(len), type = "m")
      cm2 <- as.vector(cm2 / norm)
      # When k > V (number of topics exceeds vocabulary size), SVD yields min(k, V) = V singular values,
      # whereas cm2 has length k. In Arun (2010), both distributions must have length k.
      # We pad cm1 with small epsilon values to avoid subscript mismatch:
      if (length(cm1) != length(cm2)) {
        if (length(cm1) < length(cm2)) {
          cm1 <- c(as.vector(cm1), rep(.Machine$double.eps, length(cm2) - length(cm1)))
          cm1 <- cm1 / sum(cm1)
        } else {
          return(NA_real_)
        }
      }
      divergence <- sum(cm1 * log(cm1 / cm2)) + sum(cm2 * log(cm2 / cm1))
      return(divergence)
    }, error = function(e) NA_real_)
  })
}

FindTopicsNumber <- function(dtm, topics = seq(10, 40, by = 10),
                             metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
                             method = "Gibbs", control = list(), mc.cores = NA,
                             return_models = FALSE, verbose = FALSE, libpath = NULL) {
  if (length(topics[topics < 2]) != 0) {
    if (verbose) cat("warning: topics count can't to be less than 2, incorrect values was removed.\n")
    topics <- topics[topics >= 2]
  }
  topics <- sort(topics, decreasing = TRUE)
  if ("Griffiths2004" %in% metrics) {
    if (method == "VEM") {
      if (verbose) cat("'Griffiths2004' is incompatible with 'VEM' method, excluded.\n")
      metrics <- setdiff(metrics, "Griffiths2004")
    } else {
      if (!"keep" %in% names(control)) 
        control <- c(control, keep = 50)
    }
  }
  if (verbose) cat("fit models...")
  models <- lapply(topics, function(x) {
    if (!is.null(libpath)) {
      .libPaths(libpath)
    }
    tryCatch(
      topicmodels::LDA(dtm, k = x, method = method, control = control),
      error = function(e) NULL
    )
  })
  if (verbose) cat(" done.\n")
  if (verbose) cat("calculate metrics:\n")
  if (return_models && requireNamespace("tibble", quietly = TRUE)) {
    result <- cbind(topics, tibble::enframe(models, value = "LDA_model"))
    result$name <- NULL
  } else {
    if (return_models) {
      message("The tibble package is required for returning models. Returning results only.")
    }
    result <- data.frame(topics)
  }
  for (m in metrics) {
    if (verbose) cat(sprintf("  %s...", m))
    if (!m %in% c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")) {
      cat(" unknown!\n")
    } else {
      res_m <- tryCatch({
        switch(m,
          Griffiths2004 = ldatuning::Griffiths2004(models, control),
          CaoJuan2009   = ldatuning::CaoJuan2009(models),
          Arun2010      = safe_Arun2010(models, dtm),
          Deveaud2014   = ldatuning::Deveaud2014(models),
          NaN
        )
      }, error = function(e) {
        rep(NA_real_, length(topics))
      })
      if (length(res_m) != length(topics)) {
        res_m <- rep(NA_real_, length(topics))
      }
      result[m] <- res_m
      if (verbose) cat(" done.\n")
    }
  }
  return(result)
}

# Safe implementation of stm::searchK
# Upstream stm::searchK hardcodes M = 10 for exclusivity() and semanticCoherence() metrics.
# When a corpus has small vocabulary (e.g. fewer than 10 terms after min_word_freq filtering or heldout sampling),
# exclusivity() and semanticCoherence() attempt to slice [1:M, ] on the beta matrix, triggering:
# 'Error in (function (cond) : error in evaluating the argument 'x' in selecting a method for function 'mean': subscript out of bounds'
# Furthermore, if M < 2 or vocabulary is too small, matrix slicing dimensions collapse.
# This replacement dynamically bounds M to the heldout vocabulary size and provides graceful fallbacks.
searchK <- function(documents, vocab, K, init.type = "Spectral",
                    N = floor(0.1 * length(documents)), proportion = 0.5,
                    heldout.seed = NULL, M = 10, cores = 1, ...) {
  heldout <- stm::make.heldout(documents, vocab, N = N, proportion = proportion, seed = heldout.seed)
  v_heldout <- length(heldout$vocab)
  effective_M <- min(as.integer(M), as.integer(v_heldout))
  
  if ("content" %in% names(list(...))) {
    warning("Exclusivity calculation only designed for models without content covariates", call. = FALSE)
  }
  
  g <- list()
  for (i in seq_along(K)) {
    k <- K[i]
    if (v_heldout < 2 || effective_M < 2) {
      model <- tryCatch(
        stm::stm(documents = heldout$documents, vocab = heldout$vocab, K = k, init.type = init.type, ...),
        error = function(e) NULL
      )
      out <- list(
        K = k,
        exclus = NA_real_,
        semcoh = NA_real_,
        heldout = if (!is.null(model)) stm::eval.heldout(model, heldout$missing)$expected.heldout else NA_real_,
        residual = if (!is.null(model)) stm::checkResiduals(model, heldout$documents)$dispersion else NA_real_,
        bound = if (!is.null(model)) max(model$convergence$bound) else NA_real_,
        lbound = if (!is.null(model)) max(model$convergence$bound) + lfactorial(k) else NA_real_,
        em.its = if (!is.null(model)) length(model$convergence$bound) else 0L
      )
      g[[i]] <- out
    } else {
      stat <- tryCatch(
        stm:::get_statistics(k, heldout = heldout, init.type = init.type, M = effective_M, ...),
        error = function(e) {
          model <- tryCatch(
            stm::stm(documents = heldout$documents, vocab = heldout$vocab, K = k, init.type = init.type, ...),
            error = function(e2) NULL
          )
          list(
            K = k,
            exclus = NA_real_,
            semcoh = NA_real_,
            heldout = if (!is.null(model)) stm::eval.heldout(model, heldout$missing)$expected.heldout else NA_real_,
            residual = if (!is.null(model)) stm::checkResiduals(model, heldout$documents)$dispersion else NA_real_,
            bound = if (!is.null(model)) max(model$convergence$bound) else NA_real_,
            lbound = if (!is.null(model)) max(model$convergence$bound) + lfactorial(k) else NA_real_,
            em.its = if (!is.null(model)) length(model$convergence$bound) else 0L
          )
        }
      )
      g[[i]] <- stat
    }
  }
  g <- do.call("rbind", g)
  g <- as.data.frame(g)
  toreturn <- list(results = g, call = match.call(expand.dots = TRUE))
  class(toreturn) <- "searchK"
  return(toreturn)
}



log_pipeline_event <- function(base_path, phase, status, message) {
  log_path <- file.path(base_path, "simulation_pipeline.log")
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_line <- paste0("[", timestamp, "] [", toupper(phase), "] [", toupper(status), "] ", message, "\n")
  if (trimws(tolower(status)) == "start") {
    log_line <- paste0(strrep('-', 40), '\n', log_line)
  }
  if (trimws(tolower(status)) == "finish") {
    log_line <- paste0(log_line, strrep('-', 40), '\n')
  }
  cat(log_line, file = log_path, append = TRUE)
}

# default values for command line parameters
base_data_path <- "simul_data"  # where the corpora are located - saves output files next to each corpora independently
searchk_range <- 5              # +/- the number of grounód truth topics
n_runs <- 4                     # how many independent model fit for LDA, CTM (LSI and STM with spectral init are not stochastic)
search_k_permodelfit <- FALSE   # whether to run n_runs optimal number of topic search or just one per corpus
num_cores <- 1                  # number of CPU cores to use

# command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  # If the first argument does not start with --, treat it as the base path for backwards compatibility
  if (!startsWith(args[1], "--")) {
    base_data_path <- args[1]
  }
  
  # named options
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
    if (args[i] == "--cores" && i < length(args)) {
      num_cores <- as.numeric(args[i+1])
    }
  }
}

if (!dir.exists(base_data_path)) {
  stop(paste("Base data directory does not exist:", base_data_path))
}

# check available RAM on Linux
get_free_ram_gb <- function() {
  if (file.exists("/proc/meminfo")) {
    meminfo <- readLines("/proc/meminfo")
    mem_avail_line <- meminfo[grep("MemAvailable", meminfo)]
    if (length(mem_avail_line) > 0) {
      mem_avail_kb <- as.numeric(gsub("[^0-9]", "", mem_avail_line))
      return(mem_avail_kb / 1024 / 1024)
    }
  }
  return(NA) # Fallback if not Linux or fails
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

  if (max_freq > 1) {
    if (length(modes) == 1) {
      return(modes)
    } else {
      # If multiple modes exist, pick mode closest to their median
      med_modes <- median(modes)
      return(modes[which.min(abs(modes - med_modes))])
    }
  } else {
    return(floor(median(x)))  # fallback to median when no mode exists
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
  
  # check if entire folder fits are complete and skip already completed model fits (to be able to resume execution if needed)
  folder_complete <- TRUE
  
  # check global stm files
  if (!search_k_permodelfit) {
    if (!file.exists(file.path(path_prefix_save, "stm_theta.csv")) || !file.exists(file.path(path_prefix_save, "stm_beta_group_0.csv"))) {
      folder_complete <- FALSE
    }
    if (!file.exists(file.path(path_prefix_save, "simulation_master_log.csv"))) {
      folder_complete <- FALSE
    }
  }
  
  # check lsi files
  if (!file.exists(file.path(path_prefix_save, "lsi_theta.csv")) || !file.exists(file.path(path_prefix_save, "lsi_beta_overall.csv"))) {
    folder_complete <- FALSE
  }
  
  # check all stochastic runs
  if (folder_complete) {
    for (run in 1:n_runs) {
      run_dir <- file.path(path_prefix_save, paste0("run_", run))
      required_files <- c("lda_theta.csv", "lda_beta_overall.csv", "ctm_theta.csv", "ctm_beta_overall.csv")
      if (search_k_permodelfit) {
        required_files <- c(required_files, "stm_theta.csv", "stm_beta_group_0.csv", "searchK_metrics.json")
      }
      for (rf in required_files) {
        if (!file.exists(file.path(run_dir, rf))) {
          folder_complete <- FALSE
          break
        }
      }
      if (!folder_complete) break
    }
  }
  
  if (folder_complete) {
    msg <- paste0("Skipped fits for simulation: ", simul_name, " (already completely fitted)")
    cat(paste0("  [Skip Folder] ", msg, "\n"))
    log_pipeline_event(base_path, "fitting", "skipped", msg)
    return()
  }

  ##############
  #
  # loading data
  corpus_json_path <- file.path(path_prefix, "_corpus.json")
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

  # Run garbage collection preemptively
  gc(verbose = FALSE)

  if (!search_k_permodelfit) {
    # resume check: check if global master log already exists
    master_log_path <- file.path(path_prefix_save, "simulation_master_log.csv")
    if (file.exists(master_log_path)) {
      cat(paste0("  [Skip Step] Global SearchK logs for ", simul_name, " are complete. Loading parameters...\n"))
      master_log <- tryCatch(read.csv(master_log_path, stringsAsFactors = FALSE), error = function(e) fread(master_log_path))
      best_k_lda_global <- as.numeric(master_log$Final_Consensus_K_LDA[1])
      best_k_stm_global <- as.numeric(master_log$Final_Consensus_K_STM[1])
      if (is.na(best_k_lda_global) || length(best_k_lda_global) == 0) best_k_lda_global <- true_k
      if (is.na(best_k_stm_global) || length(best_k_stm_global) == 0) best_k_stm_global <- true_k
      best_k_lda_run1 <- best_k_lda_global
      log_pipeline_event(base_path, "fitting", "skipped", paste0("global SearchK logs loaded from master_log for ", simul_name))
    } else {
      cat(paste0("  [", simul_name, "] Running STM searchK across K = [", paste(k_range, collapse = ", "), "]...\n"))
      flush.console()
      # global STM searchK once outside the loop
      k_results_stm <- searchK(
        stm_data$documents, stm_data$vocab, K = k_range,
        prevalence =~ prev_covar, data = meta_df, verbose = FALSE
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

      gc(verbose = FALSE)

      # traditional global Search K for LDA
      cat(paste0("  [", simul_name, "] Running LDA FindTopicsNumber across K = [", paste(k_range, collapse = ", "), "]...\n"))
      flush.console()
      k_results_lda <- FindTopicsNumber(
        dtm_matrix, topics = k_range,
        metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
        method = "Gibbs", verbose = FALSE
      )
      lda_clean <- as.data.frame(lapply(k_results_lda, as.numeric))
      
      k_lda_gr <- find_elbow_kneedle(lda_clean$topics, lda_clean$Griffiths2004, decreasing = FALSE)
      k_lda_dev <- find_elbow_kneedle(lda_clean$topics, lda_clean$Deveaud2014, decreasing = FALSE)
      k_lda_cao <- find_elbow_kneedle(lda_clean$topics, lda_clean$CaoJuan2009, decreasing = TRUE)
      k_lda_arun <- find_elbow_kneedle(lda_clean$topics, lda_clean$Arun2010, decreasing = TRUE)
      
      best_k_lda_global <- get_consensus_k(c(k_lda_gr, k_lda_dev, k_lda_cao, k_lda_arun))
      if(is.na(best_k_lda_global)) best_k_lda_global <- true_k
      best_k_lda_run1 <- best_k_lda_global

      gc(verbose = FALSE)

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
      write.csv(master_log_row, master_log_path, row.names = FALSE)
      log_pipeline_event(base_path, "fitting", "success", paste0("SearchK ldatuning completed and saved for ", simul_name))
    }

    # resume check: STM fit
    stm_complete <- file.exists(file.path(path_prefix_save, "stm_theta.csv")) && file.exists(file.path(path_prefix_save, "stm_beta_group_0.csv"))
    if (stm_complete) {
      cat(paste0("  [Skip Step] STM fit is already complete. Skipping...\n"))
      log_pipeline_event(base_path, "fitting", "skipped", paste0("STM skipped for ", simul_name))
    } else {
      cat(paste0("  [", simul_name, "] Fitting STM (K = ", best_k_stm_global, ")...\n"))
      flush.console()
      # Fit STM once globally
      m_stm <- stm(
        documents = stm_data$documents, vocab = stm_data$vocab, K = best_k_stm_global,
        prevalence =~ prev_covar, content =~ content_covar, data = meta_df,
        init.type = "Spectral", verbose = FALSE
      )
      write.csv(m_stm$theta, paste0(path_prefix_save, "/stm_theta.csv"), row.names = FALSE)
      content_levels <- m_stm$settings$covariates$yvarlevels
      for (i in seq_along(content_levels)) {
        beta_matrix <- exp(m_stm$beta$logbeta[[i]])
        colnames(beta_matrix) <- stm_data$vocab
        write.csv(beta_matrix, paste0(path_prefix_save, "/stm_beta_group_", content_levels[i], ".csv"), row.names = FALSE)
      }
      log_pipeline_event(base_path, "fitting", "success", paste0("STM fit saved to ", path_prefix_save, " for ", simul_name))
      gc(verbose = FALSE)
    }
  }

  #####################
  ##  stochastic fitting loop
  for (run in 1:n_runs) {
    run_dir <- file.path(path_prefix_save, paste0("run_", run))
    dir.create(run_dir, showWarnings = FALSE, recursive = TRUE)
    
    # resume check: individual runs
    run_complete <- TRUE
    required_files <- c("lda_theta.csv", "lda_beta_overall.csv", "ctm_theta.csv", "ctm_beta_overall.csv")
    if (search_k_permodelfit) {
      required_files <- c(required_files, "stm_theta.csv", "stm_beta_group_0.csv", "searchK_metrics.json")
    }
    for (rf in required_files) {
      if (!file.exists(file.path(run_dir, rf))) {
        run_complete <- FALSE
        break
      }
    }
    
    if (run_complete) {
      cat(paste0("  [Skip Step] Stochastic run ", run, " for ", simul_name, " is already complete. Skipping...\n"))
      log_pipeline_event(base_path, "fitting", "skipped", paste0("run ", run, " skipped for ", simul_name))
      
      # recover best_k_lda_run1 if needed for LSI SVD
      if (run == 1) {
        if (search_k_permodelfit) {
          run_metrics_path <- file.path(run_dir, "searchK_metrics.json")
          if (file.exists(run_metrics_path)) {
            run_metrics <- fromJSON(run_metrics_path)
            best_k_lda_run1 <- as.numeric(run_metrics$Final_Consensus_K_LDA)
            if (is.na(best_k_lda_run1) || length(best_k_lda_run1) == 0) best_k_lda_run1 <- true_k
          }
        } else {
          best_k_lda_run1 <- best_k_lda_global
          if (is.na(best_k_lda_run1) || length(best_k_lda_run1) == 0) best_k_lda_run1 <- true_k
        }
      }
      next
    }
    
    # local run variables
    current_best_k_lda <- best_k_lda_global
    
    if (search_k_permodelfit) {
      # stochastic Search K inside loop
      k_results_stm <- searchK(
        stm_data$documents, stm_data$vocab, K = k_range,
        prevalence =~ prev_covar, data = meta_df, verbose = FALSE
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
      m_stm <- stm(
        documents = stm_data$documents, vocab = stm_data$vocab, K = current_best_k_stm,
        prevalence =~ prev_covar, content =~ content_covar, data = meta_df,
        init.type = "Spectral", verbose = FALSE
      )
      write.csv(m_stm$theta, paste0(run_dir, "/stm_theta.csv"), row.names = FALSE)
      content_levels <- m_stm$settings$covariates$yvarlevels
      for (i in seq_along(content_levels)) {
        beta_matrix <- exp(m_stm$beta$logbeta[[i]])
        colnames(beta_matrix) <- stm_data$vocab
        write.csv(beta_matrix, paste0(run_dir, "/stm_beta_group_", content_levels[i], ".csv"), row.names = FALSE)
      }

      gc(verbose = FALSE)

      # Stochastic search K for LDA
      k_results_lda <- FindTopicsNumber(
        dtm_matrix, topics = k_range,
        metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
        method = "Gibbs", verbose = FALSE
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
      
      gc(verbose = FALSE)
    }
    
    # LDA
    # current_seed <- as.integer(42 + run * 1000)
    cat(paste0("  [", simul_name, "] Run ", run, "/", n_runs, ": Fitting LDA Gibbs (k=", current_best_k_lda, ")...\n"))
    flush.console()
    m_lda <- LDA(
      dtm_matrix, k = current_best_k_lda, method = "Gibbs",
      # control = list(seed = current_seed, iter = 1000, burnin = 100)
      control = list(iter = 1000, burnin = 100)
    )
    write.csv(posterior(m_lda)$topics, paste0(run_dir, "/lda_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_lda)$terms, paste0(run_dir, "/lda_beta_overall.csv"), row.names = FALSE)
    
    # CTM
    cat(paste0("  [", simul_name, "] Run ", run, "/", n_runs, ": Fitting CTM (k=", current_best_k_lda, ")...\n"))
    flush.console()
    ctm_control <- list(
      # seed = current_seed,
      em = list(iter.max = 500, tol = 1e-3),
      var = list(iter.max = 250, tol = 1e-4),
      cg = list(iter.max = 250, tol = 1e-4)
    )
    m_ctm <- tryCatch({
      CTM(dtm_matrix, k = current_best_k_lda, control = ctm_control)
    }, error = function(e) {
      warning(paste0("CTM fit warning on run ", run, " for ", simul_name, ": ", e$message, ". Retrying with basic VEM..."))
      # CTM(dtm_matrix, k = current_best_k_lda, method = "VEM", control = list(seed = current_seed, em = list(iter.max = 100)))
      CTM(dtm_matrix, k = current_best_k_lda, method = "VEM", control = list(em = list(iter.max = 500)))
    })
    write.csv(posterior(m_ctm)$topics, paste0(run_dir, "/ctm_theta.csv"), row.names = FALSE)
    write.csv(posterior(m_ctm)$terms, paste0(run_dir, "/ctm_beta_overall.csv"), row.names = FALSE)
    
    log_pipeline_event(base_path, "fitting", "success", paste0("Stochastic run ", run, " complete for ", simul_name))
    gc(verbose = FALSE)
  }

  # resume check: LSI
  lsi_complete <- file.exists(file.path(path_prefix_save, "lsi_theta.csv")) && file.exists(file.path(path_prefix_save, "lsi_beta_overall.csv"))
  if (lsi_complete) {
    cat(paste0("  [Skip Step] LSI fit is already complete. Skipping...\n"))
    log_pipeline_event(base_path, "fitting", "skipped", paste0("LSI skipped for ", simul_name))
  } else {
    # LSI
    k_lsi <- best_k_lda_run1
    if (is.na(k_lsi) || length(k_lsi) == 0) k_lsi <- true_k
    max_lsi_dims <- max(1, min(nrow(dtm_matrix), ncol(dtm_matrix)) - 1)
    k_lsi <- max(1, min(as.integer(k_lsi), as.integer(max_lsi_dims)))
    cat(paste0("  [", simul_name, "] Fitting LSI (dims = ", k_lsi, ")...\n"))
    flush.console()
    m_lsa <- lsa(t(dtm_matrix), dims = k_lsi)
    write.csv(m_lsa$dk, paste0(path_prefix_save, "/lsi_theta.csv"), row.names = FALSE)
    write.csv(t(m_lsa$tk), paste0(path_prefix_save, "/lsi_beta_overall.csv"), row.names = FALSE)
    log_pipeline_event(base_path, "fitting", "success", paste0("LSI fit complete for ", simul_name))
    gc(verbose = FALSE)
  }
}

# recursively find all corpus directories containing _corpus.json under base_data_path
corpus_files <- list.files(base_data_path, pattern = "_corpus.json$", recursive = TRUE, full.names = FALSE)

# Log fitting pipeline start
log_pipeline_event(base_data_path, "fitting", "start", paste0("Initiating R model fitting pipeline. Cores: ", num_cores, ", runs: ", n_runs))

# safe memory check and core limiter
avail_cores <- parallel::detectCores(logical = FALSE)
if (is.na(avail_cores) || avail_cores == 0) {
  avail_cores <- parallel::detectCores()
}

# setup progress tracker directory (atomic, lock-free parallel progress tracker)
progress_dir <- file.path(base_data_path, ".progress_tracker")
if (dir.exists(progress_dir)) {
  unlink(progress_dir, recursive = TRUE)
}
dir.create(progress_dir, showWarnings = FALSE, recursive = TRUE)

# estimate safe core allocations based on physical available RAM on Linux
free_ram <- get_free_ram_gb()
if (!is.na(free_ram) && length(corpus_files) > 0) {
  # load the first corpus to estimate average memory footprint of a single worker
  first_json_path <- file.path(base_data_path, corpus_files[1])
  if (file.exists(first_json_path)) {
    first_corpus <- jsonlite::fromJSON(first_json_path, simplifyVector = TRUE)
    n_docs_est <- length(first_corpus$dtm$indices)
    v_size_est <- length(first_corpus$vocab)
    
    if (is.list(first_corpus$config$num_topics) || length(first_corpus$config$num_topics) > 1) {
      true_k_est <- max(unlist(first_corpus$config$num_topics))
    } else {
      true_k_est <- as.numeric(first_corpus$config$num_topics)
    }
    if (is.na(true_k_est) || length(true_k_est) == 0) true_k_est <- 5
    
    # Mathematical scaling formula for STM/ldatuning memory footprint (in GB)
    est_ram_per_worker <- 0.6 + (n_docs_est * v_size_est * (true_k_est + searchk_range)) / (2.5e7 * 2)  # this is just eyeballed
    if (est_ram_per_worker < 1.0) est_ram_per_worker <- 1.0
    
    safe_cores_mem <- floor(free_ram / est_ram_per_worker)
    if (safe_cores_mem < 1) safe_cores_mem <- 1
    
    cat(paste0("\n--- estimating memory need ---"))
    cat(paste0("\nused corpus size: ", n_docs_est, " docs, ", v_size_est, " vocabulary terms, max topic count: ", true_k_est + searchk_range))
    cat(paste0("\nestimated peak RAM per worker: ", round(est_ram_per_worker, 2), " GB (available: ", round(free_ram, 1), " GB)"))
    
    if (num_cores > safe_cores_mem) {
      warning(paste("Restricting parallel workers from", num_cores, "to", safe_cores_mem, 
                    "to prevent Out-Of-Memory (OOM) crash. Available system RAM:", round(free_ram, 1), "GB"))
      num_cores <- safe_cores_mem
    }
    cat(paste0("\n-----------------------------------\n"))
  }
}

if (num_cores > avail_cores) {
  warning(paste("Requested cores (", num_cores, ") exceeds system physical cores (", avail_cores, "). Limiting to", avail_cores))
  num_cores <- avail_cores
}

if (num_cores > 1) {
  cat(paste("\nSpawning parallel fitting pool with", num_cores, "cores...\n"))
  
  if (.Platform$OS.type == "unix") {
    # On Unix/Linux: Use parallel::mclapply. It uses copy-on-write process forks,
    # consumes significantly less memory than PSOCK sockets, and natively flushes
    # worker stdout (cat/print statements) directly to the parent terminal in real-time!
    cat("Using mclapply fork pipeline\n")
    
    results <- mclapply(corpus_files, function(corpus_file) {
      # Lock thread counts in spawned child processes
      Sys.setenv(OMP_NUM_THREADS = "1")
      Sys.setenv(MKL_NUM_THREADS = "1")
      Sys.setenv(OPENBLAS_NUM_THREADS = "1")
      Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
      Sys.setenv(NUMEXPR_NUM_THREADS = "1")
      if (requireNamespace("data.table", quietly = TRUE)) {
        data.table::setDTthreads(1)
      }
      if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        RhpcBLASctl::blas_set_num_threads(1)
        RhpcBLASctl::omp_set_num_threads(1)
      }
      if (exists("defaultClusterOptions", envir = asNamespace("parallel"))) {
        # Randomize default cluster port to prevent collisions if any package attempts to spawn a PSOCK cluster
        try(assign("port", sample(11500:29000, 1), envir = parallel:::defaultClusterOptions), silent = TRUE)
      }
      
      sim_name <- dirname(corpus_file)
      if (sim_name != ".") {
        # Atomic progress logging
        started_token <- file.path(progress_dir, paste0("started_", gsub("/", "_", sim_name)))
        writeLines("", started_token)
        
        started_count <- length(list.files(progress_dir, pattern = "^started_"))
        total_sims <- length(corpus_files)
        
        # Check if entire folder is already completed to print starting log correctly
        folder_complete <- TRUE
        path_prefix <- file.path(base_data_path, sim_name)
        path_prefix_save <- file.path(path_prefix, "model_fits")
        
        if (!search_k_permodelfit) {
          if (!file.exists(file.path(path_prefix_save, "stm_theta.csv")) || !file.exists(file.path(path_prefix_save, "stm_beta_group_0.csv"))) {
            folder_complete <- FALSE
          }
          if (!file.exists(file.path(path_prefix_save, "simulation_master_log.csv"))) {
            folder_complete <- FALSE
          }
        }
        if (!file.exists(file.path(path_prefix_save, "lsi_theta.csv")) || !file.exists(file.path(path_prefix_save, "lsi_beta_overall.csv"))) {
          folder_complete <- FALSE
        }
        if (folder_complete) {
          for (run in 1:n_runs) {
            run_dir <- file.path(path_prefix_save, paste0("run_", run))
            required_files <- c("lda_theta.csv", "lda_beta_overall.csv", "ctm_theta.csv", "ctm_beta_overall.csv")
            if (search_k_permodelfit) {
              required_files <- c(required_files, "stm_theta.csv", "stm_beta_group_0.csv", "searchK_metrics.json")
            }
            for (rf in required_files) {
              if (!file.exists(file.path(run_dir, rf))) {
                folder_complete <- FALSE
                break
              }
            }
            if (!folder_complete) break
          }
        }
        
        if (folder_complete) {
          # Log skipped to console and file-system tracker
          finished_token <- file.path(progress_dir, paste0("finished_", gsub("/", "_", sim_name)))
          writeLines("", finished_token)
          finished_count <- length(list.files(progress_dir, pattern = "^finished_"))
          cat(paste0("\n[Parallel Worker (", finished_count, "/", total_sims, " Skipped)] <<< SKIPPED fits for simulation: ", sim_name, " (already complete)\n"))
          log_pipeline_event(base_data_path, "fitting", "skipped", paste0("Simulation skipped: ", sim_name))
        } else {
          cat(paste0("\n[Parallel Worker (", started_count, "/", total_sims, " Started)] >>> STARTED fits for simulation: ", sim_name, "\n"))
          
          start_time_sim <- Sys.time()
          run_full_eval(
            simul_name = sim_name,
            searchk_range = searchk_range,
            n_runs = n_runs,
            base_path = base_data_path,
            search_k_permodelfit = search_k_permodelfit
          )
          end_time_sim <- Sys.time()
          duration <- round(as.numeric(difftime(end_time_sim, start_time_sim, units = "secs")), 1)
          
          finished_token <- file.path(progress_dir, paste0("finished_", gsub("/", "_", sim_name)))
          writeLines("", finished_token)
          
          finished_count <- length(list.files(progress_dir, pattern = "^finished_"))
          
          cat(paste0("\n[Parallel Worker (", finished_count, "/", total_sims, " Finished)] <<< COMPLETED fits for simulation: ", sim_name, " in ", duration, " seconds\n"))
          log_pipeline_event(base_data_path, "fitting", "success", paste0("Completed fits for simulation ", sim_name, " in ", duration, "s"))
        }
      }
      return(TRUE)
    }, mc.cores = num_cores)
    
    # Check for child process errors
    failed_indices <- which(sapply(results, function(res) inherits(res, "try-error") || is.null(res)))
    if (length(failed_indices) > 0) {
      for (idx in failed_indices) {
        err <- results[[idx]]
        file_name <- corpus_files[idx]
        if (inherits(err, "try-error")) {
          warning(paste0("Worker for '", file_name, "' failed with error: ", as.character(err)))
        } else if (is.null(err)) {
          warning(paste0("Worker for '", file_name, "' terminated abnormally or was killed (e.g. OOM/crash)."))
        }
      }
      log_pipeline_event(base_data_path, "fitting", "failure", paste(length(failed_indices), "parallel fitting worker(s) failed."))
      stop(paste(length(failed_indices), "parallel fitting worker(s) failed with errors."))
    }
    
  } else {
    # Fallback to standard sockets for Windows
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    
    # Run evaluations in parallel
    foreach(corpus_file = corpus_files, 
            .packages = c("stm", "topicmodels", "ldatuning", "lsa", "data.table", "jsonlite", "kneedle"),
            .export = c("run_full_eval", "find_elbow_kneedle", "find_elbow", "get_consensus_k", "FindTopicsNumber", "safe_Arun2010", "searchK")) %dopar% {
      
      # Redundant assurance
      Sys.setenv(OMP_NUM_THREADS = "1")
      Sys.setenv(MKL_NUM_THREADS = "1")
      Sys.setenv(OPENBLAS_NUM_THREADS = "1")
      Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
      Sys.setenv(NUMEXPR_NUM_THREADS = "1")
      if (requireNamespace("data.table", quietly = TRUE)) {
        data.table::setDTthreads(1)
      }
      
      sim_name <- dirname(corpus_file)
      if (sim_name != ".") {
        run_full_eval(
          simul_name = sim_name,
          searchk_range = searchk_range,
          n_runs = n_runs,
          base_path = base_data_path,
          search_k_permodelfit = search_k_permodelfit
        )
      }
    }
    stopCluster(cl)
  }
} else {
  # Sequential fallback
  for (corpus_file in corpus_files) {
    sim_name <- dirname(corpus_file)
    if (sim_name != ".") {
      cat(paste("\n=========================================\n"))
      cat(paste("Running model fits for simulation:", sim_name, "\n"))
      cat(paste("=========================================\n"))
      start_time_sim <- Sys.time()
      run_full_eval(
        simul_name = sim_name,
        searchk_range = searchk_range,
        n_runs = n_runs,
        base_path = base_data_path,
        search_k_permodelfit = search_k_permodelfit
      )
      duration <- round(as.numeric(difftime(Sys.time(), start_time_sim, units = "secs")), 1)
      log_pipeline_event(base_data_path, "fitting", "success", paste0("Completed fits sequentially for simulation ", sim_name, " in ", duration, "s"))
    }
  }
}

# Clean up progress tracker folder
if (dir.exists(progress_dir)) {
  unlink(progress_dir, recursive = TRUE)
}

# Log fitting pipeline finish
log_pipeline_event(base_data_path, "fitting", "finish", "Model fitting pipeline completed successfully.")
