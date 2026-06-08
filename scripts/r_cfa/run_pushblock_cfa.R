#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  config <- list(
    input = "testdata/1/cfa_ready/episode_construct_metrics.csv",
    out_dir = "outputs/cfa",
    split_metric = "composite_score"
  )

  if (length(args) %% 2 != 0) {
    stop("Arguments must be passed as --name value pairs.", call. = FALSE)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- args[[i + 1]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    name <- substring(key, 3)
    if (!name %in% names(config)) {
      stop(sprintf("Unknown argument: %s", key), call. = FALSE)
    }
    config[[name]] <- value
    i <- i + 2
  }
  config
}

require_package <- function(name) {
  if (!requireNamespace(name, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed.", name), call. = FALSE)
  }
}

model_spec <- paste(
  "TaskAchievement =~ neg_final_goal_distance + best_progress_ratio + final_progress_ratio",
  "ProgressRetention =~ neg_goal_regression + neg_final_20pct_mean_distance",
  "ExecutionEfficiency =~ neg_episode_steps + log_progress_rate + block_movement_directness",
  "TaskAchievement ~~ ProgressRetention",
  "TaskAchievement ~~ ExecutionEfficiency",
  "ProgressRetention ~~ ExecutionEfficiency",
  sep = "\n"
)

indicator_columns <- c(
  "neg_final_goal_distance",
  "best_progress_ratio",
  "final_progress_ratio",
  "neg_goal_regression",
  "neg_final_20pct_mean_distance",
  "neg_episode_steps",
  "log_progress_rate",
  "block_movement_directness"
)

safe_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

compute_composite_score <- function(df) {
  z <- lapply(indicator_columns, function(col) {
    values <- safe_num(df[[col]])
    scale(values)
  })
  z_df <- as.data.frame(z)
  names(z_df) <- indicator_columns
  rowMeans(z_df, na.rm = TRUE)
}

fit_indices_table <- function(fit) {
  wanted <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")
  values <- lavaan::fitMeasures(fit, wanted)
  data.frame(
    measure = names(values),
    value = as.numeric(values),
    row.names = NULL
  )
}

extract_loading_table <- function(fit) {
  params <- lavaan::parameterEstimates(fit, standardized = TRUE)
  params[params$op == "=~", c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")]
}

extract_factor_correlation_table <- function(fit) {
  params <- lavaan::parameterEstimates(fit, standardized = TRUE)
  latent_names <- c("TaskAchievement", "ProgressRetention", "ExecutionEfficiency")
  params[
    params$op == "~~" & params$lhs %in% latent_names & params$rhs %in% latent_names & params$lhs != params$rhs,
    c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")
  ]
}

write_text_summary <- function(path, label, fit, fit_df, loading_df, corr_df, n_rows) {
  lines <- c(
    sprintf("Model: %s", label),
    sprintf("Rows used: %d", n_rows),
    "",
    "Fit indices:"
  )
  for (i in seq_len(nrow(fit_df))) {
    lines <- c(lines, sprintf("- %s: %.6f", fit_df$measure[[i]], fit_df$value[[i]]))
  }
  lines <- c(lines, "", "Standardized loadings:")
  for (i in seq_len(nrow(loading_df))) {
    lines <- c(
      lines,
      sprintf(
        "- %s =~ %s | est=%.6f std=%.6f p=%.6g",
        loading_df$lhs[[i]],
        loading_df$rhs[[i]],
        loading_df$est[[i]],
        loading_df$std.all[[i]],
        loading_df$pvalue[[i]]
      )
    )
  }
  lines <- c(lines, "", "Factor correlations:")
  for (i in seq_len(nrow(corr_df))) {
    lines <- c(
      lines,
      sprintf(
        "- %s ~~ %s | est=%.6f std=%.6f p=%.6g",
        corr_df$lhs[[i]],
        corr_df$rhs[[i]],
        corr_df$est[[i]],
        corr_df$std.all[[i]],
        corr_df$pvalue[[i]]
      )
    )
  }
  lines <- c(lines, "", capture.output(summary(fit, standardized = TRUE, fit.measures = TRUE)))
  writeLines(lines, con = path)
}

run_cfa <- function(df, label, out_dir) {
  usable <- df[, indicator_columns, drop = FALSE]
  complete_mask <- stats::complete.cases(usable)
  data_used <- df[complete_mask, , drop = FALSE]
  if (nrow(data_used) < 200) {
    stop(sprintf("Not enough complete rows for %s: %d", label, nrow(data_used)), call. = FALSE)
  }

  scaled <- as.data.frame(scale(data_used[, indicator_columns, drop = FALSE]))
  zero_sd_cols <- names(scaled)[vapply(scaled, function(x) any(is.na(x)), logical(1))]
  if (length(zero_sd_cols) > 0) {
    stop(sprintf("Zero-variance indicators in %s after scaling: %s", label, paste(zero_sd_cols, collapse = ", ")), call. = FALSE)
  }
  data_fit <- data_used
  data_fit[, indicator_columns] <- scaled

  fit <- suppressWarnings(lavaan::cfa(model_spec, data = data_fit, std.lv = TRUE, missing = "listwise", check.gradient = FALSE))
  converged <- isTRUE(lavaan::lavInspect(fit, "converged"))
  fit_df <- if (converged) fit_indices_table(fit) else data.frame(measure = character(), value = numeric())
  loadings <- extract_loading_table(fit)
  corrs <- extract_factor_correlation_table(fit)

  base_name <- gsub("[^A-Za-z0-9]+", "_", tolower(label))
  write.csv(data_fit, file.path(out_dir, sprintf("%s_input.csv", base_name)), row.names = FALSE)
  write.csv(fit_df, file.path(out_dir, sprintf("%s_fit_indices.csv", base_name)), row.names = FALSE)
  write.csv(loadings, file.path(out_dir, sprintf("%s_loadings.csv", base_name)), row.names = FALSE)
  write.csv(corrs, file.path(out_dir, sprintf("%s_factor_correlations.csv", base_name)), row.names = FALSE)
  summary_path <- file.path(out_dir, sprintf("%s_summary.txt", base_name))
  if (converged) {
    write_text_summary(summary_path, label, fit, fit_df, loadings, corrs, nrow(data_used))
  } else {
    lines <- c(
      sprintf("Model: %s", label),
      sprintf("Rows used: %d", nrow(data_used)),
      "Converged: FALSE",
      "",
      "The three-factor model did not converge for this sample.",
      "",
      "Standardized loadings from the non-converged fit attempt:",
      capture.output(print(loadings, row.names = FALSE)),
      "",
      "Latent factor correlations from the non-converged fit attempt:",
      capture.output(print(corrs, row.names = FALSE))
    )
    writeLines(lines, con = summary_path)
  }

  list(
    label = label,
    n = nrow(data_used),
    converged = converged,
    fit = fit_df,
    loadings = loadings,
    corrs = corrs
  )
}

metric_value <- function(result, measure) {
  if (!result$converged || nrow(result$fit) == 0) {
    return(NA_real_)
  }
  idx <- which(result$fit$measure == measure)
  if (length(idx) == 0) {
    return(NA_real_)
  }
  result$fit$value[[idx[[1]]]]
}

main <- function() {
  require_package("lavaan")
  config <- parse_args(commandArgs(trailingOnly = TRUE))

  input_path <- normalizePath(config$input, winslash = "/", mustWork = TRUE)
  out_dir <- config$out_dir
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  df <- read.csv(input_path, check.names = FALSE)
  missing_cols <- setdiff(indicator_columns, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf("Input is missing indicator columns: %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }

  for (col in indicator_columns) {
    df[[col]] <- safe_num(df[[col]])
  }

  df$composite_score <- compute_composite_score(df)
  split_cut <- stats::median(df$composite_score, na.rm = TRUE)
  df$performance_half <- ifelse(df$composite_score >= split_cut, "top_50", "bottom_50")

  full_result <- run_cfa(df, "full_sample_three_factor", out_dir)
  bottom_result <- run_cfa(df[df$performance_half == "bottom_50", , drop = FALSE], "bottom_50_three_factor", out_dir)
  top_result <- run_cfa(df[df$performance_half == "top_50", , drop = FALSE], "top_50_three_factor", out_dir)

  overview <- data.frame(
    model = c(full_result$label, bottom_result$label, top_result$label),
    n = c(full_result$n, bottom_result$n, top_result$n),
    converged = c(full_result$converged, bottom_result$converged, top_result$converged),
    chisq = c(metric_value(full_result, "chisq"), metric_value(bottom_result, "chisq"), metric_value(top_result, "chisq")),
    df = c(metric_value(full_result, "df"), metric_value(bottom_result, "df"), metric_value(top_result, "df")),
    cfi = c(metric_value(full_result, "cfi"), metric_value(bottom_result, "cfi"), metric_value(top_result, "cfi")),
    tli = c(metric_value(full_result, "tli"), metric_value(bottom_result, "tli"), metric_value(top_result, "tli")),
    rmsea = c(metric_value(full_result, "rmsea"), metric_value(bottom_result, "rmsea"), metric_value(top_result, "rmsea")),
    srmr = c(metric_value(full_result, "srmr"), metric_value(bottom_result, "srmr"), metric_value(top_result, "srmr"))
  )
  write.csv(overview, file.path(out_dir, "model_overview.csv"), row.names = FALSE)

  cat(sprintf("Wrote CFA outputs to %s\n", normalizePath(out_dir, winslash = "/", mustWork = FALSE)))
  print(overview)
}

main()