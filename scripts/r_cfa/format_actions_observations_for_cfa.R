#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  defaults <- list(
    actions = "testdata/1/actions_frame_train_01.csv",
    observations = "testdata/1/observations_frame_train_01.csv",
    out_dir = "testdata/1/cfa_ready",
    near_goal_threshold = 1.0,
    contact_threshold = 1.0,
    speed_threshold = 0.01,
    success_reward_threshold = 1.0,
    action_epsilon = 1e-8
  )

  if (length(args) %% 2 != 0) {
    stop("Arguments must be provided as --name value pairs.")
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- args[[i + 1]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument '%s'. Expected --name value pairs.", key))
    }

    name <- substring(key, 3)
    if (!name %in% names(defaults)) {
      stop(sprintf("Unknown argument '%s'.", key))
    }

    if (name %in% c("near_goal_threshold", "contact_threshold", "speed_threshold", "success_reward_threshold", "action_epsilon")) {
      defaults[[name]] <- as.numeric(value)
    } else {
      defaults[[name]] <- value
    }
    i <- i + 2
  }

  defaults
}

coalesce_num <- function(x, default = 0) {
  x[is.na(x)] <- default
  x
}

safe_ratio <- function(num, den) {
  if (is.na(num) || is.na(den) || den == 0) {
    return(NA_real_)
  }
  num / den
}

require_columns <- function(df, required, source_name) {
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing required columns: %s", source_name, paste(missing, collapse = ", ")))
  }
}

sort_by_episode_step <- function(df) {
  keys <- intersect(c("episode_id", "frame_count", "training_step", "step_index"), names(df))
  if (length(keys) == 0) {
    return(df)
  }
  df[do.call(order, unname(df[keys])), , drop = FALSE]
}

enrich_observations <- function(obs) {
  obs$block_goal_distance <- sqrt((obs$block_pos_x - obs$goal_pos_x) ^ 2 + (obs$block_pos_z - obs$goal_pos_z) ^ 2)
  obs$block_speed <- sqrt(coalesce_num(obs$block_vel_x) ^ 2 + coalesce_num(obs$block_vel_z) ^ 2)
  obs$agent_block_distance <- sqrt((obs$agent_pos_x - obs$block_pos_x) ^ 2 + (obs$agent_pos_z - obs$block_pos_z) ^ 2)
  obs
}

prepare_actions <- function(actions) {
  actions$action_x <- coalesce_num(as.numeric(actions$action_x))
  actions$action_y <- coalesce_num(as.numeric(actions$action_y))
  actions$action_magnitude <- sqrt(actions$action_x ^ 2 + actions$action_y ^ 2)
  actions
}

merge_decision_rows <- function(actions, obs) {
  decision_obs <- obs[obs$is_decision == 0, , drop = FALSE]
  merge_keys <- c("episode_id", "frame_count", "training_step", "step_index")
  merged <- merge(
    actions,
    decision_obs,
    by = merge_keys,
    all = FALSE,
    suffixes = c("_action", "_obs")
  )
  sort_by_episode_step(merged)
}

compute_episode_metrics <- function(obs, merged, cfg) {
  obs_split <- split(obs, obs$episode_id, drop = TRUE)
  merged_split <- split(merged, merged$episode_id, drop = TRUE)
  episodes <- sort(as.integer(names(obs_split)))
  rows <- vector("list", length(episodes))

  for (idx in seq_along(episodes)) {
    episode_id <- episodes[[idx]]
    ep_obs <- sort_by_episode_step(obs_split[[as.character(episode_id)]])
    ep_merged <- merged_split[[as.character(episode_id)]]
    if (is.null(ep_merged)) {
      ep_merged <- merged[0, , drop = FALSE]
    } else {
      ep_merged <- sort_by_episode_step(ep_merged)
    }
    if (nrow(ep_obs) == 0) {
      next
    }

    distances <- ep_obs$block_goal_distance
    valid_distances <- which(!is.na(distances))
    if (length(valid_distances) == 0) {
      next
    }

    start_idx <- valid_distances[[1]]
    end_idx <- valid_distances[[length(valid_distances)]]
    best_idx <- valid_distances[[which.min(distances[valid_distances])]]

    start_goal_distance <- distances[[start_idx]]
    final_goal_distance <- distances[[end_idx]]
    best_goal_distance <- distances[[best_idx]]
    final_progress_ratio <- safe_ratio(start_goal_distance - final_goal_distance, start_goal_distance)
    best_progress_ratio <- safe_ratio(start_goal_distance - best_goal_distance, start_goal_distance)
    goal_regression <- final_goal_distance - best_goal_distance

    near_goal_fraction <- mean(distances <= cfg$near_goal_threshold, na.rm = TRUE)
    after_best <- distances[best_idx:length(distances)]
    stay_near_after_best <- if (length(after_best) > 0) {
      mean(after_best <= cfg$near_goal_threshold, na.rm = TRUE)
    } else {
      NA_real_
    }

    tail_count <- max(1L, ceiling(length(valid_distances) * 0.2))
    tail_idx <- tail(valid_distances, tail_count)
    final_20pct_mean_distance <- mean(distances[tail_idx], na.rm = TRUE)

    step_span <- range(ep_obs$step_index, na.rm = TRUE)
    episode_steps <- if (all(is.finite(step_span))) {
      as.integer(step_span[[2]] - step_span[[1]] + 1)
    } else {
      nrow(ep_obs)
    }

    time_span <- range(ep_obs$realtime_since_start, na.rm = TRUE)
    episode_time <- if (all(is.finite(time_span))) {
      as.numeric(time_span[[2]] - time_span[[1]])
    } else {
      NA_real_
    }

    progress_rate <- safe_ratio(final_progress_ratio, episode_steps)
    log_progress_rate <- if (!is.na(progress_rate)) {
      sign(progress_rate) * log1p(abs(progress_rate))
    } else {
      NA_real_
    }

    block_dx <- diff(ep_obs$block_pos_x)
    block_dz <- diff(ep_obs$block_pos_z)
    block_path_length <- sum(sqrt(block_dx ^ 2 + block_dz ^ 2), na.rm = TRUE)
    block_movement_directness <- safe_ratio(start_goal_distance - final_goal_distance, block_path_length)

    engagement_fraction <- mean(ep_obs$agent_block_distance <= cfg$contact_threshold, na.rm = TRUE)
    active_push_fraction <- mean(
      ep_obs$agent_block_distance <= cfg$contact_threshold & ep_obs$block_speed > cfg$speed_threshold,
      na.rm = TRUE
    )

    reward_sum <- sum(ep_obs$reward, na.rm = TRUE)
    reward_rate <- safe_ratio(reward_sum, episode_steps)
    max_reward <- max(ep_obs$reward, na.rm = TRUE)
    success_proxy <- as.integer(!is.na(max_reward) && max_reward >= cfg$success_reward_threshold)

    action_steps <- nrow(ep_merged)
    mean_action_magnitude <- NA_real_
    max_action_magnitude <- NA_real_
    mean_action_delta <- NA_real_
    action_nonzero_fraction <- NA_real_
    progress_per_action <- NA_real_
    wasted_action_fraction <- NA_real_

    if (action_steps > 0) {
      action_mag <- ep_merged$action_magnitude
      action_dx <- c(NA_real_, diff(ep_merged$action_x))
      action_dy <- c(NA_real_, diff(ep_merged$action_y))
      action_delta <- sqrt(action_dx ^ 2 + action_dy ^ 2)
      action_nonzero <- action_mag > cfg$action_epsilon
      goal_distance_delta <- c(NA_real_, -diff(ep_merged$block_goal_distance))
      wasted_action <- action_nonzero & ep_merged$block_speed <= cfg$speed_threshold & coalesce_num(goal_distance_delta, 0) <= 0

      mean_action_magnitude <- mean(action_mag, na.rm = TRUE)
      max_action_magnitude <- max(action_mag, na.rm = TRUE)
      mean_action_delta <- mean(action_delta, na.rm = TRUE)
      action_nonzero_fraction <- mean(action_nonzero, na.rm = TRUE)
      progress_per_action <- safe_ratio(start_goal_distance - final_goal_distance, sum(action_mag, na.rm = TRUE))
      wasted_action_fraction <- mean(wasted_action, na.rm = TRUE)
    }

    rows[[idx]] <- data.frame(
      episode_id = as.integer(episode_id),
      start_goal_distance = start_goal_distance,
      final_goal_distance = final_goal_distance,
      best_goal_distance = best_goal_distance,
      neg_final_goal_distance = -final_goal_distance,
      neg_best_goal_distance = -best_goal_distance,
      final_progress_ratio = final_progress_ratio,
      best_progress_ratio = best_progress_ratio,
      goal_regression = goal_regression,
      neg_goal_regression = -goal_regression,
      near_goal_fraction = near_goal_fraction,
      stay_near_after_best = stay_near_after_best,
      final_20pct_mean_distance = final_20pct_mean_distance,
      neg_final_20pct_mean_distance = -final_20pct_mean_distance,
      episode_steps = episode_steps,
      neg_episode_steps = -episode_steps,
      episode_time = episode_time,
      neg_episode_time = -episode_time,
      progress_rate = progress_rate,
      log_progress_rate = log_progress_rate,
      block_path_length = block_path_length,
      block_movement_directness = block_movement_directness,
      engagement_fraction = engagement_fraction,
      active_push_fraction = active_push_fraction,
      reward_sum = reward_sum,
      reward_rate = reward_rate,
      max_reward = max_reward,
      success_proxy = success_proxy,
      action_steps = action_steps,
      mean_action_magnitude = mean_action_magnitude,
      max_action_magnitude = max_action_magnitude,
      mean_action_delta = mean_action_delta,
      action_nonzero_fraction = action_nonzero_fraction,
      progress_per_action = progress_per_action,
      wasted_action_fraction = wasted_action_fraction
    )
  }

  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0) {
    return(data.frame())
  }
  do.call(rbind, rows)
}

main <- function() {
  cfg <- parse_args(commandArgs(trailingOnly = TRUE))

  actions <- read.csv(cfg$actions)
  observations <- read.csv(cfg$observations)

  require_columns(
    actions,
    c("episode_id", "frame_count", "training_step", "step_index", "action_x", "action_y", "agent_id"),
    "actions CSV"
  )
  require_columns(
    observations,
    c(
      "episode_id", "frame_count", "training_step", "step_index", "agent_id", "is_decision", "reward",
      "realtime_since_start", "agent_pos_x", "agent_pos_z", "block_pos_x", "block_pos_z", "block_vel_x",
      "block_vel_z", "goal_pos_x", "goal_pos_z"
    ),
    "observations CSV"
  )

  actions <- subset(actions, !is.na(episode_id) & episode_id > 0 & agent_id != "init")
  observations <- subset(observations, !is.na(episode_id) & episode_id > 0 & agent_id != "init")

  actions <- prepare_actions(sort_by_episode_step(actions))
  observations <- enrich_observations(sort_by_episode_step(observations))

  merged <- merge_decision_rows(actions, observations)
  if (nrow(merged) == 0) {
    stop("No merged decision rows were produced. Check the join keys and source files.")
  }

  out_dir <- cfg$out_dir
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }

  write.csv(observations, file.path(out_dir, "observations_enriched.csv"), row.names = FALSE)
  write.csv(merged, file.path(out_dir, "action_observation_decision.csv"), row.names = FALSE)

  episode_metrics <- compute_episode_metrics(observations, merged, cfg)
  write.csv(episode_metrics, file.path(out_dir, "episode_construct_metrics.csv"), row.names = FALSE)

  cat(sprintf("Wrote %s\n", normalizePath(out_dir, winslash = "/", mustWork = FALSE)))
  cat(sprintf("observations_enriched rows: %d\n", nrow(observations)))
  cat(sprintf("action_observation_decision rows: %d\n", nrow(merged)))
  cat(sprintf("episode_construct_metrics rows: %d\n", nrow(episode_metrics)))
}

main()