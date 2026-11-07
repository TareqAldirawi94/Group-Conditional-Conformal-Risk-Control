# simulation_sandwich.R  (parallel, chunked)
#
# Section 5.1: two-sided verification of the one-coordinate sandwich for the
# ordinal absolute-error loss (Theorem 2.5).
#   UPPER EDGE (benign):  risk saturates alpha, stays in [alpha-(M+1)B/(n+1), alpha].
#   LOWER EDGE (witness): risk pinned to rank-grid alpha - Delta_{n,M} (sawtooth).
#
# Parallelism: one worker pool created once and reused; each cell's reps are
# split into N_WORKERS chunks, each worker runs a chunk in a tight loop and
# returns a numeric matrix (no per-rep data.frame, no per-rep task dispatch).
#
# Outputs (sim_sandwich_outputs/): fig_sandwich{,_witness}.pdf/png and CSVs.

required_packages <- c("ggplot2", "dplyr", "parallel")
missing <- required_packages[!required_packages %in% rownames(installed.packages())]
if (length(missing) > 0)
  stop("Install required packages first: install.packages(c(",
       paste(sprintf('\"%s\"', missing), collapse = ", "), "))")
invisible(lapply(required_packages, library, character.only = TRUE))

OUTDIR <- "sim_sandwich_outputs"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
K           <- 5L
B           <- K - 1L           # = 4
ALPHA       <- 0.4
NK_VALS     <- c(100L, 200L, 500L, 1000L, 2000L)   # benign (upper-edge) grid
M_VALS      <- c(1L, 2L, 4L)
N_REPS      <- 50000L           # benign reps; reduce (e.g. 5000) for a quick pass
NK_FINE     <- seq(10L, 120L, by = 2L)             # witness (lower-edge) fine grid
WITNESS_REPS <- 20000L          # witness reps; reduce (e.g. 5000) for a quick pass
GLOBAL_SEED <- 4100L
# Pin the worker count for reproducible wall-clock behavior; override with the
# environment variable SANDWICH_WORKERS. Defaults to 4 (not all machine cores).
N_WORKERS   <- {w <- suppressWarnings(as.integer(Sys.getenv("SANDWICH_WORKERS", "4")))
if (is.na(w) || w < 1L) 4L else w}

# -------------------------------------------------------------
# Math kernels (unchanged)
# -------------------------------------------------------------
# Per-path ordinal loss-drop events (handles ties): distinct thresholds mu where
# L_i(mu) = min_{j: s_ij <= mu} |j - y| strictly drops, and the drop sizes delta.
ord_events <- function(s_row, y, B, K) {
  d  <- abs(seq_len(K) - y)
  o  <- order(s_row)
  ss <- s_row[o]; dd <- d[o]
  mu <- numeric(0); delta <- numeric(0)
  prev <- B; r <- 1L
  while (r <= K) {
    v  <- ss[r]; rr <- r
    while (rr < K && ss[rr + 1L] == v) rr <- rr + 1L   # group ties at value v
    cur <- min(dd[1:rr])
    if (cur < prev - 1e-12) { mu <- c(mu, v); delta <- c(delta, prev - cur); prev <- cur }
    r <- rr + 1L
  }
  list(mu = mu, delta = delta)
}

ord_loss_at <- function(s_row, y, mu, B, K) {
  inc <- which(s_row <= mu)
  if (!length(inc)) B else min(abs(inc - y))
}

# CRC threshold from aggregated calibration events. base_total = starting summed
# loss across the n paths; default n*B reproduces benign behavior, witness passes
# active-only mass so inactive zero-paths aren't assumed to start at B.
crc_threshold_events <- function(mu, delta, n, alpha, B, base_total = n * B) {
  if (!length(mu)) {
    adj0 <- (base_total + B) / (n + 1)
    return(if (adj0 <= alpha + 1e-12) -Inf else Inf)
  }
  rs  <- rowsum(delta, group = mu, reorder = TRUE)   # C-level group sum (fast)
  umu <- as.numeric(rownames(rs)); ad <- as.numeric(rs[, 1])
  total <- base_total - cumsum(ad)
  adj   <- (n / (n + 1)) * (total / n) + B / (n + 1)
  idx   <- which(adj <= alpha + 1e-12)
  if (!length(idx)) return(Inf)
  umu[min(idx)]
}

# -------------------------------------------------------------
# Single-replication cores -> c(test_loss, Mk)  (no data.frame)
# -------------------------------------------------------------
# Vectorized: no per-path lapply, no factor building. Per (path,label) value
# v_ij = B - |j - Y_i| >= 0; path running drop D_i(mu) = max_{j: s_ij<=mu} v_ij,
# computed as a group-wise cummax via an offset trick. Verified identical to the
# per-path reference (test_loss, Mk, lambda) over thousands of random cases.
one_rep_core <- function(n_k, M) {
  N       <- n_k + 1L
  grid_sz <- ceiling(N * K / M)
  pool    <- rep(seq_len(grid_sz), each = M)
  scores  <- sample(pool, N * K, replace = FALSE)
  S       <- matrix(scores, nrow = N, ncol = K)
  Y       <- sample.int(K, N, replace = TRUE)
  
  Vmat <- B - abs(matrix(seq_len(K), N, K, byrow = TRUE) - Y)   # N x K, in [0, B]
  path <- rep(seq_len(N), times = K)
  loc  <- as.vector(S)
  val  <- as.vector(Vmat)
  ord  <- order(path, loc)                          # sort by (path, location)
  path <- path[ord]; loc <- loc[ord]; val <- val[ord]
  off  <- B + 1                                      # > max val, separates groups
  rmax <- cummax(val + off * path) - off * path      # within-path running max of v
  start <- c(TRUE, path[-1] != path[-length(path)])
  prevm <- c(0, rmax[-length(rmax)]); prevm[start] <- 0
  delta <- rmax - prevm                              # >= 0; positive at jump locations
  keep  <- delta > 1e-12
  path  <- path[keep]; loc <- loc[keep]; delta <- delta[keep]
  
  cal    <- path <= n_k                              # calibration paths
  lambda <- crc_threshold_events(loc[cal], delta[cal], n_k, ALPHA, B)
  test_loss <- ord_loss_at(S[N, ], Y[N], lambda, B, K)
  
  # realized M_k = max distinct paths jumping at a common location
  if (length(loc)) {
    upl <- !duplicated(cbind(path, loc))             # collapse within-path same-loc events
    Mk  <- max(tabulate(loc[upl]))
  } else Mk <- 1L
  c(test_loss, Mk)
}

one_rep_witness_core <- function(n_k, M) {
  N    <- n_k + 1L
  Kgrp <- N %/% M
  r    <- N - Kgrp * M
  grp  <- c(rep(seq_len(Kgrp), each = M), rep(0L, r))[sample.int(N)]   # 0 = inactive
  ac   <- grp[seq_len(n_k)]; ac <- ac[ac > 0L]
  mu_cal    <- as.numeric(ac)
  delta_cal <- rep(B, length(mu_cal))
  lambda    <- crc_threshold_events(mu_cal, delta_cal, n_k, ALPHA, B,
                                    base_total = sum(delta_cal))
  g_test    <- grp[N]
  test_loss <- if (g_test == 0L) 0 else if (lambda < g_test) B else 0
  c(test_loss, M)                                                      # Mk = M exactly
}

# -------------------------------------------------------------
# Worker pool: created once, reused for every cell of both sweeps
# -------------------------------------------------------------
cl <- NULL
if (N_WORKERS > 1L) {
  cl <- parallel::makeCluster(N_WORKERS)            # PSOCK on Windows
  parallel::clusterExport(cl, c(
    "ord_events", "ord_loss_at", "crc_threshold_events",
    "one_rep_core", "one_rep_witness_core", "K", "B", "ALPHA"))
}
on.exit(if (!is.null(cl)) parallel::stopCluster(cl), add = TRUE)

# Chunked evaluator: split `reps` across workers, each runs a tight batch loop.
run_cell <- function(core_fun, reps, n_k, M, seed) {
  force(core_fun); force(n_k); force(M)            # defensive: forbid stray promises
  W      <- if (is.null(cl)) 1L else length(cl)
  base   <- reps %/% W; rem <- reps %% W
  chunks <- base + c(rep(1L, rem), rep(0L, max(0L, W - rem)))
  chunks <- chunks[chunks > 0L]
  batch  <- function(nrep, core_fun, n_k, M) {       # take everything as args
    m <- matrix(0, nrep, 2L)
    for (b in seq_len(nrep)) m[b, ] <- core_fun(n_k, M)
    m
  }
  if (is.null(cl)) {
    set.seed(seed)
    res <- lapply(chunks, batch, core_fun, n_k, M)
  } else {
    parallel::clusterSetRNGStream(cl, iseed = seed)  # independent, reproducible streams
    res <- parallel::parLapply(cl, chunks, batch, core_fun, n_k, M)  # values, not promises
  }
  out <- do.call(rbind, res)
  data.frame(n_k = n_k, M = M, test_loss = out[, 1], Mk = out[, 2])
}

# =============================================================
# UPPER EDGE: benign saturation
# =============================================================
cat(sprintf("Workers=%d. Upper edge: reps=%d, n_k in {%s}, M in {%s}\n",
            N_WORKERS, N_REPS, paste(NK_VALS, collapse = ","),
            paste(M_VALS, collapse = ",")))
cells <- expand.grid(n_k = NK_VALS, M = M_VALS)
t0 <- Sys.time(); records <- vector("list", nrow(cells))
for (i in seq_len(nrow(cells))) {
  n_k <- cells$n_k[i]; M <- cells$M[i]
  records[[i]] <- run_cell(one_rep_core, N_REPS, n_k, M,
                           GLOBAL_SEED + 1000L * M + n_k)
  cat(sprintf("  n_k=%4d M=%d done (%.0fs)\n", n_k, M,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
df_all <- do.call(rbind, records)
write.csv(df_all, file.path(OUTDIR, "sandwich_raw.csv"), row.names = FALSE)

df_summary <- df_all %>%
  dplyr::group_by(n_k, M) %>%
  dplyr::summarise(risk = mean(test_loss), se = sd(test_loss) / sqrt(dplyr::n()),
                   Mk_max = max(Mk), Mk_mean = mean(Mk), .groups = "drop") %>%
  dplyr::mutate(band_lo = pmax(0, ALPHA - (M + 1) * B / (n_k + 1)),
                band_hi = ALPHA,
                M_label = factor(sprintf("M = %d", M),
                                 levels = sprintf("M = %d", M_VALS)))
write.csv(df_summary, file.path(OUTDIR, "sandwich_summary.csv"), row.names = FALSE)

cat("\n--- Realized jump count per cell (should match the budget M) ---\n")
print(df_summary %>% dplyr::group_by(M) %>%
        dplyr::summarise(Mk_max = max(Mk_max), .groups = "drop"))

p <- ggplot(df_summary, aes(x = n_k, y = risk)) +
  geom_ribbon(aes(ymin = band_lo, ymax = band_hi), fill = "steelblue", alpha = 0.15) +
  geom_hline(yintercept = ALPHA, linetype = "dashed", colour = "grey40") +
  geom_line(colour = "steelblue", linewidth = 0.6) +
  geom_errorbar(aes(ymin = risk - 2 * se, ymax = risk + 2 * se),
                width = 0.04, colour = "steelblue") +
  geom_point(size = 2, colour = "steelblue") +
  facet_wrap(~ M_label, nrow = 1L) +
  scale_x_log10(breaks = NK_VALS) +
  labs(x = expression(Calibration~count~n[k]), y = "Empirical within-group risk",
       caption = sprintf("Shaded band: [max(0, alpha-(M+1)B/(n_k+1)), alpha]. Dashed: alpha=%.2f. +/-2 MC SE over %d reps.",
                         ALPHA, N_REPS)) +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(), plot.caption = element_text(hjust = 0, size = 8),
        strip.background = element_rect(fill = "grey95"), strip.text = element_text(face = "bold"))
ggsave(file.path(OUTDIR, "fig_sandwich.pdf"), p, width = 10, height = 3.6)
ggsave(file.path(OUTDIR, "fig_sandwich.png"), p, width = 10, height = 3.6, dpi = 250)

# =============================================================
# LOWER EDGE: rank-grid witness
# =============================================================
cat(sprintf("\nLower edge (witness): reps=%d, n_k in {%d,...,%d}, M in {%s}\n",
            WITNESS_REPS, min(NK_FINE), max(NK_FINE), paste(M_VALS, collapse = ",")))
cells_w <- expand.grid(n_k = NK_FINE, M = M_VALS)
t0 <- Sys.time(); records_w <- vector("list", nrow(cells_w))
for (i in seq_len(nrow(cells_w))) {
  n_k <- cells_w$n_k[i]; M <- cells_w$M[i]
  records_w[[i]] <- run_cell(one_rep_witness_core, WITNESS_REPS, n_k, M,
                             GLOBAL_SEED + 700000L + 1000L * M + n_k)
  if (i %% 20 == 0 || i == nrow(cells_w))
    cat(sprintf("  [witness] %d/%d cells (%.0fs)\n", i, nrow(cells_w),
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
df_w <- do.call(rbind, records_w)

df_w_sum <- df_w %>%
  dplyr::group_by(n_k, M) %>%
  dplyr::summarise(risk = mean(test_loss), se = sd(test_loss) / sqrt(dplyr::n()),
                   .groups = "drop") %>%
  dplyr::mutate(N = n_k + 1,
                predicted = (B * M / N) * floor(N * ALPHA / (B * M)),  # alpha - Delta
                band_lo = pmax(0, ALPHA - (M + 1) * B / N),
                M_label = factor(sprintf("M = %d", M),
                                 levels = sprintf("M = %d", M_VALS)))
write.csv(df_w_sum, file.path(OUTDIR, "sandwich_witness_summary.csv"), row.names = FALSE)

cat(sprintf("\nWitness vs rank-grid: max |empirical - predicted| = %.4f  (mean MC SE %.4f)\n",
            max(abs(df_w_sum$risk - df_w_sum$predicted)), mean(df_w_sum$se)))
# Anchors must lie on the even support grid seq(10,120,by=2). Note that the exact
# alpha cells (predicted == 0.400) occur only at N a multiple of 10, i.e. n_k in
# {9,19,29,...}, which are odd and hence NOT on this grid; the even grid therefore
# samples the sawtooth near, but never exactly at, its peaks.
cat("  Anchors (M=1): n_k=20 -> 0.381, n_k=28 -> 0.276, n_k=40 -> 0.390\n")
print(df_w_sum %>% dplyr::filter(M == 1, n_k %in% c(20L, 28L, 40L)) %>%
        dplyr::select(n_k, risk, predicted, se))

p_w <- ggplot(df_w_sum, aes(x = n_k)) +
  geom_ribbon(aes(ymin = band_lo, ymax = ALPHA), fill = "grey70", alpha = 0.25) +
  geom_hline(yintercept = ALPHA, linetype = "dashed", colour = "grey40") +
  geom_line(aes(y = predicted), colour = "firebrick", linewidth = 0.5) +
  geom_point(aes(y = risk), size = 1.0, colour = "black") +
  facet_wrap(~ M_label, nrow = 1L) +
  labs(x = expression(Calibration~count~n[k]),
       y = "Empirical within-group risk (witness)",
       caption = sprintf("Grey band: [max(0, alpha-(M+1)B/(n+1)), alpha]. Red: rank-grid alpha-Delta. Dashed: alpha=%.2f. %d reps/cell.",
                         ALPHA, WITNESS_REPS)) +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(), plot.caption = element_text(hjust = 0, size = 8),
        strip.background = element_rect(fill = "grey95"), strip.text = element_text(face = "bold"))
ggsave(file.path(OUTDIR, "fig_sandwich_witness.pdf"), p_w, width = 10, height = 3.6)
ggsave(file.path(OUTDIR, "fig_sandwich_witness.png"), p_w, width = 10, height = 3.6, dpi = 250)

if (!is.null(cl)) { parallel::stopCluster(cl); cl <- NULL }
cat("\nDone. Wrote figures and CSVs to sim_sandwich_outputs/\n")