# fig_s4_barrier.R
# ---------------------------------------------------------------------------
# Section 4 motivating figure: full-resolution multilabel conditioning collapses
# under the Section 3 support barrier.
#
#   Panel A: refine the label-vector identifier (condition on the k most prevalent
#            labels, k = 1..L). As resolution grows, the fraction of cells that MEET
#            THE NECESSARY SUPPORT COUNT, and the fraction of (in-sample) observation
#            mass in such cells, both collapse toward zero.
#   Panel B: at full resolution (all L labels), the per-pattern calibration count
#            (rank-frequency) lies mostly below the barrier line, so most patterns
#            lack the support needed to admit a nonmaximal cutoff.
#
# Correlated multilabel data via a latent Gaussian copula (skewed prevalences,
# moderate positive co-occurrence). Miscoverage-type loss: B=1, alpha=0.10. The
# count n_a >= ceil(B/alpha) - 1 = 9 is NECESSARY for a nonmaximal cutoff, not a
# guarantee that the deployed set is nontrivial; counts and mass here are computed
# in-sample on the calibration draw, not on an independent test set.
#
# Output: fig_s4_barrier.{pdf,png}
# ---------------------------------------------------------------------------

OUTDIR <- "."                       # edit if you want the figure elsewhere
L      <- 20L
N      <- 30000L
B      <- 1.0
ALPHA  <- 0.10
BARRIER <- as.integer(ceiling(B / ALPHA)) - 1L    # n_a >= 9 to be nontrivial
RHO    <- 0.35
SEED   <- 20L
set.seed(SEED)

# skewed per-label prevalences: a few common findings, many rare ones (descending)
prev <- sort(runif(L, 0.02, 0.35), decreasing = TRUE)

# latent Gaussian copula -> correlated Bernoulli labels
Sigma <- (1 - RHO) * diag(L) + RHO * matrix(1, L, L)
Lc    <- chol(Sigma)                              # upper-triangular, Z = G %*% Lc
Z     <- matrix(rnorm(N * L), N, L) %*% Lc
thr   <- qnorm(1 - prev)                          # P(Z_j > thr_j) = prev_j
Y     <- (sweep(Z, 2, thr, ">")) * 1L             # N x L binary (0/1) matrix
storage.mode(Y) <- "integer"

# map rows of a 0/1 matrix to an integer pattern code (columns = binary digits)
pattern_codes <- function(M) {
  k <- ncol(M)
  w <- 2^((k - 1):0)                              # exact in double up to 2^20
  as.vector(M %*% w)
}

# ---- Panel A: resolution sweep -------------------------------------------------
ks            <- 1:L
frac_cells_ok <- numeric(L)
frac_mass_ok  <- numeric(L)
n_cells       <- numeric(L)
for (i in ks) {
  codes <- pattern_codes(Y[, 1:i, drop = FALSE])
  cnt   <- as.integer(table(codes))
  n_cells[i]       <- length(cnt)
  ok               <- cnt >= BARRIER
  frac_cells_ok[i] <- mean(ok)
  frac_mass_ok[i]  <- sum(cnt[ok]) / N
}

# ---- Panel B: full-resolution rank-frequency ----------------------------------
cnt_full   <- as.integer(table(pattern_codes(Y)))
cnt_sorted <- sort(cnt_full, decreasing = TRUE)
n_patterns <- length(cnt_sorted)
n_ok       <- sum(cnt_sorted >= BARRIER)
mass_ok    <- sum(cnt_sorted[cnt_sorted >= BARRIER]) / N
pigeonhole <- N %/% BARRIER

cat(sprintf("distinct patterns at full resolution : %d\n", n_patterns))
cat(sprintf("patterns meeting support count (n_a>=%d): %d (%.2f%% of observed)\n",
            BARRIER, n_ok, 100 * n_ok / n_patterns))
cat(sprintf("pigeonhole cap  floor(N/%d)            : %d\n", BARRIER, pigeonhole))
cat(sprintf("in-sample mass in supported cells     : %.1f%%\n", 100 * mass_ok))

# ---- plotting -----------------------------------------------------------------
required <- c("ggplot2")
miss <- required[!required %in% rownames(installed.packages())]
if (length(miss))
  stop("Install required packages first: install.packages(c(",
       paste(sprintf('"%s"', miss), collapse = ", "), "))")
suppressMessages(library(ggplot2))

C_CELL <- "steelblue"
C_MASS <- "firebrick"
C_BAR  <- "grey30"

# Panel A data (long form)
dfA <- rbind(
  data.frame(k = ks, pct = 100 * frac_cells_ok,
             series = "cells meeting support count"),
  data.frame(k = ks, pct = 100 * frac_mass_ok,
             series = "in-sample mass in supported cells")
)
pA <- ggplot(dfA, aes(k, pct, color = series, shape = series)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2.1) +
  scale_color_manual(values = c("cells meeting support count" = C_CELL,
                                "in-sample mass in supported cells" = C_MASS)) +
  scale_shape_manual(values = c("cells meeting support count" = 16,
                                "in-sample mass in supported cells" = 15)) +
  scale_x_continuous(breaks = c(1, 5, 10, 15, 20)) +
  coord_cartesian(ylim = c(0, 102)) +
  labs(x = expression("resolution: number of labels conditioned on " * k),
       y = "percent", color = NULL, shape = NULL,
       title = "(A) refining the identifier collapses support") +
  theme_bw(base_size = 12) +
  theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1),
        legend.background = element_blank(), legend.key = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 11))

# Panel B data
dfB <- data.frame(rank = seq_len(n_patterns), count = cnt_sorted)
dfB_triv <- subset(dfB, count < BARRIER)
pB <- ggplot(dfB, aes(rank, count)) +
  geom_area(data = dfB_triv, aes(rank, count), fill = C_BAR, alpha = 0.12) +
  geom_line(color = C_CELL, linewidth = 0.7) +
  geom_hline(yintercept = BARRIER, linetype = "dashed", color = C_BAR, linewidth = 0.6) +
  # barrier label: left end, just above the dashed line, clean math
  annotate("text", x = 1.2, y = BARRIER * 1.45, hjust = 0, vjust = 0, size = 3.1,
           color = C_BAR, parse = TRUE,
           label = sprintf("barrier:~n[a]==%d", BARRIER)) +
  # summary: open top-right corner, right-justified, no arrow
  annotate("text", x = max(dfB$rank), y = max(dfB$count) * 0.75, hjust = 1, vjust = 1,
           size = 3.1, color = C_MASS, lineheight = 0.95,
           label = sprintf("%d of %d patterns meet the support count\n(%.0f%% of in-sample mass)",
                           n_ok, n_patterns, 100 * mass_ok)) +
  annotate("text", x = max(1, round(n_patterns * 0.6)), y = 1.7, size = 3.1,
           color = C_BAR, fontface = "italic", label = "below support count") +
  scale_x_log10() + scale_y_log10() +
  labs(x = "pattern rank (most to least frequent)",
       y = expression("calibration count " * n[a]),
       title = "(B) full resolution: counts sit below the barrier") +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(size = 11))

# combine
if (requireNamespace("gridExtra", quietly = TRUE)) {
  g <- gridExtra::arrangeGrob(pA, pB, nrow = 1)
  ggsave(file.path(OUTDIR, "fig_s4_barrier.pdf"), g, width = 10.4, height = 4.3)
  ggsave(file.path(OUTDIR, "fig_s4_barrier.png"), g, width = 10.4, height = 4.3, dpi = 220)
} else {
  ggsave(file.path(OUTDIR, "fig_s4_barrier_A.pdf"), pA, width = 5.3, height = 4.3)
  ggsave(file.path(OUTDIR, "fig_s4_barrier_B.pdf"), pB, width = 5.3, height = 4.3)
  cat("(install gridExtra for the combined figure; wrote panels separately)\n")
}
cat("done.\n")
