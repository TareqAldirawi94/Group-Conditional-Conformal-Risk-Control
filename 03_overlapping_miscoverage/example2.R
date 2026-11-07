## Section 5.2, E4a: Overlapping-group miscoverage and local support
## Audited revision. Base R; no external packages.
## Main change: independent held-out evaluation draw per replication so that
## across-replication Monte Carlo standard errors include test-sample variation.
## The calibration algorithms and score are unchanged from the supplied code.
## Outputs: figures/fig_overlap_miscoverage.pdf and .png;
##          figures/example2_summary.csv and example2_atoms.csv.

set.seed(20261025)
L <- 3L; B <- 1; ALPHA <- 0.10; NV <- 2L^L
VEC <- t(sapply(0:(NV-1L), function(a)
  bitwAnd(bitwShiftR(a, 0:(L-1L)), 1L)))
PAT <- apply(VEC, 1L, function(v) paste(v, collapse=""))
P <- c(0.55, 0.40, 0.18)
GAM <- c(1.70, 1.00, 0.40)
NOISE <- 1.0; TAU <- qnorm(1-P)
BARRIER <- B / ALPHA
NS <- c(100L, 400L, 1600L, 6400L)
REPS <- 500L
NEVAL <- 6000L  # independent test observations per replicate
LMAX <- 1e9    # finite maximal action; scores bounded by probability clipping
POW2 <- 2^(0:(L-1L))
METHODS <- c(marg=1L, atom=2L, union=3L)

# X can be taken as eta (or q); Y is generated from latent Gaussian z.
# q is a fixed predictive score component, NOT assumed to be the true
# conditional probability when labels are dependent.
gen <- function(n, rho) {
  f <- matrix(rnorm(n), n, 1L)
  e <- matrix(rnorm(n*L), n, L)
  z <- sqrt(rho)*f[, rep(1L, L)] + sqrt(1-rho)*e
  Y <- 0L + (z > matrix(TAU, n, L, byrow=TRUE))
  eta <- matrix(GAM, n, L, byrow=TRUE) *
    (z - matrix(TAU, n, L, byrow=TRUE)) +
    NOISE*matrix(rnorm(n*L), n, L)
  q <- pmin(pmax(plogis(eta), 1e-6), 1-1e-6)
  base <- rowSums(-log1p(-q))
  delta <- log1p(-q) - log(q)
  sc <- base + delta %*% t(VEC)
  yid <- as.integer(round(Y %*% POW2))
  list(Y=Y, yid=yid, sc=sc)
}

# For continuous true-label scores this is exactly the CRC cutoff:
# the floor((n+1)*alpha)-th largest score. If count is insufficient,
# choose the maximal threshold, which admits all eligible candidates.
crc_cut <- function(strue, alpha=ALPHA) {
  n <- length(strue)
  k <- floor((n+1)*alpha)
  if (n==0L || k<1L) return(list(t=LMAX, def=TRUE))
  list(t=sort(strue, decreasing=TRUE)[k], def=FALSE)
}

mc_hw <- function(x) 1.96 * sd(x) / sqrt(length(x))

run <- function(rho) {
  # Group index 1 is G0={Y=000}; indices 2,3,4 are focal G1,G2,G3.
  Rg <- array(NA_real_, c(REPS, L+1L, length(METHODS), length(NS)))
  Rp <- array(NA_real_, c(REPS, length(METHODS), length(NS)))
  Sz <- array(NA_real_, c(REPS, length(METHODS), length(NS)))
  # Test-pattern-probability-weighted fallback probability.
  Fb <- array(NA_real_, c(REPS, 2L, length(NS)))
  # Fraction of the 8 atom groups / 4 original groups selecting max.
  Fg <- array(NA_real_, c(REPS, 2L, length(NS)))
  atoms <- matrix(0, length(NS), NV)
  adefr <- matrix(0, length(NS), NV)
  tmr <- c(marg=0, atom=0, union=0)

  for (ni in seq_along(NS)) {
    n <- NS[ni]
    for (r in seq_len(REPS)) {
      dc <- gen(n, rho)
      Yc <- dc$Y; yidc <- dc$yid
      strue_c <- dc$sc[cbind(seq_len(n), yidc+1L)]

      start <- proc.time()[3]
      tm <- crc_cut(strue_c)$t
      tmr["marg"] <- tmr["marg"] + proc.time()[3]-start
      cut_marg <- rep(tm, NV)

      start <- proc.time()[3]
      ta <- numeric(NV); adef <- logical(NV)
      for (a in 0:(NV-1L)) {
        cc <- crc_cut(strue_c[yidc==a])
        ta[a+1L] <- cc$t; adef[a+1L] <- cc$def
      }
      tmr["atom"] <- tmr["atom"] + proc.time()[3]-start
      cut_atom <- ta
      atoms[ni, ] <- atoms[ni, ] + tabulate(yidc+1L, nbins=NV)
      adefr[ni, ] <- adefr[ni, ] + adef

      start <- proc.time()[3]
      foc <- lapply(seq_len(L), function(j) crc_cut(strue_c[Yc[,j]==1]))
      tf <- vapply(foc, function(z) z$t, numeric(1))
      cb <- crc_cut(strue_c[yidc==0L]); t0b <- cb$t
      cut_un <- vapply(0:(NV-1L), function(a) {
        on <- which(VEC[a+1L, ]==1)
        if (length(on)>0L) max(tf[on]) else t0b
      }, numeric(1))
      tmr["union"] <- tmr["union"] + proc.time()[3]-start

      Fg[r,1,ni] <- mean(adef)
      Fg[r,2,ni] <- mean(c(cb$def, vapply(foc, function(z) z$def, logical(1))))

      # A fresh test sample for every replication. Common test points within
      # the replication give paired comparisons across the three methods.
      ep <- gen(NEVAL, rho)
      Yp <- ep$Y; yidp <- ep$yid; scp <- ep$sc
      gm <- c(list(yidp==0L), lapply(seq_len(L), function(j) Yp[,j]==1L))
      Fb[r,1,ni] <- mean(adef[yidp+1L])
      Fb[r,2,ni] <- mean(cut_un[yidp+1L] >= LMAX)

      for (nm in names(METHODS)) {
        mi <- METHODS[nm]
        cut <- switch(nm, marg=cut_marg, atom=cut_atom, union=cut_un)
        adm <- sweep(scp, 2L, cut, FUN="<=")
        covered <- adm[cbind(seq_len(NEVAL), yidp+1L)]
        for (g in seq_len(L+1L)) Rg[r,g,mi,ni] <- mean(!covered[gm[[g]]])
        Rp[r,mi,ni] <- mean(!covered)
        Sz[r,mi,ni] <- mean(rowSums(adm))
      }
    }
    atoms[ni, ] <- atoms[ni, ]/REPS
    adefr[ni, ] <- adefr[ni, ]/REPS
  }
  list(Rg=Rg, Rp=Rp, Sz=Sz, Fb=Fb, Fg=Fg, atoms=atoms,
       adefr=adefr, cal_ms=1000*tmr/(REPS*length(NS)), midx=METHODS)
}

res_ind <- run(0.0)
res_cor <- run(0.5)

mk_summary <- function(res, cond) {
  do.call(rbind, lapply(names(res$midx), function(nm) {
    mi <- res$midx[nm]
    do.call(rbind, lapply(seq_along(NS), function(ni) {
      is_marg <- nm=="marg"
      defidx <- if (nm=="atom") 1L else 2L
      data.frame(
        condition=cond,
        method=c(marg="Marginal CRC",atom="Atom-wise Mondrian",union="Union (ours)")[nm],
        n=NS[ni],
        pooled_risk=mean(res$Rp[,mi,ni]),
        risk_G0=mean(res$Rg[,1,mi,ni]),
        risk_G1=mean(res$Rg[,2,mi,ni]),
        risk_G2=mean(res$Rg[,3,mi,ni]),
        risk_G3=mean(res$Rg[,4,mi,ni]),
        mean_size=mean(res$Sz[,mi,ni]),
        weighted_default_rate=if(is_marg) NA_real_ else mean(res$Fb[,defidx,ni]),
        fraction_groups_default=if(is_marg) NA_real_ else mean(res$Fg[,defidx,ni]),
        cal_ms_per_rep=unname(res$cal_ms[nm]),
        risk_G3_hw=mc_hw(res$Rg[,4,mi,ni]),
        mean_size_hw=mc_hw(res$Sz[,mi,ni]),
        row.names=NULL
      )
    }))
  }))
}

summary <- rbind(mk_summary(res_ind, "independent"),
                 mk_summary(res_cor, "correlated"))
dir.create("figures", recursive=TRUE, showWarnings=FALSE)
write.csv(summary, "figures/example2_summary.csv", row.names=FALSE)
atom_tab <- data.frame(atom=PAT,
                       avg_count_n100=res_ind$atoms[1,],
                       default_rate_n100=res_ind$adefr[1,])
write.csv(atom_tab, "figures/example2_atoms.csv", row.names=FALSE)

cat(sprintf("Example 2: L=%d, alpha=%.2f, B/alpha=%.0f, %d reps, %d fresh test samples per rep\n",
            L,ALPHA,BARRIER,REPS,NEVAL))
print(summary[,c("condition","method","n","risk_G0","risk_G1","risk_G2","risk_G3",
                "mean_size","weighted_default_rate","fraction_groups_default")],
      row.names=FALSE, digits=3)
cat("\nAtom counts and fallback rates at n=100 (independent labels):\n")
print(atom_tab, row.names=FALSE, digits=3)

make_plot <- function(file, format=c("pdf", "png")) {
  format <- match.arg(format)
  if(format=="pdf") pdf(file, width=14, height=4.1, useDingbats=FALSE)
  else png(file, width=1750, height=512, res=125)
  old <- par(mfrow=c(1,3), mar=c(4.3,4.3,2.2,1))
  on.exit({par(old); dev.off()}, add=TRUE)
  R <- res_ind; mi <- R$midx
  cols <- c(marg="grey55", atom="forestgreen", union="firebrick")
  labs <- c(marg="Marginal CRC",atom="Atom-wise Mondrian",union="Union (ours)")

  w <- .26; gx <- seq_len(L)
  plot(NA, xlim=c(.5,L+.5), ylim=c(0,.14), xaxt="n", xlab="",
       ylab="Conditional miscoverage", main="Per-group miscoverage, n = 100")
  abline(h=ALPHA, lty=2)
  for(i in seq_along(mi)) {
    nm <- names(mi)[i]; x <- gx+(i-2)*w
    mu <- vapply(1:L, function(g) mean(R$Rg[,g+1,mi[nm],1]), numeric(1))
    hw <- vapply(1:L, function(g) mc_hw(R$Rg[,g+1,mi[nm],1]), numeric(1))
    rect(x-w/2,0,x+w/2,mu,col=cols[nm],border=NA)
    segments(x,pmax(0,mu-hw),x,mu+hw)
  }
  axis(1,gx,c("G1","G2","G3 (hard, rare)"))
  legend("topright",labs,fill=cols,bty="n",cex=.7)

  plot(NA,xlim=range(NS),ylim=c(3.5,6.6),log="x",xaxt="n",
       xlab="Calibration size n",ylab="Mean set size",
       main="Prediction-set size")
  axis(1,NS,NS)
  for(nm in names(mi))
    lines(NS,vapply(seq_along(NS),function(ni) mean(R$Sz[,mi[nm],ni]),numeric(1)),
          type="o",pch=19,col=cols[nm])
  legend("topright",labs,lty=1,pch=19,col=cols,bty="n",cex=.7)

  plot(NA,xlim=range(NS),ylim=c(0,.20),log="x",xaxt="n",
       xlab="Calibration size n",ylab="Test-weighted fallback rate",
       main="Local-support cost: atoms vs focal groups")
  axis(1,NS,NS)
  for(k in 1:2) {
    nm <- c("atom","union")[k]
    lines(NS,vapply(seq_along(NS),function(ni) mean(R$Fb[,k,ni]),numeric(1)),
          type="o",pch=c(15,17)[k],col=cols[nm])
  }
  legend("topright",c("Atom-wise","Union (ours)"),pch=c(15,17),
         col=cols[c("atom","union")],bty="n",cex=.7)
  invisible(file)
}
make_plot("figures/fig_overlap_miscoverage.pdf","pdf")
make_plot("figures/fig_overlap_miscoverage.png","png")
cat("\nSaved figures/fig_overlap_miscoverage.{pdf,png} and CSV files.\n")
