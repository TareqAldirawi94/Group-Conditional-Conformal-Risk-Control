## Section 5.1 -- Group-conditional ordinal risk control (base R)
## Three feature-defined groups; independent, identically distributed observations.
## Y in {0,...,4}, Yhat=Y+sigma_g U, U~Unif(-sqrt(3),sqrt(3)).
## Prediction sets: C_t={j:abs(j-Yhat)<=t}.
## Loss: min_{j in C_t} abs(j-Y), with loss(empty,Y)=B=4.
## Methods: pooled CRC, coordinatewise Mondrian CRC, and groupwise
## Hoeffding--Bentkus (HB) RCPS at PER-GROUP delta=.10.
## Emits figures/fig_general.{pdf,png} and two CSVs.
##
## Merged version: base engineering (PDF/CSV, error bars, defensive checks,
## theoretical-M sandwich floor) PLUS per-method calibration timing and an
## EMPIRICAL M_g verification printed next to the theoretical M=1 used in the
## bound. The Python twin example1.py reproduces every number and artifact.
## M_g=1 almost surely: iid observations with atomless candidate-score
## distributions => distinct jumps across loss paths (checked empirically below).

set.seed(20261024)
K <- 5L; B <- 4; ALPHA <- 0.5
SIG <- c(0.7, 1.3, 2.0); NG <- length(SIG)
NCAL <- 1500L; NTEST <- 3000L; REPS <- 500L
DELTA <- 0.10  # A per-group, not a simultaneous, RCPS confidence level
GR <- 0:(K-1L); sqrt3 <- sqrt(3)
THRESHOLD_MIN <- 0; THRESHOLD_MAX <- 12
M_THEORY <- 1L

stopifnot(THRESHOLD_MAX > (K-1L) + max(SIG)*sqrt3)

draw <- function(n) {
  g <- sample.int(NG, n, replace=TRUE)
  Y <- sample.int(K, n, replace=TRUE)-1L
  U <- runif(n, -sqrt3, sqrt3)
  list(g=g, Y=Y, Yhat=Y+SIG[g]*U)
}

loss_at <- function(t, Yhat, Y) {
  if (length(Y)==0L) return(numeric(0))
  losses <- rep(Inf, length(Y))
  for (k in seq_len(K)) {
    included <- abs(GR[k]-Yhat) <= t
    losses <- pmin(losses, ifelse(included, abs(GR[k]-Y), Inf))
  }
  losses[!is.finite(losses)] <- B
  losses
}

size_at <- function(t, Yhat) {
  if (length(Yhat)==0L) return(integer(0))
  sizes <- integer(length(Yhat))
  for (k in seq_len(K)) sizes <- sizes+(abs(GR[k]-Yhat) <= t)
  sizes
}

## Empirical jump multiplicity: largest number of within-group augmented loss
## paths that break at a common threshold (expect 1 under atomless scores).
jump_mult <- function(Yhat) {
  bp <- round(as.vector(abs(outer(Yhat, GR, function(a,b) b-a))), 9)
  if (length(bp)==0L) return(0L)
  max(tabulate(match(bp, unique(bp))))
}

## Find smallest feasible continuous threshold by bisection.
crc_threshold <- function(Yhat, Y, alpha=ALPHA,
                          lo=THRESHOLD_MIN, hi=THRESHOLD_MAX) {
  n <- length(Y)
  if (n==0L) return(hi)
  criterion <- function(t) (sum(loss_at(t,Yhat,Y))+B)/(n+1)
  if (criterion(lo) <= alpha) return(lo)
  if (criterion(hi) > alpha) return(hi)
  for (i in 1:50) {
    mid <- (lo+hi)/2
    if (criterion(mid) <= alpha) hi <- mid else lo <- mid
  }
  hi
}

## One-sided HB lower-tail p-value at hypothesized population risk R.
hb_pvalue <- function(R, muhat, n) {
  if (muhat >= R) return(1)
  if (muhat <= 0) kl <- -log1p(-R)
  else kl <- muhat*log(muhat/R)+(1-muhat)*log((1-muhat)/(1-R))
  p_b <- exp(1)*pbinom(ceiling(n*muhat), n, R)
  min(1, p_b, exp(-n*kl))
}

rcps_threshold <- function(Yhat, Y, alpha=ALPHA, delta=DELTA, bound=B,
                           lo=THRESHOLD_MIN, hi=THRESHOLD_MAX) {
  n <- length(Y)
  if (n==0L) return(hi)
  target <- alpha/bound
  pval <- function(t) hb_pvalue(target, mean(loss_at(t,Yhat,Y))/bound, n)
  if (pval(lo) <= delta) return(lo)
  if (pval(hi) > delta) return(hi)
  for (i in 1:50) {
    mid <- (lo+hi)/2
    if (pval(mid) <= delta) hi <- mid else lo <- mid
  }
  hi
}

methods <- c('marg','mond','rcps')
labels <- c(marg='Marginal CRC', mond='Mondrian CRC (ours)',
            rcps='Groupwise RCPS (HB)')
Rg <- array(NA_real_, c(REPS,NG,length(methods)),
            dimnames=list(NULL,paste0('G',seq_len(NG)),methods))
Sg <- Rg
Rp <- matrix(NA_real_, REPS, length(methods), dimnames=list(NULL,methods))
Sz <- Rp
counts <- matrix(0L, REPS, NG)
floorv <- matrix(NA_real_, REPS, NG)
fall <- matrix(FALSE, REPS, NG)
Memp <- matrix(NA_integer_, REPS, NG)          # empirical per-group jump multiplicity
tmr <- c(marg=0, mond=0, rcps=0)               # accumulated calibration wall time (s)

for (r in seq_len(REPS)) {
  dc <- draw(NCAL); dt <- draw(NTEST)
  t0 <- proc.time()[3]; t_marg <- crc_threshold(dc$Yhat,dc$Y)
  tmr['marg'] <- tmr['marg'] + (proc.time()[3]-t0)
  t_group <- t_rcps <- numeric(NG)
  for (g in seq_len(NG)) {
    m <- dc$g==g
    ng <- sum(m); counts[r,g] <- ng
    Memp[r,g] <- jump_mult(dc$Yhat[m])
    t0 <- proc.time()[3]; t_group[g] <- crc_threshold(dc$Yhat[m],dc$Y[m])
    tmr['mond'] <- tmr['mond'] + (proc.time()[3]-t0)
    t0 <- proc.time()[3]; t_rcps[g] <- rcps_threshold(dc$Yhat[m],dc$Y[m])
    tmr['rcps'] <- tmr['rcps'] + (proc.time()[3]-t0)
    fall[r,g] <- t_group[g] == THRESHOLD_MAX
    ## Random-count floor uses the theoretical augmented-sample M_g=1,
    ## not approximate matches of candidate scores.
    floorv[r,g] <- ALPHA-(M_THEORY+1)*B/(ng+1)
  }
  for (nm in methods) {
    losses <- sizes <- numeric(NTEST)
    for (g in seq_len(NG)) {
      m <- dt$g==g
      if (!any(m)) stop('Empty evaluation group: raise NTEST or adjust DGP.')
      t <- switch(nm,marg=t_marg,mond=t_group[g],rcps=t_rcps[g])
      losses[m] <- loss_at(t,dt$Yhat[m],dt$Y[m])
      sizes[m] <- size_at(t,dt$Yhat[m])
      Rg[r,g,nm] <- mean(losses[m]); Sg[r,g,nm] <- mean(sizes[m])
    }
    Rp[r,nm] <- mean(losses); Sz[r,nm] <- mean(sizes)
  }
}

mc_halfwidth <- function(x) 1.96*sd(x)/sqrt(length(x))
cal_ms <- 1000*tmr/REPS
per_group <- do.call(rbind,lapply(methods,function(nm) {
  do.call(rbind,lapply(seq_len(NG),function(g) {
    data.frame(method=labels[nm],group=g,sigma=SIG[g],
               risk=mean(Rg[,g,nm]),risk_halfwidth=mc_halfwidth(Rg[,g,nm]),
               set_size=mean(Sg[,g,nm]),size_halfwidth=mc_halfwidth(Sg[,g,nm]),
               avg_cal_count=mean(counts[,g]),
               mondrian_fallback_rate=mean(fall[,g]))
  }))
}))
summary <- do.call(rbind,lapply(methods,function(nm) {
  grisk <- colMeans(Rg[,,nm])
  data.frame(method=labels[nm],pooled_risk=mean(Rp[,nm]),
             pooled_risk_halfwidth=mc_halfwidth(Rp[,nm]),
             worst_group_risk=max(grisk),
             max_abs_group_deviation=max(abs(grisk-ALPHA)),
             pooled_mean_set_size=mean(Sz[,nm]),
             cal_ms_per_rep=unname(cal_ms[nm]))
}))

cat(sprintf('Section 5.1: %d reps, n_cal=%d, n_test=%d, target=%.2f\n',
            REPS,NCAL,NTEST,ALPHA))
cat(sprintf('RCPS: per-group delta=%.2f (NOT a simultaneous group certificate)\n',DELTA))
cat(sprintf('M_g: theoretical=%d used in floor; empirical max=%d, mean=%.3f (checks no-ties)\n',
            M_THEORY, max(Memp), mean(Memp)))
cat(sprintf('mean random-count lower bound alpha-(M+1)B/(n_g+1)=%.4f\n\n', mean(floorv)))
print(summary,row.names=FALSE,digits=4)
cat('\n')
print(per_group,row.names=FALSE,digits=4)

dir.create('figures',showWarnings=FALSE,recursive=TRUE)
write.csv(summary,'figures/example1_summary.csv',row.names=FALSE)
write.csv(per_group,'figures/example1_per_group.csv',row.names=FALSE)

## Paper figure: both panels show +/-1.96 Monte Carlo SE error bars.
make_plot <- function(file='figures/fig_general.pdf',format=c('pdf','png')) {
  format <- match.arg(format)
  if (format=='pdf') pdf(file,width=10.4,height=4.4,useDingbats=FALSE)
  else png(file,width=1300,height=550,res=125)
  old <- par(mfrow=c(1,2),mar=c(4.5,4.4,2,1),oma=c(0,0,0,0))
  on.exit({par(old);dev.off()},add=TRUE)
  cols <- c(marg='grey55',mond='firebrick',rcps='steelblue')
  w <- .24; gx <- seq_len(NG)
  for (panel in c('risk','size')) {
    is_risk <- panel=='risk'
    ylim <- if (is_risk) c(0,1.22) else c(0,3.65)
    plot(NA,xlim=c(.5,NG+.5),ylim=ylim,xaxt='n',xlab='Noise scale by group',
         ylab=if (is_risk) 'Group-conditional ordinal risk' else 'Mean set size',
         main=if (is_risk) 'Conditional risk' else 'Prediction-set size')
    if (is_risk) {
      rect(.5,mean(floorv),NG+.5,ALPHA,col='grey93',border=NA)
      abline(h=ALPHA,lty=2,lwd=1.2)
    }
    for (i in seq_along(methods)) {
      nm <- methods[i]; x <- gx+(i-2)*w
      mu <- if(is_risk) colMeans(Rg[,,nm]) else colMeans(Sg[,,nm])
      hw <- if(is_risk) vapply(seq_len(NG),function(g)mc_halfwidth(Rg[,g,nm]),numeric(1))
            else vapply(seq_len(NG),function(g)mc_halfwidth(Sg[,g,nm]),numeric(1))
      rect(x-w/2,0,x+w/2,mu,col=cols[nm],border=NA)
      segments(x,mu-hw,x,mu+hw,lwd=1.2)
      segments(x-.045,mu-hw,x+.045,mu-hw,lwd=1.2)
      segments(x-.045,mu+hw,x+.045,mu+hw,lwd=1.2)
    }
    axis(1,gx,paste0('sigma=',SIG))
    legend('topleft',legend=labels,fill=cols,bty='n',cex=.67)
  }
  invisible(file)
}
make_plot()
make_plot('figures/fig_general.png','png')
cat('\nSaved figures/fig_general.pdf, figures/fig_general.png, and two CSV files.\n')
