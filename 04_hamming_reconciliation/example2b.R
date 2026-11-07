## Section 5.2 E4b: residual reconciliation gap under Hamming-to-set loss
## Base R; corrected implementation of Section 4's overlapping-group union.
## Scenario A: G_j={x:m_j=1} feature-defined, union over APPLICABLE groups only.
## Scenario B: G_j={y:y_j=1} focal-label groups, no background group.
##             Candidate 000 is ineligible, not admitted by an extra cutoff.
## The groups need not cover the response space for groupwise risk guarantees.
## All shaped-slice losses are zero at threshold 42 (scores are <=41.447).
## Outputs: figures/fig_overlap_hamming.pdf, .png, and example2b CSVs.

set.seed(20261025)
L <- 3L; B <- 3; ALPHA <- 0.50; NV <- 2L^L
VEC <- t(sapply(0:(NV-1L), function(a)
  bitwAnd(bitwShiftR(a, 0:(L-1L)), 1L)))
DH <- as.matrix(dist(VEC, method="manhattan"))
RHO <- 0.4; GAM <- 1.3; NOISE <- 1.0
NS <- c(400L, 1600L, 6400L, 25600L)
REPS <- 100L; NEVAL <- 15000L; POP <- 400000L
POW2 <- 2^(0:(L-1L)); LMIN <- -2; LMAX <- 42

## q is a score component, not asserted to be a calibrated probability.
gen <- function(n) {
  f <- matrix(rnorm(n), n, 1L)
  e <- matrix(rnorm(n*L), n, L)
  z <- sqrt(RHO)*f[,rep(1L,L)] + sqrt(1-RHO)*e
  Y <- 0L + (z > 0)
  eta <- GAM*z + NOISE*matrix(rnorm(n*L), n, L)
  q <- pmin(pmax(plogis(eta), 1e-6), 1-1e-6)
  base <- rowSums(-log1p(-q))
  delta <- log1p(-q)-log(q)
  sc <- base + delta %*% t(VEC)
  yid <- as.integer(round(Y %*% POW2))
  m <- matrix(sample(0:1, n*L, replace=TRUE), n, L)
  list(Y=Y, yid=yid, sc=sc, m=m)
}

## cut is length NV (candidatewise), or a matrix nrow(sc) x NV.
admitted <- function(sc, cut) {
  if (is.matrix(cut)) stopifnot(identical(dim(sc), dim(cut)))
  if (is.matrix(cut)) sc <= cut else sweep(sc, 2L, cut, "<=")
}
loss_set <- function(sc, cut, dcol) {
  adm <- admitted(sc, cut)
  ds <- ifelse(adm, dcol, Inf)
  out <- do.call(pmin, as.data.frame(ds))
  out[!is.finite(out)] <- B
  out
}
size_set <- function(sc,cut) rowSums(admitted(sc,cut))

c_lab_slice <- function(j,t) {
  out <- rep(-Inf,NV); out[VEC[,j]==1L] <- t; out
}
## The 000 candidate does not belong to any focal group.
c_lab_union <- function(tf) vapply(seq_len(NV),function(a) {
  on <- which(VEC[a,]==1L)
  if(length(on)>0L) max(tf[on]) else -Inf
},numeric(1))
c_feat <- function(t) rep(t,NV)
## Only feature groups actually containing this test feature contribute.
c_feat_union <- function(m,tf) {
  trow <- apply(m,1L,function(row) {
    on <- which(row==1L)
    if(length(on)>0L) max(tf[on]) else -Inf
  })
  matrix(trow,nrow=nrow(m),ncol=NV)
}

## Standard corrected split CRC bisection (general structured loss).
crc <- function(sc,dcol,cutfn,alpha=ALPHA,lo=LMIN,hi=LMAX,iters=44L) {
  n <- nrow(sc)
  if(is.null(n)||n==0L) return(hi)
  criterion <- function(t) (sum(loss_set(sc,cutfn(t),dcol))+B)/(n+1L)
  if(criterion(hi)>alpha) return(hi)
  for (i in seq_len(iters)) {
    mid <- (lo+hi)/2
    if(criterion(mid)<=alpha) hi <- mid else lo <- mid
  }
  hi
}

## Population reference, shared generated distribution in both scenarios.
ep <- gen(POP)
Yp <- ep$Y; mp <- ep$m; scp <- ep$sc; DHp <- DH[ep$yid+1L,]
Rslice_pop <- function(t,mode) {
  if(mode=="B") { mask <- Yp[,1]==1L; cut <- c_lab_slice(1L,t) }
  else          { mask <- mp[,1]==1L; cut <- c_feat(t) }
  mean(loss_set(scp[mask,,drop=FALSE],cut,DHp[mask,,drop=FALSE]))
}
find_K <- function(mode,lo=LMIN,hi=LMAX) {
  for(i in 1:48) {
    mid <- (lo+hi)/2
    if(Rslice_pop(mid,mode)>ALPHA) lo <- mid else hi <- mid
  }
  hi
}
POPREF <- list(); population_rows <- list()
for(mode in c("A","B")) {
  K <- find_K(mode)
  if(mode=="B") {
    mask <- Yp[,1]==1L
    rs <- Rslice_pop(K,mode)
    rd <- mean(loss_set(scp[mask,,drop=FALSE],c_lab_union(rep(K,L)),
                        DHp[mask,,drop=FALSE]))
  } else {
    rs <- Rslice_pop(K,mode); rd <- rs
  }
  POPREF[[mode]] <- list(K=K,Rslice=rs,Rdep=rd,Delta=rs-rd)
  population_rows[[mode]] <- data.frame(
    scenario=if(mode=="A") "A: feature-defined" else "B: label-defined",
    K=K,R_slice_inf=rs,R_dep_inf=rd,Delta_star=rs-rd)
}
pop_df <- do.call(rbind,population_rows)

## Independent fresh evaluation sample for each replication.
mc_hw <- function(v) 1.96*sd(v)/sqrt(length(v))
store <- list(A=list(),B=list()); summary_rows <- list()
for(mode in c("A","B")) for(ni in seq_along(NS)) {
  n <- NS[ni]
  Rs <- Rd <- H <- BdS <- spread <- numeric(REPS)
  elapsed <- numeric(REPS)
  for(r in seq_len(REPS)) {
    dc <- gen(n); DHc <- DH[dc$yid+1L,]
    t0 <- proc.time()[3]
    if(mode=="B") {
      tf <- vapply(seq_len(L),function(j) {
        sel <- dc$Y[,j]==1L
        crc(dc$sc[sel,,drop=FALSE],DHc[sel,,drop=FALSE],
            function(t)c_lab_slice(j,t))
      },numeric(1))
      cut_union <- c_lab_union(tf)
      cut_slice <- function(j) c_lab_slice(j,tf[j])
    } else {
      tf <- vapply(seq_len(L),function(j) {
        sel <- dc$m[,j]==1L
        crc(dc$sc[sel,,drop=FALSE],DHc[sel,,drop=FALSE],c_feat)
      },numeric(1))
      cut_slice <- function(j) c_feat(tf[j])
    }
    elapsed[r] <- proc.time()[3]-t0
    spread[r] <- diff(range(tf))
    et <- gen(NEVAL); DHt <- DH[et$yid+1L,]
    if(mode=="A") cut_union <- c_feat_union(et$m,tf)
    risk_slice_j <- risk_dep_j <- size_delta_j <- numeric(L)
    for(j in seq_len(L)) {
      idx <- if(mode=="B") et$Y[,j]==1L else et$m[,j]==1L
      ss <- et$sc[idx,,drop=FALSE]; d <- DHt[idx,,drop=FALSE]
      cs <- cut_slice(j)
      cu <- if(mode=="A") cut_union[idx,,drop=FALSE] else cut_union
      ls <- loss_set(ss,cs,d); ld <- loss_set(ss,cu,d)
      sd <- size_set(ss,cu)-size_set(ss,cs)
      ## Pointwise lemma must hold for every observation in each group.
      stopifnot(all(ld <= ls),all(sd >= 0),all(ls-ld <= B*sd))
      risk_slice_j[j] <- mean(ls); risk_dep_j[j] <- mean(ld)
      size_delta_j[j] <- mean(sd)
    }
    Rs[r] <- mean(risk_slice_j); Rd[r] <- mean(risk_dep_j)
    H[r] <- mean(risk_slice_j-risk_dep_j)
    BdS[r] <- B*mean(size_delta_j)
  }
  store[[mode]][[as.character(n)]] <- list(Rs=Rs,Rd=Rd,H=H,BdS=BdS,spread=spread)
  summary_rows[[paste(mode,n)]] <- data.frame(
    scenario=if(mode=="A") "A: feature-defined" else "B: label-defined",
    n=n,R_slice=mean(Rs),R_slice_hw=mc_hw(Rs),
    R_dep=mean(Rd),R_dep_hw=mc_hw(Rd),H=mean(H),B_dS=mean(BdS),
    cutoff_spread=mean(spread),
    ineq_ok=all(H >= -1e-12 & H <= BdS+1e-12),
    cal_ms_per_rep=1000*mean(elapsed))
}
fin_df <- do.call(rbind,summary_rows)
dir.create("figures",recursive=TRUE,showWarnings=FALSE)
write.csv(pop_df,"figures/example2b_population.csv",row.names=FALSE)
write.csv(fin_df,"figures/example2b_summary.csv",row.names=FALSE)
cat("Population reference:\n");print(pop_df,row.names=FALSE,digits=5)
cat("\nFinite-sample:\n");print(fin_df,row.names=FALSE,digits=4)

make_plot <- function(file,format=c("pdf","png")) {
  format <- match.arg(format)
  if(format=="pdf") pdf(file,width=11,height=4.2,useDingbats=FALSE)
  else png(file,width=1375,height=525,res=125)
  old <- par(mfrow=c(1,2),mar=c(4.3,4.5,2.2,1))
  on.exit({par(old);dev.off()},add=TRUE)
  key <- as.character(NS)
  b <- POPREF[["B"]]
  sb <- sapply(key,function(k) mean(store[["B"]][[k]]$Rs))
  db <- sapply(key,function(k) mean(store[["B"]][[k]]$Rd))
  plot(NS,sb,type="o",pch=19,col="steelblue",log="x",xaxt="n",
       ylim=c(.25,.56),xlab="Calibration size",ylab="Conditional Hamming risk",
       main="Label-defined focal groups")
  axis(1,NS,NS)
  lines(NS,db,type="o",pch=15,col="firebrick")
  abline(h=ALPHA,lty=2,col="steelblue")
  abline(h=b$Rdep,lty=3,col="firebrick")
  legend("bottomleft",c("Slice risk","Deployed risk",
        "Population slice risk","Population deployed risk"),
        col=c("steelblue","firebrick","steelblue","firebrick"),
        lty=c(1,1,2,3),pch=c(19,15,NA,NA),bty="n",cex=.70)
  plot(NA,xlim=range(NS),ylim=c(0,1.45),log="x",xaxt="n",
       xlab="Calibration size",ylab="Risk reduction and bound",
       main="Risk--size inequality")
  axis(1,NS,NS)
  for(mode in c("A","B")) {
    col <- if(mode=="A") "forestgreen" else "firebrick"
    hh <- sapply(key,function(k) mean(store[[mode]][[k]]$H))
    bb <- sapply(key,function(k) mean(store[[mode]][[k]]$BdS))
    lines(NS,hh,type="o",pch=15,col=col)
    lines(NS,bb,type="o",pch=17,col=col,lty=2)
  }
  legend("right",c("Feature: risk reduction","Feature: B x size inflation",
                   "Label: risk reduction","Label: B x size inflation"),
         col=c("forestgreen","forestgreen","firebrick","firebrick"),
         lty=c(1,2,1,2),pch=c(15,17,15,17),bty="n",cex=.68)
  invisible(file)
}
make_plot("figures/fig_overlap_hamming.pdf","pdf")
make_plot("figures/fig_overlap_hamming.png","png")
cat("\nSaved corrected E4b figure and CSVs to figures/.\n")
