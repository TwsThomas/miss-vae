#!/usr/bin/env Rscript
################################################################
# Compare different ATE estimators that handle missing values
################################################################

# script.dir <- dirname(sys.frame(1)$ofile)
script.dir <- "~/CausalInference/Simulations/causal-inference-missing/IHDP/"
setwd(script.dir)
source("../load.R")
source("./vdorie-npci/data.R")
source("./vdorie-npci/util.R")
source("./vdorie-npci/transform.R")

library("optparse")
 
option_list = list(
  make_option(c("--propmiss"), type="double", default=0.1, 
              help="proportion of missing values [default= %default]", metavar="number")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


grf_diffmu <- function(X, y, w){
  var.factor <- colnames(X)[sapply(X, is.factor)]
  X.1.m = model.matrix(~. -1, data=X[which(w==1),], 
                       contrasts = as.list(setNames(rep("contr.sum", length(var.factor)), var.factor)))
  X.0.m = model.matrix(~. -1, data=X[which(w==0),],
                       contrasts = as.list(setNames(rep("contr.sum", length(var.factor)), var.factor)))
  
  X.m = model.matrix(~. -1, data=X, 
                     contrasts = as.list(setNames(rep("contr.sum", length(var.factor)), var.factor)))
  
  forest.1.Y = regression_forest(X.1.m, y[which(w==1)], tune.parameters = TRUE)
  y_1.hat = predict(forest.1.Y, X.m)$predictions
  
  forest.0.Y = regression_forest(X.0.m, y[which(w==0)], tune.parameters = TRUE)
  y_0.hat = predict(forest.0.Y, X.m)$predictions
  
  delta_i <- y_1.hat - y_0.hat
  return(mean(delta_i))
}


data.dir <- "./data/"
rdata.dir <- "./RData/"


prop_miss <- opt$propmiss




loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))



ate_grf_mdcproc <- data.frame()
ate_mugrf_mdcproc <- data.frame()
ate_grf_mdcmi <- data.frame()
ate_mugrf_mdcmi <- data.frame()

for (seed in 35:1000){
  writeLines(as.character(seed), sep = " ")
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  #vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  #for (j in vars.binary){
  #  x.r[,j] <- as.factor(x.r[,j])
  #}
  if (prop_miss == 0){
      Z_imp <- read.csv(paste0(data.dir, 'ihdp_prop_miss',
                               '0',
                               'set_id1_zhat.csv'), sep = ';', header=F)
  } else {
      Z_imp <- read.csv(paste0(data.dir, 'ihdp_prop_miss',
                               format(round(prop_miss,1), nsmall=1),
                               'set_id1_zhat.csv'), sep = ';', header=F)
  }
  res <- dr(X=Z_imp, 
            outcome = y[2:length(y)], 
            treat = z[2:length(z)],
            ps.method= "grf.ate",  
            target= "all", 
            seed = seed,
            out.method = "grf.ate")
  ate_grf_mdcproc <-  rbind(ate_grf_mdcproc, 
                       cbind(seed, prop_miss, "mdcproc.grf", res$dr, res$se, mean(mu.1 - mu.0)))
  
  res <- grf_diffmu(Z_imp, y = y[2:length(y)], w = z[2:length(z)])
  ate_mugrf_mdcproc <-  rbind(ate_mugrf_mdcproc, 
                            cbind(seed, prop_miss, "mdcproc.grf", res, mean(mu.1 - mu.0)))
  
  res_dr <- c()
  res_mu <- c()
  for (m in 0:199){
    if (prop_miss == 0){
      Z_imp <- read.csv(paste0(data.dir, 'ihdp_prop_miss',
                               '0',
                               'set_id1_zhat_m',m,'.csv'), sep = ';', header=F)
    } else {
      Z_imp <- read.csv(paste0(data.dir,'ihdp_prop_miss',
                               format(round(prop_miss,1), nsmall=1),
                               'set_id1_zhat_m',m,'.csv'), sep = ';', header=F)
    }
    res_dr <- rbind(res_dr,dr(X=Z_imp, 
              outcome = y[2:length(y)], 
              treat = z[2:length(z)],
              ps.method= "grf.ate",  
              target= "all", 
              seed = seed,
              out.method = "grf.ate"))
    
    res_mu <- c(res_mu,grf_diffmu(Z_imp, y = y[2:length(y)], w = z[2:length(z)]))
  }
  ate_grf_mdcmi <-  rbind(ate_grf_mdcmi, 
                            cbind(seed, prop_miss, "mdcmi.grf", mean(res_dr[,1]), mean(res_dr[,2]), mean(mu.1 - mu.0)))
  ate_mugrf_mdcmi <-  rbind(ate_mugrf_mdcmi, 
                              cbind(seed, prop_miss, "mdcmi.grf", mean(res_mu), mean(mu.1 - mu.0)))
  
  save(ate_grf_mdcproc, ate_mugrf_mdcproc, ate_grf_mdcmi, ate_mugrf_mdcmi,
     file = paste0(rdata.dir, "ihdp_prop_miss",format(round(prop_miss, 2), nsmall=1), 
                   "_mdc_grf",".RData"))

}






