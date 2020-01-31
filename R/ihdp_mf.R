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



ate_grf_mf <- data.frame()
ate_mugrf_mf <- data.frame()
ate_ols_mf <- data.frame()
ate_dr_mf <- data.frame()

for (seed in 1:1000){
  writeLines(as.character(seed), sep = " ")
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  #x.r <- data.frame(x.r)
  #vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  #for (j in vars.binary){
  #  x.r[,j] <- as.factor(x.r[,j])
  #}
  Z_imp <- read.csv(paste0(data.dir, 'ihdp_prop_miss',
                               format(round(prop_miss,1), nsmall=1),
                               'set_id1_zhat_mf_soft.csv'), sep = ',', header=F)
  
  res <- dr(X=Z_imp, 
            outcome = y, 
            treat = z,
            ps.method= "grf.ate",  
            target= "all", 
            seed = seed,
            out.method = "grf.ate")
  ate_grf_mf <-  rbind(ate_grf_mf, 
                       cbind(seed, prop_miss, "mf", res$dr, res$se, mean(mu.1 - mu.0)))
  
  res <- grf_diffmu(Z_imp, y = y, w = z)
  ate_mugrf_mf <-  rbind(ate_mugrf_mf, 
                         cbind(seed, prop_miss, "mf", res, mean(mu.1 - mu.0)))

  res <- dr(X=Z_imp, 
            outcome = y, 
            treat = z,
            ps.method= "glm",  
            target= "all", 
            seed = seed,
            out.method = "glm")
  ate_dr_mf <-  rbind(ate_dr_mf, 
                       cbind(seed, prop_miss, "mf", res$dr, res$se, mean(mu.1 - mu.0)))
  
  tau.hat <- lm(y ~ ., data = data.frame(cbind(z,Z_imp)))$coefficients[2]
  ate_ols_mf <-  rbind(ate_ols_mf, 
                       cbind(seed, prop_miss, "mf", tau.hat,  mean(mu.1 - mu.0)))

  
  save(ate_grf_mf, ate_mugrf_mf, ate_ols_mf, ate_dr_mf,
     file = paste0(rdata.dir, "ihdp_prop_miss",format(round(prop_miss, 2), nsmall=1), 
                   "_mf",".RData"))

}






