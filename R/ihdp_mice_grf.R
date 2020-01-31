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



ate_grf_mice <- data.frame()
ate_mugrf_mice <- data.frame()

for (seed in 320:1000){
  writeLines(as.character(seed), sep=" ")
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  for (j in vars.binary){
    x.r[,j] <- as.factor(x.r[,j])
  }
  if (prop_miss == 0){
      X_imp <- x.r
  } else {
    amputed <- produce_NA(x.r, mechanism = "MCAR", perc.missing = prop_miss, seed = seed) 
    X_imp <- prepare_data_ate(amputed$data.incomp, 
                              w=z, y=y, 
                              imputation.method = "mice", mi.m=10, 
                              mask= FALSE,
                              use.outcome=TRUE, use.interaction=FALSE)$df.imp
  }
  if (prop_miss == 0) {
    res <- dr(X=X_imp, 
              outcome = y, 
              treat = z,
              ps.method= "grf.ate",  
              target= "all", 
              seed = seed,
              out.method = "grf.ate")
    ate_grf_mice <-  rbind(ate_grf_mice, 
                        cbind(seed, prop_miss, "mice.quant", res$dr, res$se, mean(mu.1 - mu.0)))
    
    tau.hat <- grf_diffmu(X_imp, y = y, w = z)
    ate_mugrf_mice <-  rbind(ate_mugrf_mice, 
                         cbind(seed, prop_miss, "mice.quant", tau.hat,  mean(mu.1 - mu.0)))
  } else {
    res_mice <- c()
    for (k in 1:10){
      try(res_mice <- rbind(res_mice, dr(X=data.frame(X_imp[[k]]), outcome = y, treat = z,
                               ps.method= "grf.ate",  
                               target= "all", 
                               seed = seed,
                               out.method = "grf.ate")))
      
    }

    ate_grf_mice <-  rbind(ate_grf_mice, 
                          cbind(seed, prop_miss, "mice.quant", mean(res_mice[,1]), mean(res_mice[,2]), mean(mu.1 - mu.0)))
    
    res_mice <- c()
    for (k in 1:10){
      res_mice <- c(res_mice, grf_diffmu(X_imp[[k]], y = y, w = z))
    }
    ate_mugrf_mice <-  rbind(ate_mugrf_mice, 
                             cbind(seed, prop_miss, "mice.quant", tau.hat = mean(res_mice),  mean(mu.1 - mu.0)))
  }
  save(ate_grf_mice, ate_mugrf_mice, 
     file = paste0(rdata.dir, "ihdp_prop_miss",format(round(prop_miss, 2), nsmall=1), 
                   "_mice_grf",".RData"))
}








