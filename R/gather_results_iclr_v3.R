setwd("~/Documents/TraumaMatrix/CausalInference/Simulations/causal-inference-missing/IHDP/")
source("../load.R")

source("./vdorie-npci/data.R")
source("./vdorie-npci/util.R")
source("./vdorie-npci/transform.R")

source("../Helper/helper_causalInference.R")
source("../Helper/helper_simulations.R")
source("../Helper/helper_imputation.R")
source("../Helper/helper_udell.R")
source("../Utils/miss.saem.v2.R")
source("../Utils/amputation.R")


loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))

prop_miss_vals <- c(0,0.1,0.3, 0.5)


ate_grf_mdcproc <- data.frame()
ate_mugrf_mdcproc <- data.frame()
ate_grf_mdcmi <- data.frame()
ate_mugrf_mdcmi <- data.frame()

for (seed in 1:10){
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  #vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  #for (j in vars.binary){
  #  x.r[,j] <- as.factor(x.r[,j])
  #}
  for (prop_miss in prop_miss_vals){
    if (prop_miss == 0){
      Z_imp <- read.csv(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/ihdp_prop_miss',
                               '0',
                               'set_id1_zhat.csv'), sep = ';', header=F)
    } else {
      Z_imp <- read.csv(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/ihdp_prop_miss',
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
    
    # res_dr <- c()
    # res_mu <- c()
    # for (m in 0:199){
    #   if (prop_miss == 0){
    #     Z_imp <- read.csv(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/ihdp_prop_miss',
    #                              '0',
    #                              'set_id1_zhat_m',m,'.csv'), sep = ';', header=F)
    #   } else {
    #     Z_imp <- read.csv(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/ihdp_prop_miss',
    #                              format(round(prop_miss,1), nsmall=1),
    #                              'set_id1_zhat_m',m,'.csv'), sep = ';', header=F)
    #   }
    #   res_dr <- rbind(res_dr,dr(X=Z_imp, 
    #             outcome = y[2:length(y)], 
    #             treat = z[2:length(z)],
    #             ps.method= "grf.ate",  
    #             target= "all", 
    #             seed = seed,
    #             out.method = "grf.ate"))
    #   
    #   res_mu <- c(res_mu,grf_diffmu(Z_imp, y = y[2:length(y)], w = z[2:length(z)]))
    # }
    # ate_grf_mdcmi <-  rbind(ate_grf_mdcmi, 
    #                           cbind(seed, prop_miss, "mdcmi.grf", mean(res_dr[,1]), mean(res_dr[,2]), mean(mu.1 - mu.0)))
    # ate_mugrf_mdcmi <-  rbind(ate_mugrf_mdcmi, 
    #                             cbind(seed, prop_miss, "mdcmi.grf", mean(res_mu), mean(mu.1 - mu.0)))
    # 
  }
}
  

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



######################################################################################

loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))

prop_miss <- 0.1

ate_grf_mice <- data.frame()
ate_mugrf_mice <- data.frame()

for (seed in 1:1){
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
    ate_dr_mice <-  rbind(ate_dr_mice, 
                          cbind(seed, prop_miss, "mice.quant", res$dr, res$se, mean(mu.1 - mu.0)))
    
    tau.hat <- grf_diffmu(X_imp, y = y, w = z)
    ate_ols_mice <-  rbind(ate_ols_mice, 
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
}


################################################################################

ate_mugrf_mdcproc_all <- c()
ate_mugrf_mdcmi_all <- c()
ate_grf_mdcproc_all <- c()
ate_grf_mdcmi_all <- c()
for (prop_miss in c(0, 0.1, 0.3, 0.5)){
  load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
              format(prop_miss,nsmall=1), '_mdc_grf_part1.RData'))

  ate_mugrf_mdcproc_all <- rbind(ate_mugrf_mdcproc_all,
                                 ate_mugrf_mdcproc)
  ate_mugrf_mdcmi_all <- rbind(ate_mugrf_mdcmi_all,
                                 ate_mugrf_mdcmi)
  ate_grf_mdcproc_all <- rbind(ate_grf_mdcproc_all,
                                 ate_grf_mdcproc)
  ate_grf_mdcmi_all <- rbind(ate_grf_mdcmi_all,
                               ate_grf_mdcmi)
  
  load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
              format(prop_miss,nsmall=1), '_mdc_grf.RData'))

  ate_mugrf_mdcproc_all <- rbind(ate_mugrf_mdcproc_all,
                                 ate_mugrf_mdcproc)
  ate_mugrf_mdcmi_all <- rbind(ate_mugrf_mdcmi_all,
                               ate_mugrf_mdcmi)
  ate_grf_mdcproc_all <- rbind(ate_grf_mdcproc_all,
                               ate_grf_mdcproc)
  ate_grf_mdcmi_all <- rbind(ate_grf_mdcmi_all,
                             ate_grf_mdcmi)

}

ate_mugrf_mdcproc_all[,4] <- as.double(as.character(ate_mugrf_mdcproc_all[,4]))
ate_mugrf_mdcmi_all[,4] <- as.double(as.character(ate_mugrf_mdcmi_all[,4]))

ate_grf_mdcproc_all[,4] <- as.double(as.character(ate_grf_mdcproc_all[,4]))
ate_grf_mdcmi_all[,4] <- as.double(as.character(ate_grf_mdcmi_all[,4]))

res_mugrf_mdcproc <- ate_mugrf_mdcproc_all %>%
  mutate(V5 = as.double(as.character(V5))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(res - V5)), sem = sem(abs(res - V5)))

res_grf_mdcproc <- ate_grf_mdcproc_all %>%
  mutate(V6 = as.double(as.character(V6))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))

res_mugrf_mdcmi <- ate_mugrf_mdcmi_all %>%
  mutate(V5 = as.double(as.character(V5))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V5)), sem = sem(abs(V4 - V5)))

res_grf_mdcmi <- ate_grf_mdcmi_all %>%
  mutate(V6 = as.double(as.character(V6))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))

print('MDC.process')
res_mugrf_mdcproc
res_grf_mdcproc
print('MDC.mi')
res_mugrf_mdcmi
res_grf_mdcmi



##########################################################################

ate_mugrf_mice_all <- c()
ate_grf_mice_all <- c()
for (prop_miss in c(0, 0.1, 0.3, 0.5)){
  if (prop_miss != 0){
    load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
                format(prop_miss,nsmall=1), '_mice_grf_part1.RData'))
    
    ate_mugrf_mice_all <- rbind(ate_mugrf_mice_all,
                                   ate_mugrf_mice)
    
    ate_grf_mice_all <- rbind(ate_grf_mice_all,
                               ate_grf_mice)
  }
  
  load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
              format(prop_miss,nsmall=1), '_mice_grf.RData'))
  
  ate_mugrf_mice_all <- rbind(ate_mugrf_mice_all,
                              ate_mugrf_mice)
  
  ate_grf_mice_all <- rbind(ate_grf_mice_all,
                            ate_grf_mice)
  
}

ate_mugrf_mice_all[,4] <- as.double(as.character(ate_mugrf_mice_all[,4]))

ate_grf_mice_all[,4] <- as.double(as.character(ate_grf_mice_all[,4]))

res_mugrf_mice <- ate_mugrf_mice_all %>%
  mutate(V5 = as.double(as.character(V5))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(tau.hat - V5)), sem = sem(abs(tau.hat - V5)))

res_grf_mice <- ate_grf_mice_all %>%
  mutate(V6 = as.double(as.character(V6))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))

res_mugrf_mice
res_grf_mice


###############################################################################
ate_ols_mf_all <- c()
ate_dr_mf_all <- c()
ate_grf_mf_all <- c()
ate_mugrf_mf_all <- c()

for (prop_miss in c(0, 0.1, 0.3, 0.5)){
  load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
              format(prop_miss,nsmall=1), '_mf.RData')) # mimi with only gaussian
  # load(paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_prop_miss',
  #             format(prop_miss,nsmall=1), '_mf_mimi.RData')) # mimi with gaussian + binom
  
  ate_mugrf_mf_all <- rbind(ate_mugrf_mf_all,
                              ate_mugrf_mf)
  
  ate_grf_mf_all <- rbind(ate_grf_mf_all,
                            ate_grf_mf)
  
  ate_dr_mf_all <- rbind(ate_dr_mf_all,
                          ate_dr_mf)
  
  ate_ols_mf_all <- rbind(ate_ols_mf_all,
                        ate_ols_mf)
  
}

ate_mugrf_mf_all[,4] <- as.double(as.character(ate_mugrf_mf_all[,4]))
ate_grf_mf_all[,4] <- as.double(as.character(ate_grf_mf_all[,4]))
ate_dr_mf_all[,4] <- as.double(as.character(ate_dr_mf_all[,4]))
ate_ols_mf_all[,4] <- as.double(as.character(ate_ols_mf_all[,4]))

res_mugrf_mf <- ate_mugrf_mf_all %>%
  mutate(V5 = as.double(as.character(V5))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(res - V5)), sem = sem(abs(res - V5)))

res_grf_mf <- ate_grf_mf_all %>%
  mutate(V6 = as.double(as.character(V6))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))

res_dr_mf <- ate_dr_mf_all %>%
  mutate(V6 = as.double(as.character(V6))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))

res_ols_mf <- ate_ols_mf_all %>%
  mutate(V5 = as.double(as.character(V5))) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(tau.hat - V5)), sem = sem(abs(tau.hat - V5)))

res_mugrf_mf
res_grf_mf
res_dr_mf
res_ols_mf

##################################################################################
loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))

prop_miss <- 0.1

ate_ols_mf <- data.frame()
ate_dr_mf <- data.frame()
ate_grf_mf <- data.frame()
ate_mugrf_mf <- data.frame()

# for (seed in 1:1){
#   writeLines(as.character(seed), sep=" ")
#   generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
# 
#   x.r <- data.frame(x.r)
#   vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
#   for (j in vars.binary){
#     x.r[,j] <- as.factor(x.r[,j])
#   }
#   if (prop_miss == 0){
#     X_imp <- x.r
#   } else {
#     amputed <- produce_NA(x.r, mechanism = "MCAR", perc.missing = prop_miss, seed = seed)
#     X_imp <- amputed$data.incomp
#   }
#   df <- X_imp
#   var.type <- sapply(data.frame(df), FUN = function(x) {if_else(is.numeric(x), "gaussian", "binomial")})
#   df[,var.type=="binomial"] <- sapply(df[,var.type=="binomial"], as.integer)
#   # var.type <- rep("gaussian", dim(df)[2])
#   lambdas <- cv.mimi(y=df, model = "low-rank", var.type, thresh=1e-4,
#                                               algo = "bcgd",
#                                                maxit = 100, max.rank = dim(df)[2], trace.it = T, parallel = T,
#                                                len = 15)
#   glrm_mod <- mimi::mimi(y=df, model = "low-rank", var.type = var.type, max.rank = dim(df)[2], algo="bcgd", lambda1=lambdas$lambda)
#   tt <- PCA(glrm_mod$theta)
#   #
#   # write.table(tt$svd$U, sep=',',file = paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/',
#   #                             'ihdp_prop_miss',
#   #                              format(round(prop_miss,1), nsmall=1),
#   #                             'set_id1_zhat_mf_soft.csv'), # when using mimi with gaussian only
#   #             row.names = F, col.names = F)
#   write.table(tt$svd$U, sep=',',file = paste0('~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp/',
#                                               'ihdp_prop_miss',
#                                               format(round(prop_miss,1), nsmall=1),
#                                               'set_id1_zhat_mf.csv'), # when using mimi with gaussian + binomial
#               row.names = F, col.names = F)
# }

sem <- function(x){
  return (sd(x)/max(1,sqrt(length(x))))
}
