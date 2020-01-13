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

sem <- function(x){
  return (sd(x)/max(1,sqrt(length(x))))
}

loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))

seeds <- 100
prop_miss_vals <- c(0,0.1,0.3)
methods <- c("glm", "grf.ate")


ate_true <- data.frame()
ate_naive <- data.frame()
ate_diff_reg <- data.frame()
ate_ipw <- data.frame()
ate_dr <- data.frame()

for (seed in 1:seeds){
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  for (j in vars.binary){
    x.r[,j] <- as.factor(x.r[,j])
  }
  for (prop_miss in prop_miss_vals){
    if (prop_miss == 0){
      X_imp <- x.r
    } else {
      amputed <- produce_NA(x.r, mechanism = "MCAR", perc.missing = prop_miss, seed = seed) 
      X_imp <- get_imputeMean(amputed$data.incomp)
    }
  
  
    ate_true <- rbind(ate_true, cbind(seed, prop_miss,"true", mean(mu.1 - mu.0)))
    
    ate_naive <- rbind(ate_naive, cbind(seed, prop_miss, "naive", mean(y[which(z==1)])-mean(y[which(z==0)]), mean(mu.1 - mu.0)))
    X_imp_1 <- X_imp[which(z==1),which(sapply(X_imp[which(z==1),], function(x) length(unique(x))>1))]
    X_imp_0 <- X_imp[which(z==0),which(sapply(X_imp[which(z==0),], function(x) length(unique(x))>1))]
    
    mu1.hat <- lm(y ~ ., data = data.frame(X_imp_1, y=y[which(z==1)]))
    mu1.hat <- predict(mu1.hat, X_imp[,which(sapply(X_imp[which(z==1),], function(x) length(unique(x))>1))])
    mu0.hat <- lm(y ~ ., data = data.frame(X_imp_0, y=y[which(z==0)]))
    mu0.hat <- predict(mu0.hat, X_imp[,which(sapply(X_imp[which(z==0),], function(x) length(unique(x))>1))])
    
    ate_diff_reg <- rbind(ate_diff_reg, cbind(seed, prop_miss,  "diff_reg", mean(mu1.hat-mu0.hat), mean(mu.1 - mu.0)))
    for (m in 1:length(methods)){
      if (!(methods[m] %in% c("grf.ate"))){
        res <- ipw(X=X_imp, outcome = y, treat = z,
                   ps.method = methods[m], seed = seed)
        ate_ipw <- rbind(ate_ipw,cbind(seed, prop_miss, methods[m], res[,2], res[,3], mean(mu.1 - mu.0)))
      }
      res <- dr(X=X_imp, 
                outcome = y, 
                treat = z,
                ps.method= methods[m],  
                target= "all", 
                seed = seed,
                out.method = methods[m])
      ate_dr <-  rbind(ate_dr, cbind(seed, prop_miss, methods[m], res$dr, res$se, mean(mu.1 - mu.0)))
      
    }
  }
}

ate_dr[,4] <- as.double(as.character(ate_dr[,4]))
ate_dr[,6] <- as.double(as.character(ate_dr[,6]))

res_dr <- ate_dr %>%
            group_by(prop_miss, V3) %>%
            mutate(n=n()) %>%
            summarize(delta = mean(abs(V4 - V6)), sem = sd(abs(V4 - V6))/sqrt(12)) 

#####################################################################################


library(causalToolbox)
packageVersion("causalToolbox")

setwd("~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/data/IHDP/csv/")

err <- c()
for (id in 1:10){
  X <- read.csv(paste0("R_ate_ihdp_npci_",id,".csv"),header = F)
  w <- X[,1]         
  y <- X[,2]
  mu1 <- X[,5]
  mu0 <- X[,4]
  X <- X[,6:30]
  tau <- mu1-mu0
  xl_bart <- X_BART(feat = X, tr = w, yobs = y)
  cate_esti_bart <- EstimateCate(xl_bart, X)
  err[id] <- mean((tau - cate_esti_bart)^2)
  #View(cbind(tau, cate_esti_bart))
}



#####################################################################################
ate_dr_mia <- data.frame()

for (seed in 101:1000){#seeds){
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  for (j in vars.binary){
    x.r[,j] <- as.factor(x.r[,j])
  }
  for (prop_miss in prop_miss_vals){
    if (prop_miss == 0){
      X_imp <- x.r
    } else {
      amputed <- produce_NA(x.r, mechanism = "MCAR", perc.missing = prop_miss, seed = seed) 
      X_imp <- prepare_data_ate(amputed$data.incomp, 
                                w=z, y=y, 
                                imputation.method = "mia.grf.ate", mi.m=1, 
                                mask= FALSE,
                                use.outcome=FALSE, use.interaction=FALSE)$df.imp
    }
    
    res <- dr(X=X_imp, 
              outcome = y, 
              treat = z,
              ps.method= "grf.ate",  
              target= "all", 
              seed = seed,
              out.method = "grf.ate")
    ate_dr_mia <-  rbind(ate_dr_mia, 
                          cbind(seed, prop_miss, "mia.grf", res$dr, res$se, mean(mu.1 - mu.0)))
      
  }
}

ate_dr_mia[,4] <- as.double(as.character(ate_dr_mia[,4]))
ate_dr_mia[,6] <- as.double(as.character(ate_dr_mia[,6]))

res_dr <- ate_dr_mia %>%
  #filter(as.integer(as.character(seed)) <=26) %>%
  group_by(prop_miss, V3) %>%
  mutate(n=n()) %>%
  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6))) 

#save(ate_dr_mia, file="~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_mia_grf.RData")

#####################################################################################
ate_ols_mf <- data.frame()
ate_dr_mf <- data.frame()

for (seed in 1:1000){#seeds){
  writeLines(paste0(seed," "))
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  #vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  #for (j in vars.binary){
  #  x.r[,j] <- as.factor(x.r[,j])
  #}
  for (prop_miss in prop_miss_vals){
    if (prop_miss == 0){
      X_imp <- x.r
    } else {
      amputed <- produce_NA(x.r, mechanism = "MCAR", perc.missing = prop_miss, seed = seed) 
      X_imp <- prepare_data_ate(amputed$data.incomp, 
                                w=z, y=y, 
                                imputation.method = "udell", mi.m=1, 
                                mask= FALSE,
                                use.outcome=FALSE, use.interaction=FALSE)$df.imp
    }
    
    res <- dr(X=X_imp, 
              outcome = y, 
              treat = z,
              ps.method= "glm",  
              target= "all", 
              seed = seed,
              out.method = "glm")
    ate_dr_mf <-  rbind(ate_dr_mf, 
                         cbind(seed, prop_miss, "mf", res$dr, res$se, mean(mu.1 - mu.0)))
    
    tau.hat <- lm(y ~ ., data = data.frame(cbind(z,X_imp)))$coefficients[2]
    ate_ols_mf <-  rbind(ate_ols_mf, 
                        cbind(seed, prop_miss, "mf", tau.hat,  mean(mu.1 - mu.0)))
  }
}

ate_ols_mf[,4] <- as.double(as.character(ate_ols_mf[,4]))
ate_ols_mf[,5] <- as.double(as.character(ate_ols_mf[,5]))

ate_dr_mf[,4] <- as.double(as.character(ate_dr_mf[,4]))
ate_dr_mf[,6] <- as.double(as.character(ate_dr_mf[,6]))

res_ols_mf <- ate_ols_mf %>%
                filter(as.integer(as.character(seed)) <=26) %>%
                group_by(prop_miss, V3) %>%
                mutate(n=n()) %>%
                summarize(delta = mean(abs(tau.hat - V5)), sem = sd(abs(tau.hat - V5))/sqrt(12))

res_dr_mf <- ate_dr_mf %>%
                filter(as.integer(as.character(seed)) <=26) %>%
                group_by(prop_miss, V3) %>%
                mutate(n=n()) %>%
                summarize(delta = mean(abs(V4 - V6)), sem = sd(abs(V4 - V6))/sqrt(12))

#save(ate_dr_mf, ate_ols_mf, file="~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_mf.RData")

#####################################################################################
ate_ols_mice <- data.frame()
ate_dr_mice <- data.frame()

for (seed in 887:1000){#seeds){
  print(seed)
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = "all", setting = "A")
  
  x.r <- data.frame(x.r)
  vars.binary <- colnames(x.r)[apply(x.r, 2, FUN = function(e) length(unique(e))==2)]
  for (j in vars.binary){
    x.r[,j] <- as.factor(x.r[,j])
  }
  for (prop_miss in c(0.5)){#prop_miss_vals){
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
                ps.method= "glm",  
                target= "all", 
                seed = seed,
                out.method = "glm")
      ate_dr_mice <-  rbind(ate_dr_mice, 
                          cbind(seed, prop_miss, "mice.quant", res$dr, res$se, mean(mu.1 - mu.0)))
      
      tau.hat <- lm(y ~ ., data = data.frame(cbind(z,X_imp)))$coefficients[2]
      ate_ols_mice <-  rbind(ate_ols_mice, 
                           cbind(seed, prop_miss, "mice.quant", tau.hat,  mean(mu.1 - mu.0)))
    } else {
      res_mice <- c()
      for (k in 1:10){
        try(res_mice <- rbind(res_mice, dr(X=data.frame(X_imp[[k]]), outcome = y, treat = z,
                                 ps.method= "glm",  
                                 target= "all", 
                                 seed = seed,
                                 out.method = "glm")))
        
      }
  
      ate_dr_mice <-  rbind(ate_dr_mice, 
                            cbind(seed, prop_miss, "mice.quant", mean(res_mice[,1]), mean(res_mice[,2]), mean(mu.1 - mu.0)))
      
      res_mice <- c()
      for (k in 1:10){
        res_mice <- c(res_mice, lm(y ~ ., data = data.frame(cbind(y,z,X_imp[[k]])))$coefficients[2])
      }
      ate_ols_mice <-  rbind(ate_ols_mice, 
                             cbind(seed, prop_miss, "mice.quant", tau.hat = mean(res_mice),  mean(mu.1 - mu.0)))
    }
  }
}

ate_ols_mice[,4] <- as.double(as.character(ate_ols_mice[,4]))
ate_ols_mice[,5] <- as.double(as.character(ate_ols_mice[,5]))

ate_dr_mice[,4] <- as.double(as.character(ate_dr_mice[,4]))
ate_dr_mice[,6] <- as.double(as.character(ate_dr_mice[,6]))

res_ols_mice <- ate_ols_mice %>%
                  #filter(as.integer(as.character(seed)) <=26) %>%
                  group_by(prop_miss, V3) %>%
                  mutate(n=n()) %>%
                  summarize(delta = mean(abs(tau.hat - V5)), sem = sem(abs(tau.hat - V5)))

res_dr_mice <- ate_dr_mice %>%
                  #filter(as.integer(as.character(seed)) <=26) %>%
                  group_by(prop_miss, V3) %>%
                  mutate(n=n()) %>%
                  summarize(delta = mean(abs(V4 - V6)), sem = sem(abs(V4 - V6)))
res_ols_mice
res_dr_mice

#save(ate_ols_mice,ate_dr_mice, file="~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/results/ihdp_mice_ols_grf.RData")
