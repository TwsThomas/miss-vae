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

seeds <- 30
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
  
  ate_true <- rbind(ate_true, cbind(seed, "true", mean(mu.1 - mu.0)))
  
  ate_naive <- rbind(ate_naive, cbind(seed, "true", mean(y[which(z==1)])-mean(y[which(z==0)])))
  mu1.hat <- lm(y ~ ., data = data.frame(x.r[which(z==1),], y=y[which(z==1)]))
  mu1.hat <- predict(mu1.hat, x.r)
  mu0.hat <- lm(y ~ ., data = data.frame(x.r[which(z==0),], y=y[which(z==0)]))
  mu0.hat <- predict(mu0.hat, x.r)
  
  ate_diff_reg <- rbind(ate_diff_reg, cbind(seed, "diff_reg", mean(mu1.hat-mu0.hat)))
  for (m in 1:length(methods)){
    if (!(methods[m] %in% c("grf.ate"))){
      res <- ipw(X=x.r, outcome = y, treat = z,
          ps.method = methods[m], seed = seed)
      ate_ipw <- rbind(ate_ipw,cbind(seed, methods[m], res[,2], res[,3]))
    }
    res <- dr(X=x.r, 
              outcome = y, 
              treat = z,
              ps.method= methods[m],  
              target= "all", 
              seed = seed,
              out.method = methods[m])
    ate_dr <-  rbind(ate_dr, cbind(seed, methods[m], res$dr, res$se))
    
  }
}

ate_true[,3] <- as.double(as.character(ate_true[,3]))
ate_naive[,3] <- as.double(as.character(ate_naive[,3]))
ate_diff_reg[,3] <- as.double(as.character(ate_diff_reg[,3]))
ate_dr[,3:4] <- apply(ate_dr[,3:4], c(1,2), function(x) as.double(as.character(x)))
ate_ipw[,3:4] <- apply(ate_ipw[,3:4], c(1,2), function(x) as.double(as.character(x)))

results <- data.frame("seed"=as.integer(as.character(ate_true[,1])), 
                      "err_naive" = ate_true[,3] -ate_naive[,3],
                      "err_diff_reg" = ate_true[,3] - ate_diff_reg[,3],
                      "err_ipw" = ate_true[,3] -ate_ipw[,3],
                      "err_dr_glm" = ate_true[,3] - ate_dr[which(ate_dr[,2] == "glm"),3],
                      "err_dr_grf" = ate_true[,3] - ate_dr[which(ate_dr[,2] == "grf.ate"),3],
                      "naive" = ate_naive[,3],
                      "diff_reg"
                  ipw" = at
                      dr_glm" =
                      dr_grf" =)

sapply(results[,2:6], mean)

sapply(results[,2:6], function(x) mean(abs(x)))
