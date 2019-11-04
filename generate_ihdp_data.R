setwd("~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/data/")

source("./vdorie-npci/data.R")
source("./vdorie-npci/util.R")
source("./vdorie-npci/transform.R")


loadDataInCurrentEnvironment(covariates = "select", p.score = "none")

x <- as.matrix(x)
w <- rep(0.5, ncol(x))


for (seed in 1:1000){
  generateDataForIterInCurrentEnvironment(seed, x, z, w, overlap = TRUE, setting = "A")
  y.f <- if_else(z==1, y.1, y.0)
  y.cf <- if_else(z==0, y.1, y.0)

  write.csv(cbind(z, y.f, y.cf, mu.0, mu.1, x.r),
            file=paste0("./IHDP/csv/R_ihdp_npci_",seed,".csv"))
}

