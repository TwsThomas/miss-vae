library(dplyr)
setwd("~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/")

results <- read.csv('./results/ihdp_cv_10.1_11.csv_temp')

results2 <- read.csv('./results/ihdp_cv_10.2_11.csv_temp')
results <- rbind(results, results2)
res <- results %>%
  group_by(prop_miss, d_miwae, sig_prior) %>%
  summarize(avg_elbo = mean(elbo))
