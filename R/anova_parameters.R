library(FactoMineR)

setwd("~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/")

data.dir <- "./results/"
fig.dir <- "./figures/"

model <- "dlvm"

df <- read.csv(paste0(data.dir,"exp_24.1_10_choux.csv_temp"), row.names = 1)
df <- df[which(df$citcio == "False" & df$seed == 0),]
df_wy <- read.csv(paste0(data.dir,"exp_24.1_10_wy.csv_temp"), row.names = 1)
df_wy <- df_wy[which(df_wy$citcio == "False" & df_wy$prop_miss != 0 & df_wy$seed == 0),]

df <- rbind(df, df_wy)

summary(df)
df$sig_prior <- as.factor(df$sig_prior)
df$p <- as.factor(df$p)
df$n_epochs <- as.factor(df$n_epochs)
df$prop_miss <- as.factor(df$prop_miss)
df$add_wy <- as.factor(df$add_wy)
df$n <- as.factor(df$n)
summary(df)

if (model == "dlvm"){
df_dlvm <-df[which(df$model=="dlvm"),]

df_dlvm$bias_ols = abs(1-df_dlvm$tau_ols)
df_dlvm$bias_dr = abs(1-df_dlvm$tau_dr)

res_dr_init <- AovSum(bias_dr ~  n + p + prop_miss + n_epochs + sig_prior + add_wy +
                                p:prop_miss + p:sig_prior + p:add_wy + p:n_epochs +
                                n:p + n:prop_miss + n:n_epochs + n:sig_prior + n:add_wy +
                                sig_prior:n_epochs + sig_prior:add_wy, 
                      data = df_dlvm)
res_dr_init$Ftest
res_dr_init$Ttest

res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior + add_wy +
                                   p:n_epochs + p:sig_prior + p:add_wy +
                                   n:p + n:n_epochs + n:sig_prior + n:add_wy +
                                   sig_prior:n_epochs, 
                                   data = df_dlvm)
res_dr_reduced$Ftest
res_dr_reduced$Ttest

res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior +
                           p:n_epochs + p:sig_prior + 
                           n:p + n:n_epochs + n:sig_prior + 
                           sig_prior:n_epochs, 
                         data = df_dlvm)
res_dr_reduced$Ftest
res_dr_reduced$Ttest



res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior +
                           p:sig_prior + 
                           n:p + n:n_epochs + n:sig_prior + 
                           sig_prior:n_epochs, 
                         data = df_dlvm)

res_dr_reduced$Ftest
res_dr_reduced$Ttest

res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior + 
                           n:p + n:n_epochs + n:sig_prior + 
                           sig_prior:n_epochs, 
                         data = df_dlvm)

res_dr_reduced$Ftest
res_dr_reduced$Ttest

res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior + 
                           n:p + n:n_epochs +
                           sig_prior:n_epochs, 
                         data = df_dlvm)

res_dr_reduced$Ftest
res_dr_reduced$Ttest

res_dr_final <- AovSum(tau_dr ~  n + p + n_epochs +  sig_prior + 
                         n:p + n:n_epochs, data = df_dlvm)
res_dr_final$Ftest
res_dr_final$Ttest


###### OLS
res_ols_init <- AovSum(bias_ols ~  n + p + prop_miss + n_epochs + sig_prior + add_wy +
                        p:prop_miss + p:sig_prior + p:add_wy + p:n_epochs +
                        n:p + n:prop_miss + n:n_epochs + n:sig_prior + n:add_wy +
                        sig_prior:n_epochs, 
                      data = df_dlvm)
res_ols_init$Ftest
res_ols_init$Ttest

res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + add_wy +
                         p:sig_prior + p:n_epochs + p:add_wy + 
                         n:p + n:n_epochs + n:sig_prior + n:add_wy +
                         sig_prior:n_epochs, 
                       data = df_dlvm)
res_ols_reduced$Ftest
res_ols_reduced$Ttest

res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + 
                            p:sig_prior + p:n_epochs +  
                            n:p + n:n_epochs + n:sig_prior +
                            sig_prior:n_epochs, 
                          data = df_dlvm)
res_ols_reduced$Ftest
res_ols_reduced$Ttest

res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + 
                            p:sig_prior +
                            n:p + n:n_epochs + n:sig_prior +
                            sig_prior:n_epochs, 
                          data = df_dlvm)
res_ols_reduced$Ftest
res_ols_reduced$Ttest

res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + 
                            p:sig_prior +
                            n:p + n:n_epochs + 
                            sig_prior:n_epochs, 
                          data = df_dlvm)
res_ols_reduced$Ftest
res_ols_reduced$Ttest

res_ols_final <- AovSum(tau_ols ~   n + p + n_epochs + sig_prior +
                          n:p + n:n_epochs + 
                          sig_prior:n_epochs,  data = df_dlvm)
res_ols_final$Ftest
res_ols_final$Ttest

} else {
  df_lrmf <-df[which(df$model=="lrmf"),]
  
  df_lrmf$bias_ols = abs(1-df_lrmf$tau_ols)
  df_lrmf$bias_dr = abs(1-df_lrmf$tau_dr)
  
  res_dr_init <- AovSum(bias_dr ~  n + p + prop_miss + n_epochs + sig_prior + add_wy +
                          p:prop_miss + p:sig_prior + p:add_wy + p:n_epochs +
                          n:p + n:prop_miss + n:n_epochs + n:sig_prior + n:add_wy +
                          sig_prior:n_epochs, 
                        data = df_lrmf)
  res_dr_init$Ftest
  res_dr_init$Ttest
  
  res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior + add_wy + 
                             p:add_wy + p:n_epochs +
                             n:p + n:n_epochs + n:sig_prior +  n:add_wy + 
                             sig_prior:n_epochs, 
                           data = df_lrmf)
  
  res_dr_reduced$Ftest
  res_dr_reduced$Ttest
  
  res_dr_reduced <- AovSum(tau_dr ~  n + p + n_epochs + sig_prior +
                             p:n_epochs +
                             n:p + n:n_epochs + n:sig_prior + 
                             sig_prior:n_epochs, 
                           data = df_lrmf)
  
  res_dr_reduced$Ftest
  res_dr_reduced$Ttest
  
  res_dr_final <- AovSum(tau_dr ~   n + p + n_epochs + sig_prior +
                           p:n_epochs +
                           n:p + n:sig_prior + 
                           sig_prior:n_epochs, data = df_lrmf)
  res_dr_final$Ftest
  res_dr_final$Ttest
  
  
  ###### OLS
  res_ols_init <- AovSum(bias_ols ~  n + p + prop_miss + n_epochs + sig_prior + add_wy +
                           p:prop_miss + p:sig_prior + p:add_wy + 
                           n:p + n:prop_miss + n:n_epochs + n:sig_prior + n:add_wy +
                           sig_prior:n_epochs, 
                         data = df_lrmf)
  res_ols_init$Ftest
  res_ols_init$Ttest
  
  res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + add_wy +
                              p:sig_prior + p:add_wy + 
                              n:p + n:n_epochs + n:sig_prior + n:add_wy +
                              sig_prior:n_epochs, 
                            data = df_lrmf)
  res_ols_reduced$Ftest
  res_ols_reduced$Ttest
  
  res_ols_reduced <- AovSum(tau_ols ~  n + p + n_epochs + sig_prior + 
                              p:sig_prior +
                              n:p + n:n_epochs + n:sig_prior + 
                              sig_prior:n_epochs, 
                            data = df_lrmf)
  res_ols_reduced$Ftest
  res_ols_reduced$Ttest
  
  res_ols_final <- AovSum(tau_ols ~   n + p + n_epochs + sig_prior + 
                            p:sig_prior +
                            n:p + n:sig_prior + 
                            sig_prior:n_epochs,  data = df_lrmf)
  res_ols_final$Ftest
  res_ols_final$Ttest
}

##### PLOT
results_ols <- data.frame(factor = rownames(res_ols_final$Ttest),
                          bias_ols = res_ols_final$Ttest[,1],
                          row.names = NULL)
results_dr <- data.frame(factor = rownames(res_dr_final$Ttest),
                          bias_dr = res_dr_final$Ttest[,1],
                         row.names = NULL)

ols_dr_diff <- setdiff(results_dr$factor, results_ols$factor) # factors in dr that are not in ols
dr_ols_diff <- setdiff(results_ols$factor, results_dr$factor) # factors in dr that are not in ols

results_ols <- rbind(results_ols[-1,], cbind(factor = ols_dr_diff, bias_ols = rep(0, length(ols_dr_diff))))
results_dr <- rbind(results_dr[-1,], cbind(factor = dr_ols_diff, bias_dr = rep(0, length(dr_ols_diff))))

results_ols <- results_ols[order(results_ols$factor),]
results_dr <- results_dr[order(results_dr$factor),]

results <- cbind(results_ols, bias_dr = results_dr[,2])
results <- data.frame(results[,2:3], row.names = results[,1])
results[,1] <- as.double(as.character(results[,1]))
results[,2] <- as.double(as.character(results[,2]))

# results <- cbind(results_ols, bias_dr = results_dr[,2])
# results <- data.frame(results, row.names = results[,1])
# 
# results <- tidyr::gather(results, key = "estimate", value = "bias", bias_dr, bias_ols)
# 
# results$bias <- as.double(results$bias)
# results$index_factor <- rep(1:length(unique(as.character(results$factor))),2)
# results$index_estimate <- rep(1:2, each=length(unique(as.character(results$factor))))
# 
# image(x = results$index_estimate, 
#       y = results$index_factor,
#       z = results$bias)


# library(RColorBrewer)
# #Get desired core colours from brewer
# cols0 <- rev(brewer.pal(n=10, name="PiYG"))
# 
# #Derive desired break/legend colours from gradient of selected brewer palette
# cols1 <- colorRampPalette(cols0, space="rgb")(10)
# 
# image(1:ncol(results), 
#       1:nrow(results), 
#       t(results), col = cols1)
# axis(1, c("bias_ols","bias_dr"))
# #axis(2, at = seq(100, 600, by = 100))

tt <- results
colnames(tt) <- c("ols", "dr")
tt$row <- rownames(tt)
tt_melt <- reshape2::melt(tt)

ggplot(data=tt_melt,
       aes(x=variable, y=row, fill=value)) + 
      geom_tile() + 
      scale_fill_gradient(low = "green", high = "red", name="effect on bias") +
      theme(axis.text.x = element_text(size=18,angle=60,vjust = 0.5),
            axis.text.y = element_text(size=18),
            legend.text = element_text(size=18),
            legend.title = element_text(size=18, face="bold"))+
      xlab("") + 
      ylab("") 

ggsave(paste0(fig.dir, Sys.Date(), "_",model, "_anova.pdf"), plot = last_plot(),
       width=11, height=8.5)

