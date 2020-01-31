setwd("~/Documents/TraumaMatrix/CausalInference/AcideTranexamique/Data/")

df <- read.csv('2019-10-23_data_preprocessed_tbi_individuals.csv', row.names = 1)

confounders <- c("Trauma.center", 
                 "Cardiac.arrest.ph", "SBP.ph.min", "DBP.ph.min", "HR.ph.max", 
                 "SBP.ph", "DBP.ph", "HR.ph", "Shock.index.ph", 
                 "Cristalloid.volume", "Colloid.volume", 
                 "SpO2.ph.min", "Vasopressor.therapy", "HemoCue.init", "Delta.hemoCue",
                 "Delta.shock.index", "AIS.external")

df <- cbind(df[,c('Tranexamic.acid','Death')], df[,confounders])

levels(df$Cardiac.arrest.ph) <- c(0,1)
df$Cardiac.arrest.ph <- as.integer(as.character(df$Cardiac.arrest.ph))
write.table(df, row.names = F, col.names = F, sep = ',',
          "~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/data/tb/tb_tbi_17conf.csv")

write(colnames(df), 
      "~/Documents/TraumaMatrix/CausalInference/Simulations/miss-vae/data/tb/columns.txt")
