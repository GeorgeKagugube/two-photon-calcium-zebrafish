###############################################################################
# This script analyses GCaMP7s activity extracted Python automated scripts ####
# Created by George W Kagugube                                             ####
# Date:                                                                    ####
###############################################################################
rm(list = ls())

set.seed(101)

setwd('/Users/gwk/Desktop/CalciumImaging/final_results')

dir()

###############################################################################
## Load the required libraries here
library(tidyverse)
library(ggplot2)
library(forcats)
library(gtsummary)
library(ggvenn)
library(RColorBrewer)
library(ggsignif)
library(gridExtra)
library(ggpubr)
###############################################################################

## Functions needed for the analysis downstream
genotype_group <- function(df, genotype = 'WT', treatment = 'Exposed'){
  df$Genotype <- rep(genotype, nrow(df))
  df$Treatment <- rep(treatment, nrow(df))
  return(df)
}

data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE),
      se = sd(x[[col]], na.rm=TRUE)/sqrt(nrow(x)))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

################################################################################
## Read in the Unexposed Mn data here
df_wtUnexp1 <- genotype_group(df = read.csv("./wt/tonic_metrics_wt_exposed1.csv"),
                              genotype = 'WT', treatment = 'Unexposed')
df_wtunexpo2 <- genotype_group(df = read.csv('./wt/tonic_metrics_wt_exposed2.csv'),
                               genotype = 'WT', treatment = 'Unexposed')

## Load the Exposed Mn data 
df_wtexpo <- genotype_group(df = read.csv('./wt_exposed/tonic_metrics_exposed_wt.csv'),
                            genotype = 'WT', treatment = 'Exposed')
  
## Load the Mutant exposed data here
df_mut1 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut1.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut2 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut2.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut3 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut3.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut4 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut4.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
                          
################################################################################
## Perform data exploration here 
head(df_wtUnexp1)
head(df_mut1)

## Univariant analysis of wt unexposed
boxplot(df_wtUnexp1$median_dff)
hist(df_wtUnexp1$mean_decay_tau_1e_s)
boxplot(df_wtunexpo2$mean_decay_tau_1e_s)
hist(df_wtunexpo2$mean_decay_tau_1e_s)

## Univariate analysis of wt exposed 
boxplot(df_wtUnexpo2$mean_decay_tau_1e_s)
hist(df_wtUnexpo2$mean_decay_tau_1e_s)
boxplot(df_wtunexpo2$mean_t90_s)
hist(df_wtunexpo2$mean_t90_s)

cdataframe <- rbind(df_wtUnexp1, df_wtunexpo2, df_wtexpo, df_mut1, df_mut2, df_mut3, df_mut4)
attach(cdataframe)
cdataframe$Genotype <- ordered(cdataframe$Genotype, 
                               level = c('WT', 'Mutant'))
cdataframe$Treatment <- ordered(cdataframe$Treatment,
                                levels = c('Unexposed', 'Exposed'))
cdataframe$class <- ordered(cdataframe$class,
                            levels = c('pre', 'during', 'post'))

## Create another categorical group to use here 
cdataframe$Group = paste(cdataframe$Genotype,cdataframe$Treatment, sep = '_')

## Export the processed clean data frame for eas of analysis next time 
getwd()
write.csv(cdataframe, 
          file = 'cleanedData_tonic.csv')

## Rearange the categorical variables here for better visualisation
attach(cdataframe)

df <- cdataframe |>
  filter(responsive == 'True') 

################## Bargraph ##########################
df |>
  #filter(class == 'during') |>
ggbarplot(y = "noise_sigma", x = "Group", 
            add = c("mean_se")) #+
 stat_compare_means(comparisons = list(c('WT_Unexposed', 'WT_Exposed'), 
                                       c('WT_Unexposed', 'Mutant_Exposed'),
                                        c('WT_Exposed', 'Mutant_Exposed')), 
                    label = "p.signif", 
                    label.y = c(0.7, 0.9, 1.0))
 
 ######################### Line graph#################
 df |>
   #filter(class == 'pre' | class == 'during') |>
 ggline( x = "Group", y = "median_dff", 
        add = c("mean_se"),
        color = "Group", palette = "jco")




anova(mean_peak_amp ~ Genotype + Treatment, data = df)

# Compute the analysis of variance
res.aov <- aov(mean_peak_amp ~ Group + class, data = df)
# Summary of the analysis
summary(res.aov)

TukeyHSD(res.aov)


dff <- read.csv('./wt/network_reorg_corr_during.csv')
head(dff)
