############################### Developed by ###################################
## George William Kagugube
## Date: 15 November 2025
## Analysis of calcium dynamices and features from GCaMP7s, 2p imaging

## Clear the workspce here 
rm(list = ls())

## Set a global seed here for reproducibility
set.seed(101)

## Set the working directory here 
setwd('/Users/gwk/Desktop/Bioinformatics/two-photon-calcium-zebrafish/scripts/out_mn_pipeline')
## Source any extarnal files that maybe needed to complete the tasks here 

## Optional, but you can check the contents of the directory to be sure that the file of interest
# is located here 
dir()

## Functions needed for the analysis downstream
genotype_group <- function(df, genotype = 'WT', treatment = 'Exposed'){
  df$Genotype <- rep(genotype, nrow(df))
  df$Treatment <- rep(treatment, nrow(df))
  return(df)
}

barPlotting <- function(df, x, y, barr = 'upper_errorbar'){
  plot1 <- ggbarplot(data = df, y = y, x = x, 
                     add = c("mean_se"),
                     color = 'Group', fill = 'Group',
                     palette = c("#222222", "#E69F00", "grey"),
                     position = position_dodge(0.8),
                     error.plot = barr
  )
  
  print(plot1)
}

## Load the modules that are needed for the analysis here 
# Load required packages
library(ggplot2)
library(ggsignif)
library(gridExtra)
library(tidyverse)
library(ggpubr)
library(patchwork)
library(tidyplots)

## Load the data to be analysed here
wtUnexpo1 <- genotype_group(read.csv('./dff_wtUnexposed1_roi_metrics_by_window.csv'),
                            treatment = 'Unexposed', genotype = 'WT')

wtUnexpo2 <- genotype_group(read.csv('./dff_wtUnexposed2_roi_metrics_by_window.csv'),
                            treatment = 'Unexposed', genotype = 'WT')

wtExpo <- genotype_group(read.csv('./dff_wtExposed_roi_metrics_by_window.csv'),
                            treatment = 'Exposed', genotype = 'WT')
mutExpo1 <- genotype_group(read.csv('./dff_mutExposed1_roi_metrics_by_window.csv'),
                            treatment = 'Exposed', genotype = 'Mut')
mutExpo2 <- genotype_group(read.csv('./dff_mutExposed2_roi_metrics_by_window.csv'),
                            treatment = 'Exposed', genotype = 'Mut')
mutExpo3 <- genotype_group(read.csv('./dff_mutExposed3_roi_metrics_by_window.csv'),
                            treatment = 'Exposed', genotype = 'Mut')
mutExpo4 <- genotype_group(read.csv('./dff_mutExposed4_roi_metrics_by_window.csv'),
                            treatment = 'Exposed', genotype = 'Mut')

## Combine the datasets here  into one huge dataset
cdataframe <- rbind(wtUnexpo1, wtUnexpo2, wtExpo, mutExpo1,
                    mutExpo2, mutExpo3, mutExpo4)

## Create another categorical group to use here 
cdataframe$Group = paste(cdataframe$Genotype,cdataframe$Treatment, sep = '_')

attach(cdataframe)

## Order the catergorical vairables for better visualisations hereafter
cdataframe$Group <- ordered(cdataframe$Group,
                            levels = c('WT_Unexposed', 'WT_Exposed', 'Mut_Exposed'))
cdataframe$Genotype <- ordered(cdataframe$Genotype, 
                               level = c('WT', 'Mutant'))
cdataframe$Treatment <- ordered(cdataframe$Treatment,
                                levels = c('Unexposed', 'Exposed'))
cdataframe$window <- ordered(cdataframe$window,
                            levels = c('pre', 'during', 'post'))

## Explore the dataset here
dff <- cdataframe |>
  filter(responsive == 'True' & window == 'during') 


## Plotting starts from here 
barPlotting(dff, x = 'Group', y = 'median_IEI_s')

dff |>
  tidyplot(x = Group, y = auc_per_s, color = Group) |>
  add_mean_bar(alpha = 0.4) |>
  add_sem_errorbar() |>
  add_data_points_beeswarm()

######################### Statistics ##############################
stats <- dff |>
  filter(window == 'post')
res.aov <- aov(burstiness ~ Group, data = dff)
# Summary of the analysis
summary(res.aov)

TukeyHSD(res.aov)

