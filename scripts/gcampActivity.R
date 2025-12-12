############################### Developed by ###################################
## George William Kagugube
## Date: 15 November 2025
## Analysis of calcium dynamices and features from GCaMP7s, 2p imaging

## Clear the workspce here 
rm(list = ls())

## Set a global seed here for reproducibility
set.seed(101)

## Set the working directory here
setwd('/Users/gwk/Desktop/CalciumImaging/gcamp_activity')

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

wtUnexpo1 <- genotype_group(read.csv('wtUnexposed1.csv'),treatment = 'Unexposed',
                     genotype = 'WT')
wtUnexpo2 <- genotype_group(read.csv('wtUnexposed2.csv'),treatment = 'Unexposed',
                            genotype = 'WT')

wtExpo <- genotype_group(read.csv('wtExposed.csv'),treatment = 'Exposed',
                         genotype = 'WT')
  
mutExpo1 <- genotype_group(read.csv('mutExposed1.csv'),treatment = 'Exposed',
                           genotype = 'Mut')

mutExpo2 <- genotype_group(read.csv('mutExposed2.csv'),treatment = 'Exposed',
                           genotype = 'Mut')

mutExpo3 <- genotype_group(read.csv('mutExposed3.csv'),treatment = 'Exposed',
                           genotype = 'Mut')

mutExpo4 <- genotype_group(read.csv('mutExposed4.csv'),treatment = 'Exposed',
                           genotype = 'Mut')

## Combine the datasets here  into one huge dataset
cdataframe <- rbind(wtUnexpo1, wtUnexpo2, wtExpo, mutExpo1,
                    mutExpo2, mutExpo3, mutExpo4)

## Create another categorical group to use here 
cdataframe$Group = paste(cdataframe$Genotype,cdataframe$Treatment, sep = '_')

attach(cdataframe)

head(cdataframe)

cdataframe |>
  filter()

barPlotting(cdataframe, x='Group', y='peak_z_stim')
