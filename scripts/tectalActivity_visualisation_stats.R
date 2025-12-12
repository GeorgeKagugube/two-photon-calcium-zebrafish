###############################################################################
# This script analyses GCaMP7s activity extracted Python automated scripts ####
# Created by George W Kagugube                                             ####
# Date:                                                                    ####
###############################################################################
rm(list = ls()); gc()

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

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
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
################################################################################
# # Read in the Unexposed Mn data here (This is all the tonic data)
# df_wtUnexp1 <- genotype_group(df = read.csv("./wt/excitability_metrics_by_class_wt_exposed1.csv"),
#                               genotype = 'WT', treatment = 'Unexposed')
# df_wtunexpo2 <- genotype_group(df = read.csv('./wt/excitability_metrics_by_class_wt_exposed2.csv'),
#                                genotype = 'WT', treatment = 'Unexposed')
# 
# ## Load the Exposed Mn data
# df_wtexpo <- genotype_group(df = read.csv('./wt_exposed/excitability_metrics_by_class_exposed_wt.csv'),
#                             genotype = 'WT', treatment = 'Exposed')
# 
# ## Load the Mutant exposed data here
# df_mut1 <- genotype_group(df = read.csv('./mutant_exposed/excitability_metrics_by_class_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut2 <- genotype_group(df = read.csv('./mutant_exposed/excitability_metrics_by_class_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut3 <- genotype_group(df = read.csv('./mutant_exposed/excitability_metrics_by_class_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut4 <- genotype_group(df = read.csv('./mutant_exposed/excitability_metrics_by_class_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
###############################################################################
## Read in the Unexposed Mn data here (This is all the tonic data)
# df_wtUnexp1 <- genotype_group(df = read.csv("./wt/tonic_metrics_wt_exposed1.csv"),
#                               genotype = 'WT', treatment = 'Unexposed')
# df_wtunexpo2 <- genotype_group(df = read.csv('./wt/tonic_metrics_wt_exposed2.csv'),
#                                genotype = 'WT', treatment = 'Unexposed')
# 
# ## Load the Exposed Mn data
# df_wtexpo <- genotype_group(df = read.csv('./wt_exposed/tonic_metrics_exposed_wt.csv'),
#                             genotype = 'WT', treatment = 'Exposed')
# 
# ## Load the Mutant exposed data here
# df_mut1 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut2 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut2.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut3 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut3.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut4 <- genotype_group(df = read.csv('./mutant_exposed/tonic_metrics_exposed_mut4.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# 
# df <- read.csv("../extractedROI/loomROI/unexposed_wt.csv", header = T)
################################################################################
## Perform data exploration here 
head(df_wtUnexp1)
head(df_mut1)

# # # Read in the Unexposed Mn data here (This is all the tonic data)
df_wtUnexp1 <- genotype_group(df = read.csv("./wt/clearance_metrics_by_class_wt_exposed1.csv"),
                              genotype = 'WT', treatment = 'Unexposed')
df_wtunexpo2 <- genotype_group(df = read.csv('./wt/clearance_metrics_by_class_wt_exposed2.csv'),
                               genotype = 'WT', treatment = 'Unexposed')

## Load the Exposed Mn data
df_wtexpo <- genotype_group(df = read.csv('./wt_exposed/clearance_metrics_by_class_exposed_wt.csv'),
                            genotype = 'WT', treatment = 'Exposed')

## Load the Mutant exposed data here
df_mut1 <- genotype_group(df = read.csv('./mutant_exposed/clearance_metrics_by_class_exposed_mut1.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut2 <- genotype_group(df = read.csv('./mutant_exposed/clearance_metrics_by_class_exposed_mut2.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut3 <- genotype_group(df = read.csv('./mutant_exposed/clearance_metrics_by_class_exposed_mut3.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')
df_mut4 <- genotype_group(df = read.csv('./mutant_exposed/clearance_metrics_by_class_exposed_mut4.csv'),
                          genotype = 'Mutant', treatment = 'Exposed')

################################################################################
# # Read in the Unexposed Mn data here (This is all the tonic data)
# df_wtUnexp1 <- genotype_group(df = read.csv("./wt/event_metrics_by_class_wt_exposed1.csv"),
#                               genotype = 'WT', treatment = 'Unexposed')
# df_wtunexpo2 <- genotype_group(df = read.csv('./wt/event_metrics_by_class_wt_exposed2.csv'),
#                                genotype = 'WT', treatment = 'Unexposed')
# 
# ## Load the Exposed Mn data
# df_wtexpo <- genotype_group(df = read.csv('./wt_exposed/event_metrics_by_class_exposed_wt.csv'),
#                             genotype = 'WT', treatment = 'Exposed')
# 
# ## Load the Mutant exposed data here
# df_mut1 <- genotype_group(df = read.csv('./mutant_exposed/event_metrics_by_class_exposed_mut1.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut2 <- genotype_group(df = read.csv('./mutant_exposed/event_metrics_by_class_exposed_mut2.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut3 <- genotype_group(df = read.csv('./mutant_exposed/event_metrics_by_class_exposed_mut3.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')
# df_mut4 <- genotype_group(df = read.csv('./mutant_exposed/event_metrics_by_class_exposed_mut4.csv'),
#                           genotype = 'Mutant', treatment = 'Exposed')


cdataframe <- rbind(df_wtUnexp1, df_wtunexpo2, df_wtexpo, df_mut1, df_mut2, df_mut3, df_mut4)
attach(cdataframe)

## Create another categorical group to use here 
cdataframe$Group = paste(cdataframe$Genotype,cdataframe$Treatment, sep = '_')

## Order the catergorical vairables for better visualisations hereafter
cdataframe$Group <- ordered(cdataframe$Group,
                            levels = c('WT_Unexposed', 'WT_Exposed', 'Mutant_Exposed'))
cdataframe$Genotype <- ordered(cdataframe$Genotype, 
                               level = c('WT', 'Mutant'))
cdataframe$Treatment <- ordered(cdataframe$Treatment,
                                levels = c('Unexposed', 'Exposed'))
cdataframe$class <- ordered(cdataframe$class,
                            levels = c('pre', 'during', 'post'))

## Export the processed clean data frame for eas of analysis next time 
getwd()
write.csv(cdataframe, 
          file = 'cleanedData_excitability_metric.csv')

## Rearange the categorical variables here for better visualisation
attach(cdataframe)

df <- cdataframe |>
  filter(class == 'during' & responsive == 'True') 
  
# df |>
#   group_by(Group) |>
#   summarise(avg = mean(mean_tail_auc_s, na.rm = T))

# # Remove outliers from the dataset here, this is for the tonic dataset only
# df <- df |>
#   filter(mean_tail_auc_s > 0 & mean_tail_auc_s < 15)
df <- na.omit(df[((df < 10) & df > 0),])
# df <- df[df$mean_tail_auc_s < 15,]
######################## Bargraph ##########################
barPlotting(df, x = 'Group', y = 'mean_tail_auc_s')

######################### Statistics ##############################
# stats <- df |>
#   filter(class == 'pre')
res.aov <- aov(mean_downstroke_slope ~ Group, data = df)
# Summary of the analysis
summary(res.aov)

TukeyHSD(res.aov)
########################### Line graph ##########################
 df |>
   #filter(Group == 'WT_Unexposed') |>
 ggline(x = "class", y = "mean_IEI_s", 
        add = c("mean_se"),
        color = "Group", palette = "jco") +
   stat_compare_means(comparisons = list(c('WT_Unexposed', 'WT_Exposed'), 
                                         c('WT_Unexposed', 'Mutant_Exposed'),
                                         c('WT_Exposed', 'Mutant_Exposed')), 
                      label = "p.signif")

#################################################################
library(reshape2)
melted_cormat <- melt(cormat)
head(melted_cormat)

library(ggplot2)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)


ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))


