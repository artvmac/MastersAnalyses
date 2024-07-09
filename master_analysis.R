######################################
# Analysis for Machado et al. (2020) #
######################################

# load file

df <- read.csv(file = "S1_File.csv")

# install and load packages

# you will need to install rjags before
# installing bayesian first aid
# https://sourceforge.net/projects/mcmc-jags/

# install.packages("tidyverse")
# install.packages("lsr")
# install.packages("moments")
# install.packages("devtools")
# devtools::install_github("rasmusab/bayesian_first_aid")
# install.packages("caret")
# install.packages("BayesFactor")
# install.packages("bayestestR")
# install.packages("insight")
# install.packages("see")
# install.packages("rcpp")
# install.packages("rstanarm")

library(tidyverse)
library(lsr)
library(moments)
library(devtools)
library(BayesianFirstAid)
library(caret)
library(BayesFactor)
library(bayestestR)
library(insight)
library(see)
library(rcpp)
library(rstanarm)

# set seed for reprodutibility

set.seed(123)

# observing the assumption of normality

kurtosis(df$pcl_c)
skewness(df$pcl_c)
qqnorm(df$pcl_c)
qqline(df$pcl_c)

kurtosis(df$dysfunctional)
skewness(df$dysfunctional)
qqnorm(df$dysfunctional)
qqline(df$dysfunctional)

kurtosis(df$emotion)
skewness(df$emotion)
qqnorm(df$emotion)
qqline(df$emotion)

kurtosis(df$problem)
skewness(df$problem)
qqnorm(df$problem)
qqline(df$problem)

# main correlations plots

plot(df$problem, df$pcl_c)
abline(lm(pcl_c ~ problem, data = df), col="blue")

plot(df$emotion, df$pcl_c)
abline(lm(pcl_c ~ emotion, data = df), col="blue")

plot(df$dysfunctional, df$pcl_c)
abline(lm(pcl_c ~ dysfunctional, data = df), col="blue")

# main correlations tests

cor.test(df$problem, df$pcl_c, method = "spearman")

cor.test(df$emotion, df$pcl_c, method = "spearman")

cor.test(df$dysfunctional, df$pcl_c)

# coping and criteria correlations

cor_df_coping  <- select(df, problem:dysfunctional)

cor_df_criteria  <- select(df, b_criterion:d_criterion)

correlate(cor_df_coping, cor_df_criteria, test = TRUE, p.adjust.method = "none")

# dysfunctional coping subscales and pcl_c correlations

cor_subscales <- select(df, sb,vt,dn:su)

correlate(cor_subscales, df$pcl_c, test = TRUE, p.adjust.method = "none")

# emotion coping subscales and pcl_c correlations

cor_subscales2 <- select(df, es:pr,at,hu)

correlate(cor_subscales2, df$pcl_c, test = TRUE, p.adjust.method = "none")

# problem coping subscales and pcl_c correlations

cor_subscales3 <- select(df, ac,pl, is)

correlate(cor_subscales3, df$pcl_c, test = TRUE, p.adjust.method = "none")

# bayesian correlations

# main correlations

c1 <- bayes.cor.test(df$problem, df$pcl_c)
c1

plot(c1)

c2 <- bayes.cor.test(df$emotion, df$pcl_c)
c2

plot(c2)

c3 <- bayes.cor.test(df$dysfunctional, df$pcl_c)
c3

plot(c3)

# coping and criteria correlations

c4 <- bayes.cor.test(df$emotion, df$b_criterion)
c4

c5 <- bayes.cor.test(df$emotion, df$c_criterion)
c5

c6 <- bayes.cor.test(df$emotion, df$d_criterion)
c6

c7 <- bayes.cor.test(df$dysfunctional, df$b_criterion)
c7

c8 <- bayes.cor.test(df$dysfunctional, df$c_criterion)
c8

c9 <- bayes.cor.test(df$dysfunctional, df$d_criterion)
c9

c10 <- bayes.cor.test(df$problem, df$b_criterion)
c10

c11 <- bayes.cor.test(df$problem, df$c_criterion)
c11

c12 <- bayes.cor.test(df$problem, df$d_criterion)
c12

# pcl and dysfunctional coping subscales correlations

c13 <- bayes.cor.test(df$sb, df$pcl_c)
c13

c14 <- bayes.cor.test(df$vt, df$pcl_c)
c14

c15 <- bayes.cor.test(df$dn, df$pcl_c)
c15

c16 <- bayes.cor.test(df$sd, df$pcl_c)
c16

c17 <- bayes.cor.test(df$bd, df$pcl_c)
c17

c18 <- bayes.cor.test(df$su, df$pcl_c)
c18

# save main correlation plots in high resolution

tiff("S1A_Fig.tiff", width = 5, height = 4, units = 'in', res = 400)
plot(c3)
dev.off()

tiff("S1B_Fig.tiff", width = 5, height = 4, units = 'in', res = 400)
plot(c2)
dev.off()

tiff("S1C_Fig.tiff", width = 5, height = 4, units = 'in', res = 400)
plot(c1)
dev.off()

# regression analyses

# set gender as factor

df$gender_coded <- as.factor(df$gender_coded)

#creating frequentist model with the confounding variable gender

m1 <- lm(pcl_c ~ gender_coded + dysfunctional +
           emotion + problem, data = df)

summary(m1)

# diagnostics plots

# check for multicollinearity

car::vif(m1)

# check linearity of the data

plot(m1, 1) # line should be close to the horinzontal line

# check homogeneity of variance

plot(m1, 3) # horinzontal line and spread and equal points are the ideal

# normality of residuals

plot(m1, 2) # plot of residuals should follow the straight qq line

# outliers and leverage

plot(m1, 5) # ideally should not exceed 3 sd (vertical axis)


# creating bayesian model with the confounding variable gender

bm1 <- lmBF(pcl_c ~ gender_coded + dysfunctional +
              emotion + problem, data = df)

bm1 # extreme evidence in favour of h1 over h0


# we are not interested in model comparison so we create
# mcmc chain to evaluate the model parameters estimations

chains <- posterior(bm1, iterations = 10000)

summary(chains[,1:6])

plot(chains[,1:6]) # only dysfunctional 95% CI is outside of 0

plot(chains[, 4]) #plot dysfunctional coping

# building glm model in stan to build beautiful plots

bm2 <- stan_glm(pcl_c ~ gender_coded + dysfunctional + 
                emotion + problem, data = df, iter = 12000, 
                chains = 1, warmup = 2000)

prob_direction <- p_direction(bm2, 
                              parameters = c("dysfunctional",
                                             "emotion",
                                             "problem"))

plot(prob_direction) + scale_fill_brewer() + theme_modern()

# save bayesian regression plot in high resolution

tiff("S2_Fig.tiff", width = 6, height = 4, units = 'in', res = 400)
plot(prob_direction) + scale_fill_brewer() + theme_modern()
dev.off()
