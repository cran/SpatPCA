## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  tidy = "styler"
)

## ----message=FALSE------------------------------------------------------------
library(SpatPCA)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gifski)
base_theme <- theme_classic(base_size = 18, base_family = "Times")

## -----------------------------------------------------------------------------
set.seed(1024)
position <- matrix(seq(-5, 5, length = 100))
true_eigen_fn <- exp(-position^2) / norm(exp(-position^2), "F")

data.frame(position = position,
           eigenfunction = true_eigen_fn) %>%
  ggplot(aes(position, eigenfunction)) +
  geom_line() +
  base_theme

## -----------------------------------------------------------------------------
realizations <- rnorm(n = 100, sd = 20) %*% t(true_eigen_fn) + matrix(rnorm(n = 100 * 100), 100, 100)

## ---- animation.hook="gifski"-------------------------------------------------
for (i in 1:100) {
  plot(x = position, y = realizations[i, ], ylim = c(-10, 10), ylab = "realization")
}

## -----------------------------------------------------------------------------
cv <- spatpca(x = position, Y = realizations)
eigen_est <- cv$eigenfn

## -----------------------------------------------------------------------------
data.frame(position = position, 
           true = true_eigen_fn, 
           spatpca = eigen_est[, 1], 
           pca = svd(realizations)$v[, 1]) %>%
  gather(estimate, eigenfunction, -position) %>%
  ggplot(aes(x = position, y = eigenfunction, color = estimate)) +
  geom_line() + 
  base_theme

## -----------------------------------------------------------------------------
realizations <- rnorm(n = 100, sd = 3) %*% t(true_eigen_fn) + matrix(rnorm(n = 100 * 100), 100, 100)

## ---- animation.hook="gifski"-------------------------------------------------
for (i in 1:100) {
  plot(x = position, y = realizations[i, ], ylim = c(-10, 10), ylab = "realization")
}

## -----------------------------------------------------------------------------
cv <- spatpca(x = position, Y = realizations)
eigen_est <- cv$eigenfn

data.frame(position = position, 
           true = true_eigen_fn, 
           spatpca = eigen_est[, 1], 
           pca = svd(realizations)$v[, 1]) %>%
  gather(estimate, eigenfunction, -position) %>%
  ggplot(aes(x = position, y = eigenfunction, color = estimate)) +
  geom_line() + 
  base_theme

