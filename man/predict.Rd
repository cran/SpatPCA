% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SpatPCA.R
\name{predict}
\alias{predict}
\title{Spatial predictions on new locations}
\usage{
predict(spatpca_object, x_new, eigen_patterns_on_new_site = NULL)
}
\arguments{
\item{spatpca_object}{An \code{spatpca} class object}

\item{x_new}{New location matrix.}

\item{eigen_patterns_on_new_site}{Eigen-patterns on x_new}
}
\value{
A prediction matrix of Y at the new locations, x_new.
}
\description{
Predict on new locations with the estimated spatial structures.
}
\examples{
# 1D: artificial irregular locations
x_1D <- as.matrix(seq(-5, 5, length = 10))
Phi_1D <- exp(-x_1D^2) / norm(exp(-x_1D^2), "F")
set.seed(1234)
Y_1D <- rnorm(n = 100, sd = 3) \%*\% t(Phi_1D) + matrix(rnorm(n = 100 * 10), 100, 10)
rm_loc <- sample(1:50, 20)
x_1Drm <- x_1D[-rm_loc]
Y_1Drm <- Y_1D[, -rm_loc]
x_1Dnew <- as.matrix(seq(-5, 5, length = 20))
cv_1D <- spatpca(x = x_1Drm, Y = Y_1Drm, tau2 = 1:100, num_cores = 2)
predictions <- predict(cv_1D, x_new = x_1Dnew)

}
\seealso{
\link{spatpca}
}
