% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vis.R
\name{image_grid}
\alias{image_grid}
\title{Plot a Grid of Images}
\usage{
image_grid(
  coordinates,
  paths,
  density = c(15, 15),
  min_dist = 0.1,
  imsize = 0.2
)
}
\description{
Plot a Grid of Images
}
\examples{
paths <- list.files("~/Documents/stability_data/tiles/", full = TRUE)
coordinates <- matrix(rnorm(2 * length(paths)), length(paths), 2)
coordinates <- data.frame(coordinates)
colnames(coordinates) <- c("x", "y")
image_grid(coordinates, paths)

}
