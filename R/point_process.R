#' Functions for Simulating Image Parameters

#' Simulate from a Matern Process
#'
#' @importFrom MASS mvrnorm
#' @examples
#' x <- expand.grid(seq(0.1, 1, 0.05), seq(0.1, 1, 0.05))
#' process_df <- matern_process(x, 1, 1)
#' ggplot(process_df) +
#'  geom_tile(aes(x = Var1, y = Var2, fill = z))
#' @export
matern_process <- function(x, nu=1, alpha=1) {
  Sigma <- matern_kernel(x, nu, alpha)
  z <- mvrnorm(1, mu=rep(0, nrow(x)), Sigma)
  data.frame(x, z)
}

#' Build Matern Kernel
#'
#' @importFrom dplyr %>%
#' @export
matern_kernel <- function(x, nu, alpha) {
  squared_dist <- dist(x) %>%
    as.matrix()
  squared_dist <- squared_dist / alpha

  Sigma <- (1/((2 ^ (nu - 1)) * gamma(nu))) *
    (squared_dist ^ nu) *
    besselK(squared_dist, nu)
  diag(Sigma) <- 1
  Sigma
}

#' Simulate Matern Probabilities
#'
#' @examples
#' library("reshape2")
#' x <- expand.grid(seq(0.1, 1, 0.05), seq(0.1, 1, 0.05))
#' probs <- relative_intensities(x, nu = 1)
#' @details #' See page 551 in SPATIAL AND SPATIO-TEMPORAL LOG-GAUSSIAN COX
#'   PROCESSES
#' @export
relative_intensities <- function(x, K = 4, betas = NULL, ...) {
  if (is.null(betas)) {
    betas <- rnorm(K, 0, 0.5)
  } else {
    K <- length(betas)
  }

  processes <- matrix(0, nrow(x), K)
  for (k in seq_len(K)) {
    processes[, k] <- matern_process(x, ...)[, 3]
  }

  lambdas <- matrix(0, nrow(x), K)
  betas_mat <- t(replicate(nrow(x), betas))
  for (k in seq_len(K)) {
    lambdas[, k] <- exp(betas_mat[, k] + processes[, k])
  }

  probs <- t(apply(lambdas, 1, function(r) r / sum(r)))
  data.frame(x, probs)
}

#' Simulate an Inhomogeneous Poisson Process
#'
#' We just thin out an ordinary poisson process.
#' @export
inhomogeneous_process <- function(N0, intensity) {
  z <- matrix(runif(2 * N0), N0, 2)
  x <- as.matrix(intensity[, 1:2])

  sample_probs <- vector(length = N0)
  max_z <- max(intensity$z)
  ixs <- vector(length = N0)

  for (i in seq_len(nrow(z))) {
    diffs <- x - t(replicate(nrow(x), z[i, ]))
    ixs[i] <- which.min(rowSums(diffs ^ 2))
    sample_probs[i] <- exp(intensity$z[ixs[i]] - max_z)
  }

  keep_ix <- which(rbinom(N0, 1, prob=sample_probs) == 1)
  z[keep_ix, ]
}

#' Mark an Inhomogeneous Poisson Process
#' @export
mark_process <- function(z, probs, tau=1, lambdas=NULL) {
  N <- nrow(z)
  marks <- vector(length = N)
  sizes <- vector(length = N)

  x <- as.matrix(probs[, 1:2])
  K <- ncol(probs) - 2

  # parameters of gamma size distn
  if (is.null(lambdas)) {
    lambdas <- seq(50, 150, length.out = K)
  }

  # simulate each cell
  for (i in seq_len(N)) {
    diffs <- x - t(replicate(nrow(x), z[i, ]))
    ix <- which.min(rowSums(diffs ^ 2))
    p <- unlist(probs[ix, 3:ncol(probs)])
    p <- p ^ tau / sum(p ^ tau)
    marks[i] <- sample(seq_len(K), 1, prob = p)
    sizes[i] <- rgamma(1, shape = 5, rate = lambdas[marks[i]])
  }

  data.frame(z, mark = as.factor(marks), size = sizes)
}

#' Helper to Wrap Simulation
#' @export
sim_wrapper <- function(x, n_original, nu, alpha, beta_r, nu_r, alpha_r, tau, lambdas) {
  intensity <- matern_process(x, nu, alpha)
  z <- inhomogeneous_process(n_original, intensity)
  probs <- relative_intensities(x, betas = beta_r, nu = nu_r, alpha = alpha_r)
  marks <- mark_process(z, probs, tau, lambdas)
  list(intensity = intensity, z = z, probs = probs, marks = marks)
}

#' Build a Spatial Data Frame
#'
#' @importFrom dplyr %>%
#' @importFrom sf st_point st_buffer st_as_sf st_geometrycollection st_sfc
#' @export
spatial_df <- function(x) {
  pts <- st_sfc(lapply(seq_along(x), function(x) st_geometrycollection()))

  for (i in seq_len(nrow(x))) {
    pt <- as.numeric(x[i, c("X1", "X2")])
    pts[i] <- st_point(pt) %>%
      st_buffer(x[i, "size"])
  }
  st_as_sf(pts)
}

#' @importFrom dplyr %>%
#' @importFrom sf st_dimension
#' @importFrom raster raster rasterize stack
#' @export
make_raster <- function(marks, n_channels=3) {
  pts <- vector("list", n_channels)
  for (i in seq_along(pts)) {
    pts[[i]] <- marks %>%
      filter(mark == levels(marks$mark)[i]) %>%
      spatial_df
  }

  r <- list()
  for (i in seq_along(pts)) {
    if (any(is.na(st_dimension(pts[[i]])))) {
      r[[i]] <- raster(ncols=64, nrows=64, ext=extent(c(0, 1, 0, 1)))
      values(r[[i]]) <- 0
    } else {
      r[[i]] <- raster(pts[[i]], ncols=64, nrows=64, ext=extent(c(0, 1, 0, 1)))
      r[[i]] <- rasterize(pts[[i]], r[[i]], field = 1, background = 0)
    }
  }
  
  stack(r)
}

#' @importFrom RStoolbox ggRGB
#' @importFrom ggplot2 %+% scale_x_continuous
#'   scale_y_continuous theme element_blank
#' @export
plot_raster <- function(r) {
  ggRGB(r) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank()
    )
}

#' @importFrom ggplot2 ggplot geom_point aes coord_fixed %+% scale_x_continuous
#'   scale_y_continuous labs geom_tile facet_wrap scale_size theme element_blank
#' @importFrom reshape2 melt
#' @importFrom dplyr %>% mutate filter
#' @export
plot_matern <- function(result) {
  marks <- result$marks %>%
    mutate(variable = paste0("X", mark)) %>%
    filter(X1 > 0.05, X2 > 0.05)
  
  mprobs <- melt(result$probs, id.vars = c("Var1", "Var2"))
  mprobs$variable <- sapply(mprobs$variable, function(x) { gsub("X", "Process ", x) })
  marks$variable <- sapply(marks$variable, function(x) { gsub("X", "Process ", x) })
  
  ggplot() +
    geom_tile(aes(x = Var1, y = Var2, fill=value), mprobs) +
    geom_point(aes(x = X1, y = X2, col = mark, size=size), marks) +
    facet_wrap(~ variable) +
    coord_fixed() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_color_manual(values = c("red", "green", "blue")) +
    scale_fill_continuous(low = "white", high = "black") +
    labs(x = "", y = "") +
    scale_size(range = c(0.05, 2)) +
    theme(
      legend.position = "none",
      axis.text = element_blank(), 
      axis.ticks = element_blank()
    )
}