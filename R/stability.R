
#' @export
subset_matrices <- function(rdata, cols = as.character(1:40)) {
  X <- rdata %>%
    select(cols) %>%
    as.matrix()

  y <- rdata %>%
    .[["y"]]

  list(X = X, y = y)
}

#' @importFrom glmnet glmnet
#' @export
stability_selection <- function(X, y, B = 1000, lambda) {
  n <- nrow(X)
  p <- ncol(X)
  coef_paths <- array(dim = c(p + 1, length(lambda), B))
  for (b in seq_len(B)) {
    ix <- sample(seq_len(n), n / 2, replace = FALSE)
    fit <- glmnet(X[ix, ], y[ix], lambda = lambda)
    coef_paths[, , b] <- as.matrix(coef(fit))
  }

  Pi <- apply(coef_paths, c(1, 2), function(z) mean(abs(z) > 0))
  list(Pi = Pi, coef_paths = coef_paths)
}

#' @importFrom reshape2 melt
#' @export
melt_stability <- function(res) {
  mpi <- melt(res$Pi, varnames = c("j", "lambda"))
  mcoef_paths <- melt(res$coef_paths, varnames = c("j", "lambda", "b"))
  list(Pi = mpi, coef_paths = mcoef_paths)
}

#' @export
untar_all <- function(paths, data_dir = ".") {
  for (i in seq_along(paths)) {
    exdir <- file.path(data_dir, tools::file_path_sans_ext(basename(paths[[i]])))
    if (!dir.exists(exdir)) {
      untar(paths[i], exdir = exdir)
    }
  }
}

#' @importFrom reticulate import
#' @export
read_npys <- function(paths, data_dir = ".") {
  results <- list()
  np <- import("numpy")
  for (f in paths) {
    results[[f]] <- np$load(f)
  }
  results
}

#' @export
procrustes <- function(x_list, tol = 0.05) {
  x_align <- array(dim = c(dim(x_list[[1]]), length(x_list)))
  M <- x_list[[1]]

  while (TRUE) {
    # solve each problem
    for (i in seq_along(x_list)) {
      svd_i <- svd(t(x_list[[i]]) %*% M)
      beta <- sum(svd_i$d) / sum(x_list[[i]] ^ 2)
      x_align[,, i] <- beta * x_list[[i]] %*% svd_i$u %*% t(svd_i$v)
    }

    # new procrustes mean
    M_old <- M
    M <- apply(x_align, c(1, 2), mean)
    coord_change <- mean(abs(M - M_old))

    #print(coord_change)
    if (coord_change < tol) break
  }

  list(x_align = x_align, M = M)
}

