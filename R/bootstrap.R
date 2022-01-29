
#' @export
features <- function(X, K = 50, sigma_e = 0.1) {
  N <- nrow(X)
  D <- nrow(X)
  svx <- svd(X)
  
  function(X) {
    Pi <- random_permutation(K)
    (X %*% svx$v[, 1:K]  + rmat(nrow(X), K, sigma_e)) %*% Pi
  }
}

#' @export
param_boot <- function(Z, K = 2) {
  svz <- svd(Z)
  u_hat <- svz$u[, 1:K]
  d_hat <- svz$d[1:K]
  v_hat <- svz$v[, 1:K]
  E <- Z - u_hat %*% diag(d_hat) %*% t(v_hat)

  function() {
    param_boot_(u_hat, d_hat, E)
  }
}

#' Zb here is like the Lb used in the writeup
#' @export
param_boot_ <- function(u_hat, d_hat, E) {
  eB <- matrix(sample(E, nrow(E) * length(d_hat), replace = TRUE), nrow(E), length(d_hat))
  Pi <- random_permutation(length(d_hat))
  Zb <- (u_hat %*% diag(d_hat) + eB) %*% Pi
  list(Zb = Zb, ub = svd(Zb)$u %*% diag(svd(Zb)$d))
}

#' Zb here is like the Lb used in the writeup
#' @importFrom irlba irlba
#' @importFrom purrr map map2
#' @export
param_boot_cmp <- function(Zb, K = 2) {
  svz <- map(Zb, ~ irlba(., nv=K))
  Eb <- map2(Zb, svz, ~ .x - .y$u %*% diag(.y$d) %*% t(.y$v))
  ud_hats <- map(svz, ~ .$u %*% diag(.$d)) %>%
    procrustes()
  
  function() {
    param_boot_cmp_(ud_hats$M, Eb)
  }
}
  
#' @export
param_boot_cmp_ <- function(M, Eb) {
  Estar <- do.call(cbind, Eb)
  K <- ncol(M)
  Estar <- matrix(sample(Estar, nrow(Estar) * K, replace = TRUE), nrow(Estar), K)
  Pi <- random_permutation(K)
  Zb <- (M + Estar) %*% Pi
  svz <- svd(Zb)
  list(Zb = Zb, ub = svz$u %*% diag(svz$d))
}

#' @importFrom purrr map
#' @export
arr_to_list <- function(x, df = F) {
  if (df) {
    res <- map(1:dim(x)[3], ~ data.frame(x[,, .]))
  } else {
    res <- map(1:dim(x)[3], ~ x[,, .])
  }
  
  res
}

#' @export
procrustes <- function(x_list, tol = 0.001, max_iter=100) {
  x_align <- array(dim = c(dim(x_list[[1]]), length(x_list)))
  M <- x_list[[1]]

  iter <- 1
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
    if (coord_change < tol | iter > max_iter) break
    iter <- iter + 1
    message(iter)
  }

  list(x_align = x_align, M = M)
}

#' @importFrom magrittr %>%
#' @export
align_to_list <- function(Zb, df = F, tol = 0.01) {
  procrustes(Zb, tol = tol) %>%
    .[["x_align"]] %>%
    arr_to_list(df = df)
}

#' @importFrom expm sqrtm
#' @export
within_ellipse <- function(x, new_point, level = 0.95) {
  params <- cov.wt(x)
  new_point_ <- solve(sqrtm(params$cov)) %*% (new_point - params$center)
  sum(new_point_ ^ 2) < sqrt(2 * qf(level, 2, 2))
}

#' @importFrom dplyr %>% select
#' @importFrom purrr map map2_dbl
#' @export
coverage <- function(samples, centers, level = 0.95) {
  samples <- samples %>%
    split(.$i) %>%
    map(~ as.matrix(select(., X1, X2)))
  
  centers <- centers %>%
    split(.$i) %>%
    map(~ unlist(select(., X1, X2)))
    
  map2_dbl(samples, centers, ~ within_ellipse(.x, .y, level = level))
}
