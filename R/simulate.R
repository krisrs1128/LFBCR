
#' @export
rmat <- function(N, D, sigma2=1) {
  if (length(sigma2) == 1) {
    sigma2 <- rep(sigma2, D)
  }
  
  result <- matrix(rnorm(N * D), N, D)
  for (d in seq_len(D)) {
    result[, d] <- sqrt(sigma2[d]) * result[, d]
  }
  
  result
}

#' @export
r_ortho <- function(N, K) {
  Z <- rmat(N, K)
  qr.Q(qr(Z))
}

#' @export
random_permutation <- function(K) {
  pi <- sample(K)
  pi_mat <- matrix(0, K, K)
  for (k in seq_len(K)) {
    pi_mat[k, pi[k]] <- 1
  }
  
  pi_mat
}
