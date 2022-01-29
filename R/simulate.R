
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

#' @importFrom magrittr %>%
#' @importFrom dplyr bind_rows group_by ungroup mutate row_number
#' @export
align_with_truth <- function(ud_hats, U, Sigma, tol = 0.01) {
  c(ud_hats, list(U %*% diag(Sigma))) %>%
    align_to_list(tol = tol, df = T) %>%
    bind_rows(.id = "b") %>%
    group_by(b) %>%
    mutate(i = row_number()) %>%
    ungroup() %>%
    mutate(b = as.integer(b))
}