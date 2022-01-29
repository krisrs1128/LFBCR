
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

random_permutation <- function(K) {
  pi <- sample(K)
  pi_mat <- matrix(0, K, K)
  for (k in seq_len(K)) {
    pi_mat[k, pi[k]] <- 1
  }
  
  pi_mat
}

factor_terms <- function(N, D, K, feature_variances=1) {
  list(L = rmat(N, K, feature_variances), V = rmat(D, K))
}

sparse_factor_terms <- function(N, D, K, feature_variances = 1, 
                                delta_mass = 0.5) {
  terms <- factor_terms(N, D, K, feature_variances)
  n_entries <- length(terms$L)
  terms$L[sample(n_entries, n_entries * delta_mass)] <- 0
  terms
}

algorithmic_features <- function(N, D, K, sparse = TRUE, sigma = 1, 
                                 feature_variances = 1, delta_mass = 0.2) {
  if (sparse) {
    elem <- sparse_factor_terms(N, D, K, feature_variances, delta_mass)
  } else {
    elem <- factor_terms(N, D, K, feature_variances)
  }
  
  Z <- algorithmic_features_(elem$L, elem$V, sigma)
  list(L = elem$L, V = elem$V, Z = Z)
}

select_features <- function(L, y, q = 0.8) {
  C <- cor(L, y)
  which(abs(C) > quantile(abs(C), q))
}

algorithmic_features_ <- function(L, V, sigma = 1) {
  N <- nrow(L)
  K <- ncol(L)
  D <- nrow(V)
  
  Pi <- random_permutation(K)
  L %*% Pi %*% t(V %*% Pi) + rmat(N, D, sigma)
}

simulate_response <- function(L, S, sigma_beta = 1, sigma_y = 1) {
  beta <- vector(length = ncol(L))
  beta[sample(ncol(L), S)] <- rnorm(S, sd = sigma_beta)
  y <- L %*% beta + rnorm(nrow(L), sd = sigma_y)
  list(beta = beta, y = y)
}

svd_projector <- function(Z, K_hat = 5) {
  sv_z <- svd(Z)
  function (x_star) {
    k <- K_hat
    x_star %*% sv_z$v[, 1:k]
  }
}

sca_projector <- function(Z, K_hat = 5) {
  library(epca)
  sc_z <- sca(Z, k = K_hat)
  function (x_star) {
    x_star %*% sc_z$loadings
  }
}

supervised_svd_projector <- function(Z, y, K_hat = 5) {
  C <- cor(Z, y)
  keep_ix <- which(abs(C) > quantile(abs(C), 1 - 2 * K_hat / ncol(Z)))
  sv_z <- svd(Z[, keep_ix])
  
  function (x_star) {
    x_star[, keep_ix] %*% sv_z$v[, 1:K_hat]
  }
}

supervised_sca_projector <- function(Z, y, K_hat = 5) {
  library(epca)
  C <- cor(Z, y)
  keep_ix <- which(abs(C) > quantile(abs(C), 1 - 2 * K_hat / ncol(Z)))
  sc_z <- sca(Z[, keep_ix], k = K_hat)
  
  function (x_star) {
    x_star[, keep_ix] %*% sc_z$loadings
  }
}

zero_relatedness <- function(L, L_hat, beta, beta_hat) {
  scores <- abs(t(L_hat) %*% L) * beta_hat
  s0 <- which(beta == 0)
  weights <- scores[s0, ]
  list(score = rowSums(weights), weights = weights)
}

eta_relatedness <- function(L, L_hat, beta, beta_hat) {
  rowSums(t(L_hat) %*% L %*% diag(beta))
}

plot_pi_hat <- function(Pi_hats) {
  mPi_hats <- melt_stability(Pi_hats)
  list(
    ggplot(mPi_hats[[2]]) +
      geom_line(aes(lambda, value, group = b), size = 0.3, alpha = 0.1) +
      facet_wrap(~ j, scale = "free_y"),
    ggplot(mPi_hats[[1]]) +
      geom_line(aes(lambda, value, group = j), size = 0.2)
  )
}

aligned_stability_curves <- function(L_hats, y) {
  library(abind)
  
  B <- length(L_hats)
  L_aligned <- procrustes(L_hats)
  Pi_hats_ <- vector(length = B, mode = "list")

  for (b in seq_len(B)) {
    Pi_hats_[[b]] <- stability_selection(
        L_aligned$x_align[,, b], 
        y, 1, fit$lambda
      )[[2]]
  }
  
  Pi_hats_ <- abind(Pi_hats_)
  list(
    Pi = apply(abs(Pi_hats_) > 0, c(1, 2), mean),
    coef_paths = Pi_hats_
  )
}

r_ortho <- function(N, K) {
  Z <- rmat(N, K)
  qr.Q(qr(Z))
}