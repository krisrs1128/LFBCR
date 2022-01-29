#' @importFrom ggplot2 element_blank element_rect theme theme_minimal
#' @export
min_theme <- function() {
  theme_minimal() + 
    theme(
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "#f7f7f7"),
      panel.border = element_rect(fill = NA, color = "#0c0c0c", size = 0.6),
      axis.text = element_text(size = 14),
      axis.title = element_text(size = 16),
      legend.position = "bottom"
    )
}

#' Plot a Grid of Images
#'
#' @examples
#' paths <- list.files("~/Documents/stability_data/tiles/", full = TRUE)
#' coordinates <- matrix(rnorm(2 * length(paths)), length(paths), 2)
#' coordinates <- data.frame(coordinates)
#' colnames(coordinates) <- c("x", "y")
#' image_grid(coordinates, paths)
#'
#' @importFrom pdist pdist
#' @importFrom reticulate import
#' @importFrom ggplot2 ggplot geom_point aes coord_fixed theme_bw
#'   annotation_raster %+%
#' @export
image_grid <- function(coordinates, paths, density = c(15, 15), min_dist=0.1, imsize = 0.2) {
  if (length(density) == 1) {
    density <- c(density, density)
  }
  
  np <- import("numpy")
  p <- ggplot() +
    coord_fixed() +
    theme(legend.position = "none") +
    scale_fill_brewer(palette = "Set3")
  
  # get distances between anchor points and scores
  x_range <- c(min(coordinates$x), max(coordinates$x))
  y_range <- c(min(coordinates$y), max(coordinates$y))
  x_grid <- seq(x_range[1], x_range[2], length.out = density[1])
  y_grid <- seq(y_range[1], y_range[2], length.out = density[2])
  xy_grid <- expand.grid(x_grid, y_grid)
  dists <- as.matrix(pdist(xy_grid, as.matrix(coordinates)))
  
  # overlay the closest points
  used_ix <- c()
  for (i in seq_len(nrow(dists))) {
    min_ix <- which.min(dists[i, ])
    if (dists[i, min_ix] > min_dist) next
    if (min_ix %in% used_ix) next
    used_ix <- c(used_ix, min_ix)
    
    im <- np$load(paths[min_ix])
    mim <- melt(im) %>%
      filter(value != 0) %>%
      mutate(
        Var1 = xy_grid[i, 1] + (Var1) * imsize / 64,
        Var2 = xy_grid[i, 2] + (Var2) * imsize / 64
      )
    if (nrow(mim) == 0) next
    
    p <- p +
      geom_rect(
        data = mim,
        aes(
          xmin = min(Var1),
          xmax = max(Var1),
          ymin = min(Var2),
          ymax = max(Var2)
        ),
        fill = "#ededed"
      ) +
      geom_tile(
        data = mim,
        aes(x = Var1, y = Var2, fill = as.factor(Var3))
      )
  }
  
  p
}

#' @importFrom dplyr bind_rows group_by mutate row_number ungroup
#' @importFrom magrittr %>%
align_with_truth <- function(ud_hats, U, Sigma, tol = 0.01) {
  c(ud_hats, list(U %*% diag(Sigma))) %>%
    align_to_list(tol = tol, df = T) %>%
    bind_rows(.id = "b") %>%
    group_by(b) %>%
    mutate(i = row_number()) %>%
    ungroup() %>%
    mutate(b = as.integer(b))
}
  
#' @importFrom ggplot2 ggplot aes geom_hline geom_vline stat_ellipse geom_point
#'   scale_color_gradient2 scale_fill_gradient2 theme element_text
#' @export
plot_overlay_combined <- function(configuration, means) {
  ggplot(configuration, aes(X1, X2, col = y, fill = y)) +
      geom_hline(yintercept = 0, col = "#d3d3d3", size = 0.5) +
      geom_vline(xintercept = 0, col = "#d3d3d3", size = 0.5) +
      stat_ellipse(aes(group = i), geom = "polygon", level = 0.95, type = "norm", alpha = 0.5, size = 0.1) +
      geom_point(data = means, size = 0.2) +
      scale_color_gradient2(low = "#A6036D", high = "#03178C", mid = "#F7F7F7") +
      scale_fill_gradient2(low = "#A6036D", high = "#03178C", mid = "#F7F7F7") +
      theme(
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 10)
      )
}

#' @importFrom ggplot2 ggplot aes geom_hline geom_vline stat_ellipse geom_point
#'   scale_color_gradient2 scale_fill_gradient2 theme element_text geom_segment
#'   facet_wrap labs
#' @importFrom dplyr group_by summarise across starts_with rename left_join
#'   select filter
#' @importFrom magrittr %>%
#' @export
plot_overlay_truth <- function(ud_combined, max_i = 10, level = 0.95, alpha = 0.05) {
  ud_combined <- filter(ud_combined, i < max_i)
  ud_mean <- ud_combined %>%
    group_by(i, y, bootstrap) %>%
    summarise(across(starts_with("X"), mean))
  
  ud_truth <- filter(ud_combined, b == max(b))
  ud_truth_wide <- ud_truth %>%
    rename(X1_true = X1, X2_true = X2) %>%
    left_join(ud_mean %>% select(-i, -y))
  
  ggplot(ud_combined %>% filter(b < max(b)), aes(X1, X2, col = y, fill = y)) +
    stat_ellipse(aes(group = i), geom = "polygon", level = level, type = "norm", alpha = alpha, size = 0.1) +
    geom_point(data = ud_mean, size = 0.8) +
    geom_hline(yintercept = 0, col = "#d3d3d3", size = 0.5) +
    geom_vline(xintercept = 0, col = "#d3d3d3", size = 0.5) +
    scale_color_gradient2(low = "#A6036D", high = "#03178C", mid = "#F7F7F7") +
    scale_fill_gradient2(low = "#A6036D", high = "#03178C", mid = "#F7F7F7") +
    geom_point(data = ud_truth, size = 1, shape = 15) +
    geom_segment(data = ud_truth_wide, aes(xend = X1_true, yend = X2_true), size = 0.3, alpha = 0.8) +
    facet_wrap(~ bootstrap, scales = "free") +
    labs(x = "Aligned Scores 1", y = "Aligned Scores 2") +
    theme(
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12),
      strip.text = element_text(size = 14)
    )
}
