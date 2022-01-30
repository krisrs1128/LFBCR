
#' Untar Outputs for Visualization
#' @export
untar_all <- function(paths, exdir = "archive") {
  for (i in seq_along(paths)) {
    exdir_i <- file.path(exdir, i)
    if (!dir.exists(exdir_i)) {
      untar(paths[i], exdir = exdir_i)
    }
  }
}

#' @importFrom purrr map
#' @export
read_learned_features <- function(data_dir, layer_prefix, transform) {
  z_paths <- list.files(data_dir, layer_prefix, recursive = TRUE, full = TRUE)
  f <- ifelse(transform == "log", function(x) log(1 + x), identity)
  map(z_paths, ~ f(drop(np$load(.))))
}

#' @importFrom readr read_csv
#' @importFrom magrittr %>%
#' @importFrom dplyr left_join rename row_number
#' @export
read_split_indices <- function(data_dir) {
  Xy <- read_csv(list.files(data_dir, "Xy.csv", recursive = TRUE, full = TRUE)[1])
  list.files(data_dir, "*subset*", recursive = TRUE, full = TRUE) %>%
    .[[1]] %>%
    read_csv() %>%
    rename(ix = X1) %>%
    left_join(Xy) %>% 
    mutate(i = row_number())
}
