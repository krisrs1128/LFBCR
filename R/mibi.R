
#' @importFrom SummarizedExperiment colData
#' @importFrom forcats fct_lump_n
#' @importFrom stringr str_extract
#' @export
load_mibi <- function(data_dir, n_paths = NULL, n_lev = 6) {
  exper <- get(load(file.path(data_dir, "mibiSCE.rda")))
  tiff_paths <- list.files(data_dir, "*.tiff", recursive = T, full.names = T)

  if (is.null(n_paths)) {
    n_paths <- length(tiff_paths)
  }

  # create a cell-type column
  colData(exper)$cell_type <- colData(exper)$tumor_group
  immune_ix <- colData(exper)$cell_type == "Immune"
  colData(exper)$cell_type[immune_ix] <- colData(exper)$immune_group[immune_ix]
  colData(exper)$cell_type <- fct_lump_n(colData(exper)$cell_type, n_lev)

  # subset to those samples with images
  tiff_paths <- tiff_paths[1:n_paths]
  sample_names <- str_extract(tiff_paths, "[0-9]+")
  list(
    tiffs = tiff_paths,
    mibi = exper[, colData(exper)$SampleID %in% sample_names],
    levels = table(colData(exper)$cell_type)
  )
}

#' @importFrom stringr str_extract
#' @importFrom raster crop raster
#' @importFrom purrr map2_dfr
#' @importFrom tidyr unite
#' @importFrom dplyr select pull
#' @importFrom SummarizedExperiment colData
#' @export
spatial_subsample <- function(tiff_paths, exper, qsize=500) {
  ims <- list()
  for (i in seq_along(tiff_paths)) {
    print(paste0("cropping ", i, "/", length(tiff_paths)))
    r <- raster(tiff_paths[[i]])
    ims[[tiff_paths[[i]]]] <- crop(r, extent(1, qsize, 1, qsize))
  }

  im_ids <- str_extract(tiff_paths, "[0-9]+")
  cur_cells <- map2_dfr(
    ims, im_ids,
    ~ data.frame(SampleID = .y, cellLabelInImage = unique(as.vector(.x)))
    ) %>%
    unite(sample_by_cell, SampleID, cellLabelInImage, remove = F)

  scell <- colData(exper) %>%
    as.data.frame() %>%
    dplyr::select(SampleID, cellLabelInImage) %>%
    unite(sample_by_cell, SampleID, cellLabelInImage) %>%
    pull("sample_by_cell")

  list(
    ims = ims,
    exper = exper[, scell %in% cur_cells$sample_by_cell]
  )
}

#' @importFrom dplyr select pull
#' @importFrom tidyr unite
#' @importFrom SummarizedExperiment colData
#' @export
subset_exper <- function(id, r, exper) {
  scell <- colData(exper) %>%
    as.data.frame() %>%
    dplyr::select(SampleID, cellLabelInImage) %>%
    unite(sample_by_cell, SampleID, cellLabelInImage) %>%
    pull("sample_by_cell")

  sample_by_cell <- data.frame(
    SampleID = id,
    cellLabelInImage = unique(as.vector(r))
  ) %>%
  unite(sample_by_cell, SampleID, cellLabelInImage, remove = F)

  exper[, scell %in% sample_by_cell$sample_by_cell]
}

#' @import raster
#' @importFrom dplyr filter pull %>%
#' @importFrom SummarizedExperiment colData
#' @export
unwrap_channels <- function(r, r_cells) {
  cell_types <- levels(r_cells$cell_type)
  r_mat <- array(0, dim = c(dim(r)[1:2], length(cell_types)))
  for (i in seq_along(cell_types)) {
    cur_cells <- colData(r_cells) %>%
      as.data.frame() %>%
      dplyr::filter(cell_type == cell_types[i]) %>%
      pull(cellLabelInImage) %>%
      unique()

    if (length(cur_cells) > 0) {
      r_mat[,, i] <- 1 * as.matrix(r %in% cur_cells)
    }
  }
  r_mat
}

#' @importFrom raster extent crop
#' @importFrom SummarizedExperiment colData assay
#' @export
extract_patch <- function(r, w, h, r_cells, qsize = 256, fct = 4, alpha = 5) {
  r <- crop(r, extent(w, w + qsize, h, h + qsize))
  r <- aggregate(r, fct, "modal")
  rm <- unwrap_channels(r, r_cells)

  cells_filter <- colData(r_cells) %>%
    as.data.frame() %>%
    filter(cellLabelInImage %in% unique(as.vector(r)))
  
  tumor_status <- cells_filter %>%
    pull(tumorYN)
  
  # log ratio tumor vs. immune (with laplace smoothing)
  xy <- data.frame(
    X.tumor = mean(tumor_status),
    X.size = mean(cells_filter$cellSize),
    X.grade = cells_filter$GRADE[1],
    y = log((1 + sum(tumor_status)) / (1 + sum(1 - tumor_status)), 2)
  )
  list(x = rm, xy = xy)
}

#' @importFrom raster raster extent crop
#' @importFrom SummarizedExperiment colData
#' @importFrom reticulate import
#' @importFrom stringr str_extract
#' @export
extract_patches <- function(tiff_paths, exper, qsize = 256, out_dir = ".", basename="patch") {
  np <- reticulate::import("numpy")
  im_ids <- str_extract(tiff_paths, "[0-9]+")
  y_path <- file.path(out_dir, "y.csv")

  for (i in seq_along(tiff_paths)) {
    r <- raster(tiff_paths[i])
    ix_start <- seq(0, ncol(r) - qsize/2 - 1, by = qsize/2)
    r_cells <- subset_exper(im_ids[i], r, exper)
    wh_pairs <- expand.grid(ix_start, ix_start)

    for (j in seq_len(nrow(wh_pairs))) {
      w <- as.integer(wh_pairs[j, 1])
      h <- as.integer(wh_pairs[j, 2])
      patch <- extract_patch(r, w, h, r_cells, qsize)

      # write results
      npy_path <- file.path(out_dir, sprintf("%s_%s-%s.npy", basename, w, h))
      np$save(npy_path, patch$x)
      xy <- cbind(data.frame(path = tiff_paths[i], i = basename, w = w, h = h), patch$xy)
      write.table(xy, y_path, sep = ",", col.names = !file.exists(y_path), append = T)
    }
  }
}