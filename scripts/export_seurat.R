library(Seurat)
library(Matrix)

export_seurat <- function(rds_path, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  message("Reading RDS: ", rds_path)
  data <- readRDS(rds_path)
  
  # Extract counts
  message("Extracting counts...")
  # Handle Seurat v5 vs v4
  if (as.numeric(substr(packageVersion("Seurat"), 1, 1)) >= 5) {
    counts <- GetAssayData(data, assay = "RNA", layer = "counts")
  } else {
    counts <- GetAssayData(data, assay = "RNA", slot = "counts")
  }
  
  # Write MTX
  message("Writing MTX...")
  writeMM(counts, file.path(out_dir, "matrix.mtx"))
  
  # Write genes/barcodes
  message("Writing metadata...")
  write.table(rownames(counts), file.path(out_dir, "genes.tsv"), 
              col.names = FALSE, row.names = FALSE, quote = FALSE)
  write.table(colnames(counts), file.path(out_dir, "barcodes.tsv"), 
              col.names = FALSE, row.names = FALSE, quote = FALSE)
  
  # Write obs (meta.data)
  write.csv(data@meta.data, file.path(out_dir, "metadata.csv"), quote = TRUE)
  
  message("Done! Files in ", out_dir)
}

# Example usage
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) {
  export_seurat(args[1], args[2])
} else {
  message("Usage: Rscript export_seurat.R <input.rds> <output_dir>")
}
