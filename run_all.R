# run_all.R — Execute all project stages sequentially
# Set your working directory to the project root first:
#   setwd("/path/to/monte_carlo_finance")

cat("╔══════════════════════════════════════════════╗\n")
cat("║   Monte Carlo Finance Engine — Full Run      ║\n")
cat("╚══════════════════════════════════════════════╝\n\n")

stages <- list(
  list(file = "R/01_infrastructure.R",     label = "Stage 1: Infrastructure"),
  list(file = "R/02_stochastic_processes.R", label = "Stage 2: Stochastic Processes"),
  list(file = "R/03_mc_pricing.R",          label = "Stage 3: MC Option Pricing"),
  list(file = "R/04_variance_reduction.R",  label = "Stage 4: Variance Reduction"),
  list(file = "R/05_parallel.R",            label = "Stage 5: Parallel Computing"),
  list(file = "R/06_exotic_options.R",      label = "Stage 6: Exotic Options"),
  list(file = "R/07_heston.R",              label = "Stage 7: Heston Model")
)

for (s in stages) {
  cat(sprintf("\n▶  %s\n", s$label))
  cat(paste(rep("─", 50), collapse = ""), "\n")
  t0 <- proc.time()
  source(s$file, echo = FALSE)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("✓  Done in %.1fs\n", elapsed))
}

cat("\n\n✅ All stages complete.\n")
