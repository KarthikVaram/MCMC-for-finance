# Stage 5: Parallel Computing

library(Rcpp)
library(ggplot2)
library(microbenchmark)

# Note: OpenMP must be available. On macOS you may need to install libomp.
# On Ubuntu: sudo apt-get install libomp-dev

sourceCpp("src/parallel_mc.cpp")
sourceCpp("src/mc_pricer.cpp")

# ── Parameters ───────────────────────────────────────────────────────────────
S0 <- 100; K <- 105; r <- 0.05; sigma <- 0.20; T <- 1.0
N  <- 1000000; steps <- 252; seed <- 42

# ── Parallel pricing ─────────────────────────────────────────────────────────
cat("\n=== Parallel MC Call Pricing (1M paths) ===\n")
par_res <- mc_parallel_call(S0, K, r, sigma, T, N, steps, seed, n_threads = 0)
cat(sprintf("Price:          %.4f\n", par_res$price))
cat(sprintf("Std Error:      %.6f\n", par_res$std_error))
cat(sprintf("BS Price:       %.4f\n", par_res$bs_price))
cat(sprintf("Threads used:   %d\n",  par_res$n_threads_used))

# ── Scalability benchmark ────────────────────────────────────────────────────
cat("\n=== Scalability Analysis ===\n")
thread_counts <- as.integer(c(1, 2, 4, 8))
bm_df <- parallel_benchmark(S0, K, r, sigma, T,
                              n_paths = 500000, n_steps = steps, seed = seed,
                              thread_counts = thread_counts)
bm_df$speedup <- bm_df$elapsed_sec[1] / bm_df$elapsed_sec

cat("\nScalability results:\n")
print(bm_df)

# ── Speedup plot ─────────────────────────────────────────────────────────────
p_speedup <- ggplot(bm_df, aes(n_threads, speedup)) +
  geom_line(colour = "#2ca25f", linewidth = 1.2) +
  geom_point(colour = "#2ca25f", size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  labs(title = "Parallel Monte Carlo Speedup",
       x = "Number of Threads",
       y = "Speedup vs Single Thread",
       caption = "Dashed line = ideal linear speedup") +
  scale_x_continuous(breaks = thread_counts) +
  theme_minimal()
print(p_speedup)

# ── Time plot ────────────────────────────────────────────────────────────────
p_time <- ggplot(bm_df, aes(n_threads, elapsed_sec)) +
  geom_col(fill = "#2ca25f", alpha = 0.8, width = 0.5) +
  geom_text(aes(label = sprintf("%.2fs", elapsed_sec)), vjust = -0.4, size = 3.5) +
  labs(title = "Wall-Clock Time vs Thread Count",
       x = "Number of Threads", y = "Elapsed Time (s)") +
  scale_x_continuous(breaks = thread_counts) +
  theme_minimal()
print(p_time)
