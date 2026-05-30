# Stage 2: Random Number Generation and Stochastic Processes

library(Rcpp)
library(ggplot2)

sourceCpp("src/rng.cpp")
sourceCpp("src/brownian.cpp")
sourceCpp("src/gbm.cpp")

# ── 1. RNG Verification ─────────────────────────────────────────────────────
cat("\n=== RNG Verification ===\n")
stats <- rng_verify(n = 1e6)
cat(sprintf("Mean: %.5f  (expected ~0)\n", stats["mean"]))
cat(sprintf("Var:  %.5f  (expected ~1)\n", stats["variance"]))

samples <- rng_normal(10000)
df_rng  <- data.frame(x = samples)
p_rng   <- ggplot(df_rng, aes(x)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = "#3182bd", colour = "white", alpha = 0.8) +
  stat_function(fun = dnorm, colour = "red", linewidth = 1) +
  labs(title = "Standard Normal RNG (C++)",
       x = "Value", y = "Density") +
  theme_minimal()
print(p_rng)

# ── 2. Brownian Motion ──────────────────────────────────────────────────────
cat("\n=== Brownian Motion ===\n")
bm_paths <- simulate_bm(n_paths = 20, n_steps = 500, T = 1.0, seed = 99)
time_pts <- seq(0, 1, length.out = 501)

bm_long <- do.call(rbind, lapply(seq_len(nrow(bm_paths)), function(i) {
  data.frame(time = time_pts, W = as.numeric(bm_paths[i, ]), path = i)
}))

p_bm <- ggplot(bm_long, aes(time, W, group = path, colour = factor(path))) +
  geom_line(alpha = 0.7, linewidth = 0.5) +
  guides(colour = "none") +
  labs(title = "Brownian Motion Paths", x = "Time", y = "W(t)") +
  theme_minimal()
print(p_bm)

# ── 3. Geometric Brownian Motion ────────────────────────────────────────────
cat("\n=== Geometric Brownian Motion ===\n")
S0 <- 100; mu <- 0.07; sigma <- 0.20; T <- 1.0; n_steps <- 252; n_paths <- 30

gbm_mat <- simulate_gbm(S0 = S0, mu = mu, sigma = sigma,
                         T = T, n_steps = n_steps, n_paths = n_paths)
time_pts <- seq(0, T, length.out = n_steps + 1)

gbm_long <- do.call(rbind, lapply(seq_len(nrow(gbm_mat)), function(i) {
  data.frame(time = time_pts, S = as.numeric(gbm_mat[i, ]), path = i)
}))

p_gbm <- ggplot(gbm_long, aes(time, S, group = path, colour = factor(path))) +
  geom_line(alpha = 0.7, linewidth = 0.5) +
  guides(colour = "none") +
  labs(title = sprintf("GBM Paths (mu=%.0f%%, sigma=%.0f%%)", mu*100, sigma*100),
       x = "Time (years)", y = "Stock Price") +
  theme_minimal()
print(p_gbm)

# Terminal distribution
ST <- gbm_terminal(S0, mu, sigma, T, n_steps, n_paths = 50000)
cat(sprintf("GBM terminal: E[S_T] = %.2f  (theoretical %.2f)\n",
            mean(ST), S0 * exp(mu * T)))

p_term <- ggplot(data.frame(ST = ST), aes(ST)) +
  geom_histogram(bins = 80, fill = "#e6550d", colour = "white", alpha = 0.8) +
  labs(title = "Terminal Stock Price Distribution (GBM)", x = "S_T", y = "Count") +
  theme_minimal()
print(p_term)
