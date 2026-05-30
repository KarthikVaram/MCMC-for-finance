# Stage 4: Variance Reduction Techniques

library(Rcpp)
library(ggplot2)
library(microbenchmark)

sourceCpp("src/variance_reduction.cpp")
sourceCpp("src/mc_pricer.cpp")      # for crude MC baseline

# ── Parameters ───────────────────────────────────────────────────────────────
S0 <- 100; K <- 105; r <- 0.05; sigma <- 0.20; T <- 1.0
N  <- 100000; steps <- 252; seed <- 42

# ── Crude MC baseline ────────────────────────────────────────────────────────
crude <- mc_call(S0, K, r, sigma, T, N, steps, seed)

# ── All variance reduction methods ───────────────────────────────────────────
anti <- mc_antithetic(S0, K, r, sigma, T, N, steps, seed)
cv   <- mc_control_variate(S0, K, r, sigma, T, N, steps, seed)
is_  <- mc_importance_sampling(S0, K, r, sigma, T, N, steps, seed)

results <- data.frame(
  Method    = c("Crude MC", "Antithetic", "Control Variate", "Importance Sampling"),
  Price     = c(crude$price, anti$price, cv$price, is_$price),
  StdError  = c(crude$std_error, anti$std_error, cv$std_error, is_$std_error),
  BSPrice   = crude$bs_price
)
results$VarReduction <- (crude$std_error / results$StdError)^2

cat("\n=== Variance Reduction Comparison ===\n")
cat(sprintf("Black-Scholes reference: %.4f\n\n", crude$bs_price))
print(results, digits = 5)

# ── Bar chart of standard errors ─────────────────────────────────────────────
p_se <- ggplot(results, aes(Method, StdError, fill = Method)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = sprintf("%.5f", StdError)), vjust = -0.4, size = 3.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Standard Error by Variance Reduction Method",
       x = NULL, y = "Std Error") +
  theme_minimal()
print(p_se)

# ── Variance reduction factors ────────────────────────────────────────────────
p_vr <- ggplot(results, aes(Method, VarReduction, fill = Method)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_text(aes(label = sprintf("%.2fx", VarReduction)), vjust = -0.4, size = 3.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Variance Reduction Factor vs Crude MC",
       x = NULL, y = "Variance Reduction Factor") +
  theme_minimal()
print(p_vr)

# ── Runtime benchmark ────────────────────────────────────────────────────────
cat("\n=== Runtime Benchmark ===\n")
bm <- microbenchmark(
  crude      = mc_call(S0, K, r, sigma, T, N, steps, seed),
  antithetic = mc_antithetic(S0, K, r, sigma, T, N, steps, seed),
  ctrl_var   = mc_control_variate(S0, K, r, sigma, T, N, steps, seed),
  imp_samp   = mc_importance_sampling(S0, K, r, sigma, T, N, steps, seed),
  times      = 10L
)
print(bm)
