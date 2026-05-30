# Stage 3: Monte Carlo Option Pricing

library(Rcpp)
library(ggplot2)

sourceCpp("src/mc_pricer.cpp")

# ── Parameters ───────────────────────────────────────────────────────────────
S0 <- 100; K <- 105; r <- 0.05; sigma <- 0.20; T <- 1.0

# ── Call Pricing ─────────────────────────────────────────────────────────────
cat("\n=== European Call ===\n")
call_res <- mc_call(S0, K, r, sigma, T, n_paths = 200000)
cat(sprintf("MC Price:  %.4f\n", call_res$price))
cat(sprintf("BS Price:  %.4f\n", call_res$bs_price))
cat(sprintf("Error:     %.4f\n", call_res$error))
cat(sprintf("Std Error: %.4f\n", call_res$std_error))
cat(sprintf("95%% CI:  [%.4f, %.4f]\n",
            call_res$price - 1.96 * call_res$std_error,
            call_res$price + 1.96 * call_res$std_error))

# ── Put Pricing ──────────────────────────────────────────────────────────────
cat("\n=== European Put ===\n")
put_res <- mc_put(S0, K, r, sigma, T, n_paths = 200000)
cat(sprintf("MC Price:  %.4f\n", put_res$price))
cat(sprintf("BS Price:  %.4f\n", put_res$bs_price))
cat(sprintf("Error:     %.4f\n", put_res$error))

# ── Put-Call Parity Check ────────────────────────────────────────────────────
parity_lhs <- call_res$price - put_res$price
parity_rhs <- S0 - K * exp(-r * T)
cat(sprintf("\nPut-Call Parity: C - P = %.4f  (theoretical %.4f)\n",
            parity_lhs, parity_rhs))

# ── Convergence Study ────────────────────────────────────────────────────────
cat("\n=== Convergence Analysis ===\n")
path_sizes <- as.integer(c(100, 500, 1000, 5000, 10000, 50000, 100000, 500000))
conv_df    <- mc_convergence(S0, K, r, sigma, T, path_sizes)
conv_df$bs <- call_res$bs_price
conv_df$lower <- conv_df$price - 1.96 * conv_df$std_error
conv_df$upper <- conv_df$price + 1.96 * conv_df$std_error

p_conv <- ggplot(conv_df, aes(n_paths, price)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#3182bd", alpha = 0.25) +
  geom_line(colour = "#3182bd", linewidth = 1) +
  geom_point(colour = "#3182bd", size = 2) +
  geom_hline(aes(yintercept = bs), colour = "red", linetype = "dashed", linewidth = 1) +
  scale_x_log10(labels = scales::comma) +
  labs(title = "Monte Carlo Convergence to Black-Scholes Price",
       x = "Number of Paths (log scale)", y = "Option Price",
       caption = "Red dashed = Black-Scholes") +
  theme_minimal()
print(p_conv)

print(conv_df[, c("n_paths", "price", "std_error")])
