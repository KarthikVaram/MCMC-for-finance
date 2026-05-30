# Stage 7: Heston Stochastic Volatility Model

library(Rcpp)
library(ggplot2)
library(tidyr)

sourceCpp("src/heston.cpp")
sourceCpp("src/mc_pricer.cpp")    # BS / GBM baseline

# ── Heston Parameters ─────────────────────────────────────────────────────────
# S0     = 100     initial price
# v0     = 0.04    initial variance (vol = 20%)
# kappa  = 2.0     mean-reversion speed
# theta  = 0.04    long-run variance (vol = 20%)
# xi     = 0.30    vol-of-vol
# rho    = -0.70   stock-vol correlation (usually negative)
# r      = 0.05    risk-free rate
# T      = 1.0     maturity

S0    <- 100;  v0    <- 0.04
kappa <- 2.0;  theta <- 0.04;  xi <- 0.30;  rho <- -0.70
r     <- 0.05; T     <- 1.0
K     <- 100
N     <- 100000; steps <- 252; seed <- 42

# ── 1. Simulate Heston paths ─────────────────────────────────────────────────
cat("\n=== Heston Path Simulation ===\n")
n_plot <- 10
paths  <- heston_paths(S0, v0, mu = r, kappa, theta, xi, rho, T, steps, n_plot, seed)

time_pts <- seq(0, T, length.out = steps + 1)
S_mat    <- paths$S
V_mat    <- paths$V

# Price paths
s_long <- do.call(rbind, lapply(seq_len(n_plot), function(i)
  data.frame(time = time_pts, value = as.numeric(S_mat[i, ]), path = i, series = "Price")))

# Variance paths (convert to vol)
v_long <- do.call(rbind, lapply(seq_len(n_plot), function(i)
  data.frame(time = time_pts, value = sqrt(as.numeric(V_mat[i, ])), path = i, series = "Vol")))

all_long <- rbind(s_long, v_long)

p_paths <- ggplot(s_long, aes(time, value, group = path, colour = factor(path))) +
  geom_line(alpha = 0.75, linewidth = 0.6) +
  guides(colour = "none") +
  labs(title = "Heston Model: Stock Price Paths",
       x = "Time (years)", y = "Stock Price") +
  theme_minimal()
print(p_paths)

p_vol <- ggplot(v_long, aes(time, value, group = path, colour = factor(path))) +
  geom_line(alpha = 0.75, linewidth = 0.6) +
  geom_hline(yintercept = sqrt(theta), linetype = "dashed", colour = "grey40") +
  guides(colour = "none") +
  labs(title = "Heston Model: Instantaneous Volatility Paths",
       x = "Time (years)", y = "Volatility (annualised)",
       caption = sprintf("Dashed = long-run vol = %.0f%%", sqrt(theta)*100)) +
  theme_minimal()
print(p_vol)

# ── 2. Option Pricing ─────────────────────────────────────────────────────────
cat("\n=== Heston Option Pricing ===\n")
h_call <- heston_call(S0, v0, K, r, kappa, theta, xi, rho, T, N, steps, seed)
h_put  <- heston_put (S0, v0, K, r, kappa, theta, xi, rho, T, N, steps, seed)
bs_c   <- mc_call(S0, K, r, sqrt(theta), T, N, steps, seed)  # BS with long-run vol

cat(sprintf("Heston Call:  %.4f  (SE: %.5f)\n", h_call$price, h_call$std_error))
cat(sprintf("BS Call:      %.4f  (at long-run vol = %.0f%%)\n",
            h_call$bs_equiv_price, sqrt(theta) * 100))
cat(sprintf("Heston Put:   %.4f  (SE: %.5f)\n", h_put$price, h_put$std_error))

# ── 3. Implied Volatility Smile ───────────────────────────────────────────────
cat("\n=== Implied Volatility Smile ===\n")

# Price calls across strikes
strikes <- seq(80, 125, by = 5)
smile   <- heston_smile(S0, v0, r, kappa, theta, xi, rho, T, strikes, N, steps, seed)

# Numerically invert BS to get implied vol for each strike
bs_call_price <- function(K_val, iv) {
  d1 <- (log(S0 / K_val) + (r + 0.5 * iv^2) * T) / (iv * sqrt(T))
  d2 <- d1 - iv * sqrt(T)
  S0 * pnorm(d1) - K_val * exp(-r * T) * pnorm(d2)
}

implied_vol <- sapply(seq_len(nrow(smile)), function(i) {
  K_val <- smile$strike[i]
  price <- smile$price[i]
  tryCatch(
    uniroot(function(iv) bs_call_price(K_val, iv) - price, c(0.001, 5))$root,
    error = function(e) NA
  )
})

smile$implied_vol <- implied_vol * 100  # percentage

p_smile <- ggplot(smile, aes(strike, implied_vol)) +
  geom_line(colour = "#756bb1", linewidth = 1.3) +
  geom_point(colour = "#756bb1", size = 3) +
  geom_hline(yintercept = sqrt(theta) * 100, linetype = "dashed", colour = "grey50") +
  labs(title = "Heston Model: Implied Volatility Smile",
       x = "Strike", y = "Implied Volatility (%)",
       caption = sprintf("Dashed = flat BS vol = %.0f%%", sqrt(theta)*100)) +
  theme_minimal()
print(p_smile)

cat("\nStrike | Heston Price | Implied Vol\n")
cat(paste(rep("-", 42), collapse = ""), "\n")
for (i in seq_len(nrow(smile))) {
  cat(sprintf("%6.0f | %12.4f | %8.2f%%\n",
              smile$strike[i], smile$price[i], smile$implied_vol[i]))
}

# ── 4. Rho / Correlation Sensitivity ─────────────────────────────────────────
cat("\n=== Rho Sensitivity ===\n")
rho_vals  <- seq(-0.9, 0.9, by = 0.3)
rho_prices <- sapply(rho_vals, function(ro) {
  heston_call(S0, v0, K, r, kappa, theta, xi, ro, T, N, steps, seed)$price
})

df_rho <- data.frame(rho = rho_vals, price = rho_prices)
p_rho  <- ggplot(df_rho, aes(rho, price)) +
  geom_line(colour = "#fd8d3c", linewidth = 1.2) +
  geom_point(colour = "#fd8d3c", size = 3) +
  labs(title = "Heston Call Price vs Correlation (rho)",
       x = expression(rho), y = "Call Price") +
  theme_minimal()
print(p_rho)
