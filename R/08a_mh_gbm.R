# =============================================================================
#  08a_mh_gbm.R  —  Bayesian GBM calibration via Metropolis-Hastings
#
#  WHAT THIS SCRIPT DOES
#  ─────────────────────
#  1. Simulates (or loads) a stock price series.
#  2. Computes daily log-returns.
#  3. Runs the MH sampler to get posterior draws of (mu, sigma).
#  4. Runs the adaptive version to auto-tune proposals.
#  5. Runs two chains and checks convergence with R-hat.
#  6. Produces all standard MCMC diagnostics:
#       • Trace plots  (detect non-convergence)
#       • Density plots (visualise posteriors)
#       • Joint posterior scatter (mu vs sigma)
#       • Autocorrelation plots (check mixing)
#       • Log-posterior over iterations (detect burn-in length needed)
#  7. Compares posterior to MLE estimates.
# =============================================================================

library(Rcpp)
library(ggplot2)
library(patchwork)    # install if missing: install.packages("patchwork")

sourceCpp("src/mh_gbm.cpp")

set.seed(42)

# =============================================================================
#  1. DATA: simulate a GBM stock price series
# =============================================================================
# True parameters (what we're trying to recover)
true_mu    <- 0.08   # 8% annual drift
true_sigma <- 0.20   # 20% annual volatility
T_years    <- 5
dt         <- 1 / 252                  # daily
n_days     <- as.integer(T_years / dt)

# Exact discretisation: S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
Z      <- rnorm(n_days)
log_S  <- cumsum((true_mu - 0.5 * true_sigma^2) * dt + true_sigma * sqrt(dt) * Z)
S      <- 100 * exp(log_S)
log_ret <- diff(log(c(100, S)))   # T log-returns

cat(sprintf("Data: %d daily log-returns\n", length(log_ret)))
cat(sprintf("Sample mean return (annualised): %.3f\n", mean(log_ret) / dt))
cat(sprintf("Sample vol (annualised):         %.3f\n", sd(log_ret) / sqrt(dt)))

# =============================================================================
#  2. MLE REFERENCE
# =============================================================================
# Under GBM:  E[r_t] = (mu - sigma^2/2)*dt,  Var[r_t] = sigma^2*dt
# Solving:
#   sigma_MLE = sqrt( Var(r) / dt )
#   mu_MLE    = mean(r)/dt + sigma_MLE^2 / 2

sigma_mle <- sd(log_ret) / sqrt(dt)
mu_mle    <- mean(log_ret) / dt + 0.5 * sigma_mle^2
cat(sprintf("\nMLE: mu = %.4f,  sigma = %.4f\n", mu_mle, sigma_mle))

# =============================================================================
#  3. MH SAMPLER (fixed proposals)
# =============================================================================
cat("\n=== Running MH sampler ===\n")
fit <- mh_gbm(
  log_returns   = log_ret,
  dt            = dt,
  n_iter        = 40000,
  burnin        = 5000,
  prop_sd_mu    = 0.02,
  prop_sd_sigma = 0.01,
  seed          = 42
)

cat(sprintf("Acceptance rate: %.2f%%  (target: 30-50%%)\n",
            fit$accept_rate * 100))
cat("\nPosterior summary:\n")
print(fit$summary, digits = 4)

# =============================================================================
#  4. ADAPTIVE MH (auto-tuned proposals)
# =============================================================================
cat("\n=== Running Adaptive MH ===\n")
fit_adapt <- mh_gbm_adaptive(
  log_returns  = log_ret,
  dt           = dt,
  n_iter       = 40000,
  burnin       = 5000,
  target_rate  = 0.44,
  adapt_every  = 100,
  seed         = 42
)
cat(sprintf("Adaptive acceptance rate: %.2f%%\n", fit_adapt$accept_rate * 100))
cat(sprintf("Adapted proposal sd (mu):    %.5f\n", fit_adapt$adapted_prop_sd_mu))
cat(sprintf("Adapted proposal sd (sigma): %.5f\n", fit_adapt$adapted_prop_sd_sigma))

# =============================================================================
#  5. CONVERGENCE: two chains + R-hat
# =============================================================================
cat("\n=== Convergence diagnostics ===\n")
conv <- mh_gbm_convergence(
  log_returns   = log_ret,
  dt            = dt,
  n_iter        = 40000,
  burnin        = 5000,
  prop_sd_mu    = 0.02,
  prop_sd_sigma = 0.01
)

cat(sprintf("R-hat mu:    %.4f  %s\n", conv$r_hat_mu,
            ifelse(conv$r_hat_mu < 1.01, "(converged)", "(NOT converged - run longer!)")))
cat(sprintf("R-hat sigma: %.4f  %s\n", conv$r_hat_sigma,
            ifelse(conv$r_hat_sigma < 1.01, "(converged)", "(NOT converged)")))

# =============================================================================
#  6. DIAGNOSTICS PLOTS
# =============================================================================

n_keep <- length(fit$mu_chain)
iter   <- seq_len(n_keep)

# ── 6a. Trace plots ───────────────────────────────────────────────────────────
# A "hairy caterpillar" that doesn't trend = good mixing.
p_trace_mu <- ggplot(data.frame(iter = iter, mu = fit$mu_chain),
                      aes(iter, mu)) +
  geom_line(colour = "#756bb1", alpha = 0.6, linewidth = 0.3) +
  geom_hline(yintercept = true_mu,   colour = "red",   linetype = "dashed") +
  geom_hline(yintercept = mu_mle,    colour = "blue",  linetype = "dotted") +
  labs(title = "Trace plot: μ",
       subtitle = "Red = true, Blue = MLE",
       x = "Iteration (post burn-in)", y = "μ") +
  theme_minimal()

p_trace_sig <- ggplot(data.frame(iter = iter, sigma = fit$sigma_chain),
                       aes(iter, sigma)) +
  geom_line(colour = "#e6550d", alpha = 0.6, linewidth = 0.3) +
  geom_hline(yintercept = true_sigma, colour = "red",  linetype = "dashed") +
  geom_hline(yintercept = sigma_mle,  colour = "blue", linetype = "dotted") +
  labs(title = "Trace plot: σ",
       subtitle = "Red = true, Blue = MLE",
       x = "Iteration (post burn-in)", y = "σ") +
  theme_minimal()

# ── 6b. Posterior density plots ───────────────────────────────────────────────
p_dens_mu <- ggplot(data.frame(mu = fit$mu_chain), aes(mu)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = "#756bb1", alpha = 0.7, colour = "white") +
  geom_density(colour = "#756bb1", linewidth = 1) +
  geom_vline(xintercept = true_mu,  colour = "red",  linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = mu_mle,   colour = "blue", linetype = "dotted", linewidth = 1) +
  geom_vline(xintercept = fit$summary$mean[1], colour = "black", linewidth = 1) +
  geom_vline(xintercept = fit$summary$q025[1], colour = "grey40", linetype = "dashed") +
  geom_vline(xintercept = fit$summary$q975[1], colour = "grey40", linetype = "dashed") +
  labs(title = "Posterior: μ",
       subtitle = sprintf("Mean = %.3f  [%.3f, %.3f]",
                          fit$summary$mean[1], fit$summary$q025[1], fit$summary$q975[1]),
       x = "μ", y = "Density") +
  theme_minimal()

p_dens_sig <- ggplot(data.frame(sigma = fit$sigma_chain), aes(sigma)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = "#e6550d", alpha = 0.7, colour = "white") +
  geom_density(colour = "#e6550d", linewidth = 1) +
  geom_vline(xintercept = true_sigma,  colour = "red",  linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = sigma_mle,   colour = "blue", linetype = "dotted", linewidth = 1) +
  geom_vline(xintercept = fit$summary$mean[2], colour = "black", linewidth = 1) +
  geom_vline(xintercept = fit$summary$q025[2], colour = "grey40", linetype = "dashed") +
  geom_vline(xintercept = fit$summary$q975[2], colour = "grey40", linetype = "dashed") +
  labs(title = "Posterior: σ",
       subtitle = sprintf("Mean = %.3f  [%.3f, %.3f]",
                          fit$summary$mean[2], fit$summary$q025[2], fit$summary$q975[2]),
       x = "σ", y = "Density") +
  theme_minimal()

# ── 6c. Joint posterior: mu vs sigma ─────────────────────────────────────────
# Shows posterior correlation between parameters.
# With enough data, mu and sigma are nearly independent under GBM.
p_joint <- ggplot(data.frame(mu = fit$mu_chain, sigma = fit$sigma_chain),
                  aes(mu, sigma)) +
  geom_point(alpha = 0.05, size = 0.5, colour = "#2ca25f") +
  geom_density_2d(colour = "#2ca25f", linewidth = 0.6) +
  geom_point(aes(x = true_mu, y = true_sigma),
             colour = "red", size = 4, shape = 4, stroke = 2) +
  geom_point(aes(x = mu_mle, y = sigma_mle),
             colour = "blue", size = 4, shape = 3, stroke = 2) +
  labs(title = "Joint posterior: μ vs σ",
       subtitle = "Red × = true  |  Blue + = MLE",
       x = "μ", y = "σ") +
  theme_minimal()

# ── 6d. Log-posterior trace ───────────────────────────────────────────────────
# Should stabilise quickly after burn-in. A gradual upward trend
# means the chain is still searching → increase burn-in.
p_lp <- ggplot(data.frame(iter = iter, lp = fit$lp_chain), aes(iter, lp)) +
  geom_line(colour = "steelblue", alpha = 0.7, linewidth = 0.3) +
  labs(title = "Log-posterior trace",
       x = "Iteration (post burn-in)", y = "log π(θ | data)") +
  theme_minimal()

# ── 6e. Two-chain trace plot ──────────────────────────────────────────────────
# Both chains should overlap completely if converged.
n_conv <- length(conv$chain1$mu_chain)
df_two_chains <- rbind(
  data.frame(iter = seq_len(n_conv), mu = conv$chain1$mu_chain,
             sigma = conv$chain1$sigma_chain, chain = "Chain 1"),
  data.frame(iter = seq_len(n_conv), mu = conv$chain2$mu_chain,
             sigma = conv$chain2$sigma_chain, chain = "Chain 2")
)

p_two_mu <- ggplot(df_two_chains, aes(iter, mu, colour = chain)) +
  geom_line(alpha = 0.5, linewidth = 0.3) +
  scale_colour_manual(values = c("Chain 1" = "#756bb1", "Chain 2" = "#e6550d")) +
  labs(title = sprintf("Two-chain trace: μ  (R-hat = %.4f)", conv$r_hat_mu),
       x = "Iteration", y = "μ", colour = NULL) +
  theme_minimal()

p_two_sig <- ggplot(df_two_chains, aes(iter, sigma, colour = chain)) +
  geom_line(alpha = 0.5, linewidth = 0.3) +
  scale_colour_manual(values = c("Chain 1" = "#756bb1", "Chain 2" = "#e6550d")) +
  labs(title = sprintf("Two-chain trace: σ  (R-hat = %.4f)", conv$r_hat_sigma),
       x = "Iteration", y = "σ", colour = NULL) +
  theme_minimal()

# ── 6f. Autocorrelation plots ─────────────────────────────────────────────────
# Fast decay to zero = good mixing. Slow decay = increase thinning or improve proposals.
acf_mu  <- acf(fit$mu_chain,    plot = FALSE, lag.max = 50)
acf_sig <- acf(fit$sigma_chain, plot = FALSE, lag.max = 50)

df_acf <- rbind(
  data.frame(lag = acf_mu$lag[-1],  acf = acf_mu$acf[-1],  param = "μ"),
  data.frame(lag = acf_sig$lag[-1], acf = acf_sig$acf[-1], param = "σ")
)

p_acf <- ggplot(df_acf, aes(lag, acf, colour = param)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_hline(yintercept = c(-0.05, 0.05), linetype = "dashed", colour = "grey70") +
  geom_segment(aes(xend = lag, yend = 0), linewidth = 0.8) +
  facet_wrap(~param) +
  scale_colour_manual(values = c("μ" = "#756bb1", "σ" = "#e6550d")) +
  labs(title = "Autocorrelation (post burn-in)",
       subtitle = sprintf("ESS: μ = %.0f,  σ = %.0f",
                          fit$summary$ess[1], fit$summary$ess[2]),
       x = "Lag", y = "Autocorrelation") +
  theme_minimal() + theme(legend.position = "none")

# ── Print all plots ───────────────────────────────────────────────────────────
print((p_trace_mu  | p_trace_sig)  + plot_annotation(title = "Trace plots"))
print((p_dens_mu   | p_dens_sig)   + plot_annotation(title = "Posterior densities"))
print(p_joint)
print(p_lp)
print((p_two_mu    | p_two_sig)    + plot_annotation(title = "Two-chain convergence"))
print(p_acf)

# =============================================================================
#  7. SUMMARY TABLE
# =============================================================================
cat("\n========================================\n")
cat("  Bayesian GBM Calibration — Results\n")
cat("========================================\n")
cat(sprintf("  True mu:       %.4f\n", true_mu))
cat(sprintf("  MLE mu:        %.4f\n", mu_mle))
cat(sprintf("  Posterior mu:  %.4f  [%.4f, %.4f]\n",
            fit$summary$mean[1], fit$summary$q025[1], fit$summary$q975[1]))
cat("\n")
cat(sprintf("  True sigma:    %.4f\n", true_sigma))
cat(sprintf("  MLE sigma:     %.4f\n", sigma_mle))
cat(sprintf("  Post sigma:    %.4f  [%.4f, %.4f]\n",
            fit$summary$mean[2], fit$summary$q025[2], fit$summary$q975[2]))
cat("\n")
cat(sprintf("  ESS mu:    %.0f / %d  (%.1f%%)\n",
            fit$summary$ess[1], n_keep, 100 * fit$summary$ess[1] / n_keep))
cat(sprintf("  ESS sigma: %.0f / %d  (%.1f%%)\n",
            fit$summary$ess[2], n_keep, 100 * fit$summary$ess[2] / n_keep))
cat(sprintf("  R-hat mu:    %.4f\n", conv$r_hat_mu))
cat(sprintf("  R-hat sigma: %.4f\n", conv$r_hat_sigma))
cat("========================================\n")
