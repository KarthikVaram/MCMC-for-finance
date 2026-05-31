#pragma once
#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>

namespace mcmc {

// ── Constants ─────────────────────────────────────────────────────────────────
constexpr double LOG_2PI = 1.8378770664093453;
constexpr double NEG_INF = -1e300;

// ── Log-likelihood: Black-Scholes implied vol surface ────────────────────────
// Evaluates log p(observed_prices | theta) under Gaussian observation noise.
// observed: vector of observed option prices
// model:    vector of model-implied prices (same length)
// sigma_obs: observation noise standard deviation
inline double log_likelihood_gaussian(const std::vector<double>& observed,
                                       const std::vector<double>& model,
                                       double sigma_obs) {
    if (observed.size() != model.size() || sigma_obs <= 0.0)
        return NEG_INF;
    int n = observed.size();
    double lp = -0.5 * n * (LOG_2PI + 2.0 * std::log(sigma_obs));
    for (int i = 0; i < n; ++i) {
        double r = (observed[i] - model[i]) / sigma_obs;
        lp -= 0.5 * r * r;
    }
    return lp;
}

// ── Log-likelihood: log-return series (Gaussian) ─────────────────────────────
// Used for GBM / constant-vol model calibration.
inline double log_likelihood_returns(const std::vector<double>& log_returns,
                                      double mu, double sigma, double dt) {
    if (sigma <= 0.0) return NEG_INF;
    int n = log_returns.size();
    double mean  = (mu - 0.5 * sigma * sigma) * dt;
    double var   = sigma * sigma * dt;
    double lp    = -0.5 * n * (LOG_2PI + std::log(var));
    for (int i = 0; i < n; ++i) {
        double r = (log_returns[i] - mean);
        lp -= 0.5 * r * r / var;
    }
    return lp;
}

// ── Prior distributions (log scale) ──────────────────────────────────────────

// Log of Gaussian prior N(mu0, sigma0^2)
inline double log_prior_normal(double x, double mu0, double sigma0) {
    double r = (x - mu0) / sigma0;
    return -0.5 * (LOG_2PI + 2.0 * std::log(sigma0) + r * r);
}

// Log of half-normal prior (for positive parameters like sigma, xi)
// Equivalent to N(0, sigma0^2) truncated to x > 0
inline double log_prior_half_normal(double x, double sigma0) {
    if (x <= 0.0) return NEG_INF;
    return -0.5 * (LOG_2PI + 2.0 * std::log(sigma0) + (x / sigma0) * (x / sigma0))
           + std::log(2.0);
}

// Log of Gamma prior Gamma(alpha, beta) — good for kappa, xi
inline double log_prior_gamma(double x, double alpha, double beta) {
    if (x <= 0.0) return NEG_INF;
    return (alpha - 1.0) * std::log(x) - beta * x
           + alpha * std::log(beta) - std::lgamma(alpha);
}

// Log of Beta prior Beta(a, b) scaled to (-1, 1) — good for rho
// x must be in (-1, 1); internally mapped to u = (x+1)/2 in (0,1)
inline double log_prior_rho(double rho, double a, double b) {
    if (rho <= -1.0 || rho >= 1.0) return NEG_INF;
    double u = 0.5 * (rho + 1.0);
    return (a - 1.0) * std::log(u) + (b - 1.0) * std::log(1.0 - u)
           - std::lgamma(a) - std::lgamma(b) + std::lgamma(a + b)
           - std::log(2.0);   // Jacobian of the transform
}

// ── Log-posterior helper ──────────────────────────────────────────────────────
// log p(theta | data) = log p(data | theta) + log p(theta)
inline double log_posterior(double log_lik, double log_prior_val) {
    return log_lik + log_prior_val;
}

// ── DIC (Deviance Information Criterion) ─────────────────────────────────────
// DIC = -2 * E[log p(y|theta)] + 2 * p_D
// where p_D = E[D(theta)] - D(theta_bar)  (effective number of parameters)
struct DIC {
    double dic;
    double p_D;        // effective parameters
    double d_bar;      // mean deviance
};

inline DIC compute_dic(const std::vector<double>& log_liks,
                        double log_lik_at_mean) {
    int n = log_liks.size();
    double d_bar = 0.0;
    for (double ll : log_liks) d_bar += -2.0 * ll;
    d_bar /= n;
    double d_mean = -2.0 * log_lik_at_mean;
    double p_D    = d_bar - d_mean;
    return {d_bar + p_D, p_D, d_bar};
}

// ── Waic (Widely Applicable IC) ──────────────────────────────────────────────
// Simple pointwise WAIC using a matrix of per-observation log likelihoods.
// ll_matrix: n_obs x n_samples
inline double compute_waic(const std::vector<std::vector<double>>& ll_matrix) {
    int n_obs     = ll_matrix.size();
    int n_samples = ll_matrix[0].size();
    double waic   = 0.0;

    for (int i = 0; i < n_obs; ++i) {
        // lppd term: log mean exp
        double max_ll = *std::max_element(ll_matrix[i].begin(), ll_matrix[i].end());
        double sum_exp = 0.0;
        for (double ll : ll_matrix[i]) sum_exp += std::exp(ll - max_ll);
        double lppd_i = max_ll + std::log(sum_exp / n_samples);

        // Variance penalty
        double mean_ll = 0.0;
        for (double ll : ll_matrix[i]) mean_ll += ll;
        mean_ll /= n_samples;
        double var_ll = 0.0;
        for (double ll : ll_matrix[i]) var_ll += (ll - mean_ll) * (ll - mean_ll);
        var_ll /= (n_samples - 1);

        waic += -2.0 * (lppd_i - var_ll);
    }
    return waic;
}

} // namespace mcmc
