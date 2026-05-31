#pragma once
// =============================================================================
//  gbm_bayes.h  —  Bayesian GBM calibration via Metropolis-Hastings
//
//  MODEL
//  ─────
//  Geometric Brownian Motion (GBM):
//
//      dS_t = μ S_t dt + σ S_t dW_t
//
//  Applying Itô's lemma to X_t = log(S_t):
//
//      dX_t = (μ − σ²/2) dt + σ dW_t
//
//  Exact Euler discretisation over Δt:
//
//      r_t  ≡  log(S_t / S_{t−1})
//           =  (μ − σ²/2)·Δt  +  σ·√Δt · Z_t,    Z_t ~ N(0,1)
//
//  ⟹  r_t | μ,σ  ~  N( m, v )
//       where  m = (μ − σ²/2)·Δt
//              v = σ²·Δt
//
//  LIKELIHOOD
//  ──────────
//  Given T observed log-returns {r_1,...,r_T} (i.i.d. given parameters):
//
//      log p(r | μ,σ)  =  Σ_t  log φ(r_t; m, v)
//                       =  −T/2 · log(2π)  −  T/2 · log(v)
//                          −  1/(2v) · Σ_t (r_t − m)²
//
//  PRIORS
//  ──────
//  We choose weakly informative priors:
//
//      μ  ~ N(0, σ_μ²)              (drift can be positive or negative)
//      σ  ~ Half-Normal(0, σ_σ²)    (volatility must be positive)
//
//  Half-Normal: p(σ) = 2·φ(σ; 0, σ_σ²)·1{σ > 0}
//
//  POSTERIOR
//  ─────────
//  By Bayes' theorem (up to normalising constant):
//
//      log π(μ,σ | r)  =  log p(r | μ,σ)  +  log p(μ)  +  log p(σ)
//                          + const
//
//  This posterior has no closed form → we use Metropolis-Hastings to sample it.
// =============================================================================

#include <cmath>
#include <vector>
#include <numeric>
#include <limits>

namespace gbm_bayes {

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr double LOG_2PI  = 1.8378770664093453;   // log(2π)
constexpr double NEG_INF  = -1e300;

// ── Log-likelihood ────────────────────────────────────────────────────────────
//
//  log p(r | μ, σ)  under GBM
//
//  Parameters
//  ----------
//  log_returns : observed daily log-returns r_t = log(S_t / S_{t-1})
//  mu          : drift  μ  (annualised)
//  sigma       : volatility  σ  (annualised, must be > 0)
//  dt          : time step in years (e.g. 1/252 for daily data)
//
//  Returns log p  (−∞ if sigma ≤ 0)

inline double log_likelihood(const std::vector<double>& log_returns,
                               double mu, double sigma, double dt) {
    if (sigma <= 0.0) return NEG_INF;

    //  Gaussian parameters for one log-return
    double m   = (mu - 0.5 * sigma * sigma) * dt;   // mean of r_t
    double v   = sigma * sigma * dt;                 // variance of r_t
    int    T   = log_returns.size();

    //  log p = −T/2 · log(2πv)  −  Σ(r_t − m)² / (2v)
    double log_p = -0.5 * T * (LOG_2PI + std::log(v));
    for (double r : log_returns)
        log_p -= 0.5 * (r - m) * (r - m) / v;

    return log_p;
}

// ── Log-priors ────────────────────────────────────────────────────────────────

//  log p(μ)  =  log N(μ; 0, prior_sd_mu²)
//
//  We centre at 0 (equal chance of positive/negative drift).
//  prior_sd_mu = 0.5 allows drifts up to ±50% annualised — weakly informative.

inline double log_prior_mu(double mu, double prior_sd_mu = 0.5) {
    //  log N(x; 0, s²)  =  −½ log(2πs²)  −  x²/(2s²)
    return -0.5 * (LOG_2PI + 2.0 * std::log(prior_sd_mu)
                   + (mu / prior_sd_mu) * (mu / prior_sd_mu));
}

//  log p(σ)  =  log Half-Normal(σ; 0, prior_sd_sigma²)
//
//  Half-Normal = N(0, s²) truncated to σ > 0.
//  log p(σ)  =  log(2) + log N(σ; 0, s²)   for σ > 0
//            = −∞                            for σ ≤ 0
//
//  prior_sd_sigma = 0.5 puts 95% mass below σ ≈ 0.98 (i.e. below 98% vol),
//  which comfortably covers all equity markets.

inline double log_prior_sigma(double sigma, double prior_sd_sigma = 0.5) {
    if (sigma <= 0.0) return NEG_INF;
    return std::log(2.0)
           - 0.5 * (LOG_2PI + 2.0 * std::log(prior_sd_sigma)
                    + (sigma / prior_sd_sigma) * (sigma / prior_sd_sigma));
}

// ── Log-posterior ─────────────────────────────────────────────────────────────
//
//  log π(μ,σ | r)  =  log p(r | μ,σ)  +  log p(μ)  +  log p(σ)
//                      (constant dropped — MH only needs ratios)

inline double log_posterior(const std::vector<double>& log_returns,
                              double mu, double sigma, double dt,
                              double prior_sd_mu    = 0.5,
                              double prior_sd_sigma = 0.5) {
    double ll   = log_likelihood(log_returns, mu, sigma, dt);
    double lp_m = log_prior_mu(mu, prior_sd_mu);
    double lp_s = log_prior_sigma(sigma, prior_sd_sigma);
    return ll + lp_m + lp_s;
}

} // namespace gbm_bayes
