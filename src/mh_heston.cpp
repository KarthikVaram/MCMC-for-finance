// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../include/likelihoods.h"
#include "../include/diagnostics.h"
using namespace Rcpp;

// ── Heston Euler path: terminal variance and log-price ────────────────────────
struct HestonTerminal { double log_S; double v_T; };

static HestonTerminal heston_step(double log_S0, double v0,
                                   double r, double kappa, double theta,
                                   double xi, double rho,
                                   double dt, int n_steps,
                                   std::mt19937& engine) {
    std::normal_distribution<double> Z(0.0, 1.0);
    double sqdt  = std::sqrt(dt);
    double rho2  = std::sqrt(std::max(1.0 - rho * rho, 0.0));
    double log_S = log_S0;
    double v     = v0;

    for (int t = 0; t < n_steps; ++t) {
        double z1 = Z(engine), z2 = Z(engine);
        double dW_S = sqdt * z1;
        double dW_v = sqdt * (rho * z1 + rho2 * z2);
        double sv   = std::sqrt(std::max(v, 0.0));

        log_S += (r - 0.5 * v) * dt + sv * dW_S;
        v     += kappa * (theta - v) * dt + xi * sv * dW_v;
        v      = std::max(v, 0.0);
    }
    return {log_S, v};
}

// ── Simulated likelihood: E[payoff] from N_sim paths, for ONE observed price ──
static double heston_sim_price(double S0, double K, double r,
                                double kappa, double theta, double xi, double rho,
                                double v0, double T, int n_steps, int n_sim,
                                std::mt19937& engine) {
    double dt = T / n_steps;
    double sum = 0.0;
    double log_S0 = std::log(S0);
    for (int p = 0; p < n_sim; ++p) {
        auto res = heston_step(log_S0, v0, r, kappa, theta, xi, rho, dt, n_steps, engine);
        sum += std::max(std::exp(res.log_S) - K, 0.0);
    }
    return std::exp(-r * T) * sum / n_sim;
}

// ── Log-prior for Heston theta = (kappa, theta_v, xi, rho, v0) ───────────────
static double log_prior_heston(double kappa, double theta_v, double xi,
                                 double rho, double v0) {
    // kappa ~ Gamma(2, 0.5) — encourages kappa > 0, mean = 4
    double lp_kappa  = mcmc::log_prior_gamma(kappa, 2.0, 0.5);
    // theta_v ~ HN(0.2) — long-run vol around 20%
    double lp_theta  = mcmc::log_prior_half_normal(std::sqrt(theta_v), 0.2);
    // xi ~ HN(0.5)
    double lp_xi     = mcmc::log_prior_half_normal(xi, 0.5);
    // rho ~ scaled Beta(2,2) on (-1,1) — mild preference for -0.5 < rho < 0
    double lp_rho    = mcmc::log_prior_rho(rho, 2.0, 2.0);
    // v0 ~ HN(0.2)
    double lp_v0     = mcmc::log_prior_half_normal(std::sqrt(v0), 0.2);
    return lp_kappa + lp_theta + lp_xi + lp_rho + lp_v0;
}

//' Metropolis-Hastings calibration of the Heston model to observed option prices
//'
//' Uses simulated likelihood (particle filter approximation) to evaluate
//' p(observed_prices | kappa, theta, xi, rho, v0).
//'
//' @param obs_prices  NumericVector of observed call prices
//' @param strikes     NumericVector of corresponding strikes
//' @param maturities  NumericVector of maturities (years)
//' @param S0          Current stock price
//' @param r           Risk-free rate
//' @param sigma_obs   Observation noise (pricing error std dev)
//' @param n_iter      MCMC iterations
//' @param burnin      Burn-in iterations
//' @param n_sim       Paths per likelihood evaluation (higher = more accurate but slower)
//' @param n_steps     Time steps per path
//' @param seed        Random seed
//' @return List: chains for each parameter, summaries, acceptance rate
//' @export
// [[Rcpp::export]]
List mh_heston(NumericVector obs_prices,
                NumericVector strikes,
                NumericVector maturities,
                double S0,
                double r         = 0.05,
                double sigma_obs = 0.5,
                int    n_iter    = 10000,
                int    burnin    = 2000,
                int    n_sim     = 200,
                int    n_steps   = 50,
                int    seed      = 42) {

    int n_obs = obs_prices.size();
    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<double> obs(obs_prices.begin(), obs_prices.end());
    std::vector<double> K(strikes.begin(), strikes.end());
    std::vector<double> T_mat(maturities.begin(), maturities.end());

    // Current parameter state
    double kappa_c  = 2.0, theta_c = 0.04, xi_c = 0.30;
    double rho_c    = -0.50, v0_c = 0.04;

    // Proposal standard deviations (tunable)
    double sd_kappa = 0.10, sd_theta = 0.005, sd_xi  = 0.02;
    double sd_rho   = 0.05, sd_v0   = 0.005;

    // Evaluate initial log-posterior
    auto sim_prices = [&](double ka, double th, double xi,
                           double rho, double v0) {
        std::vector<double> mp(n_obs);
        for (int i = 0; i < n_obs; ++i)
            mp[i] = heston_sim_price(S0, K[i], r, ka, th, xi, rho, v0,
                                      T_mat[i], n_steps, n_sim, engine);
        return mp;
    };

    auto log_post = [&](double ka, double th, double xi, double rho, double v0) {
        double lprior = log_prior_heston(ka, th, xi, rho, v0);
        if (!std::isfinite(lprior)) return mcmc::NEG_INF;
        auto mp = sim_prices(ka, th, xi, rho, v0);
        double ll = mcmc::log_likelihood_gaussian(obs, mp, sigma_obs);
        return ll + lprior;
    };

    double lp_cur = log_post(kappa_c, theta_c, xi_c, rho_c, v0_c);

    int keep = n_iter - burnin;
    std::vector<double> kappa_ch(keep), theta_ch(keep), xi_ch(keep);
    std::vector<double> rho_ch(keep), v0_ch(keep), lp_ch(keep);
    std::vector<int>    accepted(n_iter, 0);

    for (int i = 0; i < n_iter; ++i) {
        // Random-walk proposals for each parameter
        double kappa_p = kappa_c + sd_kappa * Z(engine);
        double theta_p = theta_c + sd_theta * Z(engine);
        double xi_p    = xi_c    + sd_xi    * Z(engine);
        double rho_p   = rho_c   + sd_rho   * Z(engine);
        double v0_p    = v0_c    + sd_v0    * Z(engine);

        // All parameters have positivity/range constraints
        if (kappa_p > 0 && theta_p > 0 && xi_p > 0 &&
            rho_p > -1 && rho_p < 1 && v0_p > 0) {

            double lp_prop = log_post(kappa_p, theta_p, xi_p, rho_p, v0_p);
            if (std::log(U(engine)) < lp_prop - lp_cur) {
                kappa_c = kappa_p; theta_c = theta_p; xi_c = xi_p;
                rho_c   = rho_p;   v0_c    = v0_p;    lp_cur  = lp_prop;
                accepted[i] = 1;
            }
        }

        if (i >= burnin) {
            int k = i - burnin;
            kappa_ch[k] = kappa_c; theta_ch[k] = theta_c; xi_ch[k] = xi_c;
            rho_ch[k]   = rho_c;   v0_ch[k]    = v0_c;    lp_ch[k] = lp_cur;
        }
    }

    // Summaries
    auto sk = mcmc::summarise(kappa_ch); auto st = mcmc::summarise(theta_ch);
    auto sx = mcmc::summarise(xi_ch);    auto sr = mcmc::summarise(rho_ch);
    auto sv = mcmc::summarise(v0_ch);

    DataFrame summary_df = DataFrame::create(
        Named("parameter") = CharacterVector{"kappa","theta","xi","rho","v0"},
        Named("mean")   = NumericVector{sk.mean,   st.mean,   sx.mean,   sr.mean,   sv.mean},
        Named("sd")     = NumericVector{sk.sd,     st.sd,     sx.sd,     sr.sd,     sv.sd},
        Named("q025")   = NumericVector{sk.q025,   st.q025,   sx.q025,   sr.q025,   sv.q025},
        Named("median") = NumericVector{sk.median, st.median, sx.median, sr.median, sv.median},
        Named("q975")   = NumericVector{sk.q975,   st.q975,   sx.q975,   sr.q975,   sv.q975},
        Named("ess")    = NumericVector{sk.ess_val, st.ess_val, sx.ess_val, sr.ess_val, sv.ess_val}
    );

    return List::create(
        Named("kappa_chain") = kappa_ch,
        Named("theta_chain") = theta_ch,
        Named("xi_chain")    = xi_ch,
        Named("rho_chain")   = rho_ch,
        Named("v0_chain")    = v0_ch,
        Named("lp_chain")    = lp_ch,
        Named("accept_rate") = mcmc::acceptance_rate(accepted),
        Named("summary")     = summary_df
    );
}
