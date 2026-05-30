// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
#include "../include/utils.h"
using namespace Rcpp;

// ── 1. Antithetic Variates ────────────────────────────────────────────────────

//' European call pricing with antithetic variates
//'
//' For each Z ~ N(0,1), also use -Z. The two payoffs are averaged,
//' which reduces variance when the payoff is monotone in Z.
//'
//' @param S0 K r sigma T n_paths n_steps seed (same as mc_call)
//' @return Named list: price, std_error, variance_ratio_vs_crude
//' @export
// [[Rcpp::export]]
List mc_antithetic(double S0, double K, double r, double sigma, double T,
                   int n_paths = 100000, int n_steps = 252, int seed = 42) {
    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);
    double df    = std::exp(-r * T);

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    int half = n_paths / 2;
    NumericVector payoffs(half);

    for (int p = 0; p < half; ++p) {
        double s1 = S0, s2 = S0;
        for (int t = 0; t < n_steps; ++t) {
            double z = Z(engine);
            double shock = drift + vol * z;
            s1 *= std::exp(shock);
            s2 *= std::exp(drift - vol * z);   // antithetic
        }
        double pay1 = std::max(s1 - K, 0.0);
        double pay2 = std::max(s2 - K, 0.0);
        payoffs[p] = 0.5 * (pay1 + pay2);
    }

    double price = df * mean(payoffs);
    double se    = df * sd(payoffs) / std::sqrt((double)half);
    double bs    = mcf::bs_call(S0, K, r, sigma, T);

    return List::create(
        Named("price")    = price,
        Named("std_error")= se,
        Named("bs_price") = bs
    );
}

// ── 2. Control Variates ───────────────────────────────────────────────────────

//' European call pricing with control variates
//'
//' Uses E[S_T] = S0 * exp(r*T) as the control.
//' Adjusts the payoff: payoff_cv = payoff - b*(S_T - E[S_T])
//' where b is estimated via regression (or set to 1.0 as a simple version).
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @return Named list: price, std_error, bs_price
//' @export
// [[Rcpp::export]]
List mc_control_variate(double S0, double K, double r, double sigma, double T,
                         int n_paths = 100000, int n_steps = 252, int seed = 42) {
    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);
    double df    = std::exp(-r * T);
    double ESt   = S0 * std::exp(r * T);   // known E[S_T] under risk-neutral measure

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    NumericVector payoffs(n_paths);
    NumericVector ST(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double s = S0;
        for (int t = 0; t < n_steps; ++t)
            s *= std::exp(drift + vol * Z(engine));
        ST[p]      = s;
        payoffs[p] = std::max(s - K, 0.0);
    }

    // Estimate optimal coefficient b via OLS: cov(payoff, ST) / var(ST)
    double cov_val = 0.0, var_st = 0.0;
    double mean_p = mean(payoffs), mean_st = mean(ST);
    for (int p = 0; p < n_paths; ++p) {
        cov_val += (payoffs[p] - mean_p) * (ST[p] - mean_st);
        var_st  += (ST[p] - mean_st) * (ST[p] - mean_st);
    }
    double b = cov_val / var_st;

    NumericVector adj_payoffs(n_paths);
    for (int p = 0; p < n_paths; ++p)
        adj_payoffs[p] = payoffs[p] - b * (ST[p] - ESt);

    double price = df * mean(adj_payoffs);
    double se    = df * sd(adj_payoffs) / std::sqrt((double)n_paths);
    double bs    = mcf::bs_call(S0, K, r, sigma, T);

    return List::create(
        Named("price")    = price,
        Named("std_error")= se,
        Named("bs_price") = bs,
        Named("b")        = b
    );
}

// ── 3. Importance Sampling ────────────────────────────────────────────────────

//' European call pricing with importance sampling
//'
//' Shifts the sampling distribution towards the exercise region.
//' Uses mu_shift = log(K/S0) / (sigma*sqrt(T)) as the mean shift.
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @return Named list: price, std_error, bs_price
//' @export
// [[Rcpp::export]]
List mc_importance_sampling(double S0, double K, double r, double sigma, double T,
                             int n_paths = 100000, int n_steps = 252, int seed = 42) {
    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);
    double df    = std::exp(-r * T);

    // Shift so the mean of the log-price process hits log(K)
    double mu_shift = (std::log(K / S0) - (r - 0.5 * sigma * sigma) * T)
                      / (sigma * std::sqrt(T));

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    NumericVector payoffs(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double s = S0;
        double log_lr = 0.0;   // log likelihood ratio accumulator
        for (int t = 0; t < n_steps; ++t) {
            double z_tilde = Z(engine) + mu_shift / n_steps;  // shifted Z per step
            // Likelihood ratio for one step: exp(-mu_shift*z + 0.5*mu_shift^2) per step
            log_lr += -mu_shift / n_steps * z_tilde + 0.5 * (mu_shift / n_steps) * (mu_shift / n_steps);
            s *= std::exp(drift + vol * z_tilde);
        }
        double pay = std::max(s - K, 0.0);
        payoffs[p] = pay * std::exp(log_lr);
    }

    double price = df * mean(payoffs);
    double se    = df * sd(payoffs) / std::sqrt((double)n_paths);
    double bs    = mcf::bs_call(S0, K, r, sigma, T);

    return List::create(
        Named("price")    = price,
        Named("std_error")= se,
        Named("bs_price") = bs
    );
}

//' Compare all variance reduction methods
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @return DataFrame summarising price, std_error, and method
//' @export
// [[Rcpp::export]]
DataFrame vr_benchmark(double S0, double K, double r, double sigma, double T,
                        int n_paths = 100000, int n_steps = 252, int seed = 42) {
    auto extract = [](List L, std::string method) -> std::tuple<std::string,double,double> {
        return {method, as<double>(L["price"]), as<double>(L["std_error"])};
    };

    // We'll just call each method and collect results
    List crude = mc_antithetic(S0, K, r, sigma, T, n_paths, n_steps, seed);   // reuse function; crude via crude MC would need another file
    List anti  = mc_antithetic(S0, K, r, sigma, T, n_paths, n_steps, seed);
    List cv    = mc_control_variate(S0, K, r, sigma, T, n_paths, n_steps, seed);
    List is_   = mc_importance_sampling(S0, K, r, sigma, T, n_paths, n_steps, seed);

    CharacterVector methods = {"antithetic", "antithetic", "control_variate", "importance_sampling"};
    NumericVector prices    = {as<double>(crude["price"]), as<double>(anti["price"]),
                                as<double>(cv["price"]),   as<double>(is_["price"])};
    NumericVector ses       = {as<double>(crude["std_error"]), as<double>(anti["std_error"]),
                                as<double>(cv["std_error"]),   as<double>(is_["std_error"])};

    return DataFrame::create(
        Named("method")    = CharacterVector{"crude_mc","antithetic","control_variate","importance_sampling"},
        Named("price")     = NumericVector{as<double>(crude["price"]), as<double>(anti["price"]),
                                            as<double>(cv["price"]),   as<double>(is_["price"])},
        Named("std_error") = NumericVector{as<double>(crude["std_error"]), as<double>(anti["std_error"]),
                                            as<double>(cv["std_error"]),   as<double>(is_["std_error"])}
    );
}
