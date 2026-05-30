// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
#include <algorithm>
#include "../include/utils.h"
using namespace Rcpp;

// ─────────────────────────────────────────────────────────────────────────────
//  Heston model:
//    dS = mu * S * dt + sqrt(v) * S * dW^S
//    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW^v
//    dW^S dW^v = rho * dt
//
//  Discretisation: Euler-Maruyama with full truncation (v_t = max(v_t, 0))
// ─────────────────────────────────────────────────────────────────────────────

//' Simulate Heston stock price paths
//'
//' @param S0    Initial stock price
//' @param v0    Initial variance
//' @param mu    Drift of stock price
//' @param kappa Mean-reversion speed of variance
//' @param theta Long-run variance level
//' @param xi    Vol-of-vol
//' @param rho   Correlation between the two Brownian motions
//' @param T     Time horizon
//' @param n_steps Time steps
//' @param n_paths Number of paths
//' @param seed  Random seed
//' @return List of two matrices: S (prices) and V (variances), each n_paths x (n_steps+1)
//' @export
// [[Rcpp::export]]
List heston_paths(double S0, double v0,
                  double mu, double kappa, double theta, double xi, double rho,
                  double T, int n_steps, int n_paths, int seed = 42) {
    double dt    = T / n_steps;
    double sqdt  = std::sqrt(dt);
    double rho2  = std::sqrt(1.0 - rho * rho);   // for Cholesky

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    NumericMatrix S(n_paths, n_steps + 1);
    NumericMatrix V(n_paths, n_steps + 1);

    for (int p = 0; p < n_paths; ++p) {
        S(p, 0) = S0;
        V(p, 0) = v0;

        for (int t = 1; t <= n_steps; ++t) {
            double z1 = Z(engine);
            double z2 = Z(engine);

            // Correlated Brownian increments via Cholesky
            double dW_S = sqdt * z1;
            double dW_v = sqdt * (rho * z1 + rho2 * z2);

            double v_prev = std::max(V(p, t - 1), 0.0);  // full truncation
            double sv     = std::sqrt(v_prev);

            // Variance process (Euler-Maruyama, full truncation)
            V(p, t) = v_prev + kappa * (theta - v_prev) * dt + xi * sv * dW_v;
            V(p, t) = std::max(V(p, t), 0.0);             // truncate negative variance

            // Stock price process (exact log-Euler)
            S(p, t) = S(p, t - 1) * std::exp((mu - 0.5 * v_prev) * dt + sv * dW_S);
        }
    }

    return List::create(Named("S") = S, Named("V") = V);
}

//' Price a European call under the Heston model
//'
//' @param S0 v0 K r kappa theta xi rho T n_paths n_steps seed
//' @return Named list: price, std_error, bs_equiv_price (using sqrt(theta) as vol)
//' @export
// [[Rcpp::export]]
List heston_call(double S0, double v0, double K,
                 double r, double kappa, double theta, double xi, double rho,
                 double T, int n_paths = 100000, int n_steps = 252, int seed = 42) {
    List paths = heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed);
    NumericMatrix S = as<NumericMatrix>(paths["S"]);

    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);
    for (int p = 0; p < n_paths; ++p)
        payoffs[p] = std::max(S(p, n_steps) - K, 0.0);

    double price = df * mean(payoffs);
    double se    = df * sd(payoffs) / std::sqrt((double)n_paths);
    double bs    = mcf::bs_call(S0, K, r, std::sqrt(theta), T);  // BS with long-run vol

    return List::create(
        Named("price")         = price,
        Named("std_error")     = se,
        Named("bs_equiv_price")= bs
    );
}

//' Price a European put under the Heston model
//'
//' @param S0 v0 K r kappa theta xi rho T n_paths n_steps seed
//' @return Named list: price, std_error
//' @export
// [[Rcpp::export]]
List heston_put(double S0, double v0, double K,
                double r, double kappa, double theta, double xi, double rho,
                double T, int n_paths = 100000, int n_steps = 252, int seed = 42) {
    List paths = heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed);
    NumericMatrix S = as<NumericMatrix>(paths["S"]);

    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);
    for (int p = 0; p < n_paths; ++p)
        payoffs[p] = std::max(K - S(p, n_steps), 0.0);

    double price = df * mean(payoffs);
    double se    = df * sd(payoffs) / std::sqrt((double)n_paths);

    return List::create(Named("price") = price, Named("std_error") = se);
}

//' Heston implied volatility smile
//'
//' Prices calls across a range of strikes and returns them.
//'
//' @param S0 v0 r kappa theta xi rho T n_paths n_steps seed
//' @param strikes NumericVector of strike prices
//' @return DataFrame: strike, price, std_error
//' @export
// [[Rcpp::export]]
DataFrame heston_smile(double S0, double v0,
                        double r, double kappa, double theta, double xi, double rho,
                        double T, NumericVector strikes,
                        int n_paths = 100000, int n_steps = 252, int seed = 42) {
    // Generate paths once and reuse
    List paths = heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed);
    NumericMatrix S = as<NumericMatrix>(paths["S"]);

    int nK = strikes.size();
    NumericVector prices(nK), ses(nK);
    double df = std::exp(-r * T);

    for (int k = 0; k < nK; ++k) {
        double K = strikes[k];
        NumericVector payoffs(n_paths);
        for (int p = 0; p < n_paths; ++p)
            payoffs[p] = std::max(S(p, n_steps) - K, 0.0);
        prices[k] = df * mean(payoffs);
        ses[k]    = df * sd(payoffs) / std::sqrt((double)n_paths);
    }

    return DataFrame::create(
        Named("strike")    = strikes,
        Named("price")     = prices,
        Named("std_error") = ses
    );
}
