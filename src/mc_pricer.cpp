// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
#include "../include/utils.h"
using namespace Rcpp;

// ── Internal GBM terminal price sampler ──────────────────────────────────────

static NumericVector gbm_ST(double S0, double r, double sigma,
                             double T, int n_steps, int n_paths, int seed) {
    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    NumericVector ST(n_paths);
    for (int p = 0; p < n_paths; ++p) {
        double s = S0;
        for (int t = 0; t < n_steps; ++t)
            s *= std::exp(drift + vol * Z(engine));
        ST[p] = s;
    }
    return ST;
}

// ── Exported pricing functions ────────────────────────────────────────────────

//' Monte Carlo European Call option price
//'
//' @param S0      Current stock price
//' @param K       Strike price
//' @param r       Risk-free rate (annualised)
//' @param sigma   Volatility (annualised)
//' @param T       Time to maturity (years)
//' @param n_paths Number of Monte Carlo paths
//' @param n_steps Time steps per path
//' @param seed    Random seed
//' @return Named list: price, std_error, bs_price
//' @export
// [[Rcpp::export]]
List mc_call(double S0, double K, double r, double sigma, double T,
             int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericVector ST = gbm_ST(S0, r, sigma, T, n_steps, n_paths, seed);
    NumericVector payoffs(n_paths);
    for (int p = 0; p < n_paths; ++p)
        payoffs[p] = std::max(ST[p] - K, 0.0);

    double df       = std::exp(-r * T);
    double price    = df * mean(payoffs);
    double se       = df * sd(payoffs) / std::sqrt((double)n_paths);
    double bs_price = mcf::bs_call(S0, K, r, sigma, T);

    return List::create(
        Named("price")    = price,
        Named("std_error")= se,
        Named("bs_price") = bs_price,
        Named("error")    = std::abs(price - bs_price)
    );
}

//' Monte Carlo European Put option price
//'
//' @param S0      Current stock price
//' @param K       Strike price
//' @param r       Risk-free rate
//' @param sigma   Volatility
//' @param T       Time to maturity
//' @param n_paths Number of Monte Carlo paths
//' @param n_steps Time steps per path
//' @param seed    Random seed
//' @return Named list: price, std_error, bs_price
//' @export
// [[Rcpp::export]]
List mc_put(double S0, double K, double r, double sigma, double T,
            int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericVector ST = gbm_ST(S0, r, sigma, T, n_steps, n_paths, seed);
    NumericVector payoffs(n_paths);
    for (int p = 0; p < n_paths; ++p)
        payoffs[p] = std::max(K - ST[p], 0.0);

    double df       = std::exp(-r * T);
    double price    = df * mean(payoffs);
    double se       = df * sd(payoffs) / std::sqrt((double)n_paths);
    double bs_price = mcf::bs_put(S0, K, r, sigma, T);

    return List::create(
        Named("price")    = price,
        Named("std_error")= se,
        Named("bs_price") = bs_price,
        Named("error")    = std::abs(price - bs_price)
    );
}

//' Convergence study: price call for increasing n_paths
//'
//' @param S0    Stock price
//' @param K     Strike
//' @param r     Rate
//' @param sigma Vol
//' @param T     Maturity
//' @param path_sizes IntegerVector of path counts to test
//' @param seed  Seed
//' @return DataFrame with columns: n_paths, price, std_error
//' @export
// [[Rcpp::export]]
DataFrame mc_convergence(double S0, double K, double r, double sigma, double T,
                          IntegerVector path_sizes, int seed = 42) {
    int n = path_sizes.size();
    NumericVector prices(n), errors(n);

    for (int i = 0; i < n; ++i) {
        List res = mc_call(S0, K, r, sigma, T, path_sizes[i], 252, seed + i);
        prices[i] = as<double>(res["price"]);
        errors[i] = as<double>(res["std_error"]);
    }
    return DataFrame::create(
        Named("n_paths")   = path_sizes,
        Named("price")     = prices,
        Named("std_error") = errors
    );
}
