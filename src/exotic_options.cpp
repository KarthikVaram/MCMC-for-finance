// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
using namespace Rcpp;

// ── Shared path generator (returns full n_paths x n_steps matrix) ─────────

static NumericMatrix gen_paths(double S0, double r, double sigma,
                                double T, int n_steps, int n_paths, int seed) {
    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);

    std::mt19937 engine(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    NumericMatrix S(n_paths, n_steps + 1);
    for (int p = 0; p < n_paths; ++p) {
        S(p, 0) = S0;
        for (int t = 1; t <= n_steps; ++t)
            S(p, t) = S(p, t - 1) * std::exp(drift + vol * Z(engine));
    }
    return S;
}

// ── 1. Asian Options ─────────────────────────────────────────────────────────

//' Price an Asian arithmetic-average call option
//'
//' Payoff = max(mean(S_t) - K, 0)
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @return Named list: price, std_error
//' @export
// [[Rcpp::export]]
List asian_call(double S0, double K, double r, double sigma, double T,
                int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double avg = 0.0;
        for (int t = 1; t <= n_steps; ++t)
            avg += S(p, t);
        avg /= n_steps;
        payoffs[p] = std::max(avg - K, 0.0);
    }
    return List::create(
        Named("price")    = df * mean(payoffs),
        Named("std_error")= df * sd(payoffs) / std::sqrt((double)n_paths)
    );
}

//' Price an Asian arithmetic-average put option
//'
//' Payoff = max(K - mean(S_t), 0)
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @return Named list: price, std_error
//' @export
// [[Rcpp::export]]
List asian_put(double S0, double K, double r, double sigma, double T,
               int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double avg = 0.0;
        for (int t = 1; t <= n_steps; ++t)
            avg += S(p, t);
        avg /= n_steps;
        payoffs[p] = std::max(K - avg, 0.0);
    }
    return List::create(
        Named("price")    = df * mean(payoffs),
        Named("std_error")= df * sd(payoffs) / std::sqrt((double)n_paths)
    );
}

// ── 2. Barrier Options ───────────────────────────────────────────────────────

//' Price a Down-and-Out European call (barrier option)
//'
//' The option is knocked out if S_t ever touches or crosses the lower barrier H.
//' Payoff = max(S_T - K, 0) * 1{min(S_t) > H}
//'
//' @param S0 K H r sigma T n_paths n_steps seed
//' @return Named list: price, std_error, pct_knocked_out
//' @export
// [[Rcpp::export]]
List barrier_down_out_call(double S0, double K, double H,
                            double r, double sigma, double T,
                            int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);
    int knocked = 0;

    for (int p = 0; p < n_paths; ++p) {
        bool out = false;
        for (int t = 0; t <= n_steps; ++t) {
            if (S(p, t) <= H) { out = true; break; }
        }
        if (out) {
            ++knocked;
            payoffs[p] = 0.0;
        } else {
            payoffs[p] = std::max(S(p, n_steps) - K, 0.0);
        }
    }

    return List::create(
        Named("price")          = df * mean(payoffs),
        Named("std_error")      = df * sd(payoffs) / std::sqrt((double)n_paths),
        Named("pct_knocked_out")= 100.0 * knocked / n_paths
    );
}

//' Price an Up-and-Out European call (barrier option)
//'
//' The option is knocked out if S_t touches or crosses the upper barrier H.
//' Payoff = max(S_T - K, 0) * 1{max(S_t) < H}
//'
//' @param S0 K H r sigma T n_paths n_steps seed
//' @return Named list: price, std_error, pct_knocked_out
//' @export
// [[Rcpp::export]]
List barrier_up_out_call(double S0, double K, double H,
                          double r, double sigma, double T,
                          int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);
    int knocked = 0;

    for (int p = 0; p < n_paths; ++p) {
        bool out = false;
        for (int t = 0; t <= n_steps; ++t) {
            if (S(p, t) >= H) { out = true; break; }
        }
        if (out) {
            ++knocked;
            payoffs[p] = 0.0;
        } else {
            payoffs[p] = std::max(S(p, n_steps) - K, 0.0);
        }
    }

    return List::create(
        Named("price")          = df * mean(payoffs),
        Named("std_error")      = df * sd(payoffs) / std::sqrt((double)n_paths),
        Named("pct_knocked_out")= 100.0 * knocked / n_paths
    );
}

// ── 3. Lookback Options ──────────────────────────────────────────────────────

//' Price a floating-strike lookback call
//'
//' Payoff = S_T - min(S_t)
//'
//' @param S0 r sigma T n_paths n_steps seed
//' @return Named list: price, std_error
//' @export
// [[Rcpp::export]]
List lookback_call(double S0, double r, double sigma, double T,
                   int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double S_min = S(p, 0);
        for (int t = 1; t <= n_steps; ++t)
            S_min = std::min(S_min, S(p, t));
        payoffs[p] = S(p, n_steps) - S_min;
    }
    return List::create(
        Named("price")    = df * mean(payoffs),
        Named("std_error")= df * sd(payoffs) / std::sqrt((double)n_paths)
    );
}

//' Price a floating-strike lookback put
//'
//' Payoff = max(S_t) - S_T
//'
//' @param S0 r sigma T n_paths n_steps seed
//' @return Named list: price, std_error
//' @export
// [[Rcpp::export]]
List lookback_put(double S0, double r, double sigma, double T,
                  int n_paths = 100000, int n_steps = 252, int seed = 42) {
    NumericMatrix S = gen_paths(S0, r, sigma, T, n_steps, n_paths, seed);
    double df = std::exp(-r * T);
    NumericVector payoffs(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double S_max = S(p, 0);
        for (int t = 1; t <= n_steps; ++t)
            S_max = std::max(S_max, S(p, t));
        payoffs[p] = S_max - S(p, n_steps);
    }
    return List::create(
        Named("price")    = df * mean(payoffs),
        Named("std_error")= df * sd(payoffs) / std::sqrt((double)n_paths)
    );
}
