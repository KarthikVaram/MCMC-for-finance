// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <cmath>
#include "../include/utils.h"
using namespace Rcpp;

// ── OpenMP parallel European call ────────────────────────────────────────────

//' Parallel Monte Carlo European call using OpenMP
//'
//' Each thread gets its own seeded RNG to avoid data races.
//'
//' @param S0 K r sigma T n_paths n_steps seed n_threads (0 = auto)
//' @return Named list: price, std_error, bs_price, n_threads_used
//' @export
// [[Rcpp::export]]
List mc_parallel_call(double S0, double K, double r, double sigma, double T,
                       int n_paths = 1000000, int n_steps = 252,
                       int seed = 42, int n_threads = 0) {
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    int used_threads = omp_get_max_threads();
#else
    int used_threads = 1;
#endif

    double dt    = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);
    double df    = std::exp(-r * T);

    NumericVector payoffs(n_paths, 0.0);

#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_paths; ++p) {
        // Each path gets a unique seed derived from the master seed
        std::mt19937 engine(seed + p);
        std::normal_distribution<double> Z(0.0, 1.0);

        double s = S0;
        for (int t = 0; t < n_steps; ++t)
            s *= std::exp(drift + vol * Z(engine));
        payoffs[p] = std::max(s - K, 0.0);
    }

    double price = df * mean(payoffs);
    double se    = df * sd(payoffs) / std::sqrt((double)n_paths);
    double bs    = mcf::bs_call(S0, K, r, sigma, T);

    return List::create(
        Named("price")          = price,
        Named("std_error")      = se,
        Named("bs_price")       = bs,
        Named("n_threads_used") = used_threads
    );
}

//' Benchmark single-threaded vs multi-threaded runtime
//'
//' Returns a DataFrame with thread count and elapsed time for each run.
//'
//' @param S0 K r sigma T n_paths n_steps seed
//' @param thread_counts IntegerVector of thread counts to test
//' @return DataFrame: n_threads, elapsed_sec, price
//' @export
// [[Rcpp::export]]
DataFrame parallel_benchmark(double S0, double K, double r, double sigma, double T,
                               int n_paths = 500000, int n_steps = 252, int seed = 42,
                               IntegerVector thread_counts = IntegerVector::create(1, 2, 4, 8)) {
    int n = thread_counts.size();
    NumericVector times(n), prices(n);

    for (int i = 0; i < n; ++i) {
        double t0 = (double)clock() / CLOCKS_PER_SEC;
        List res  = mc_parallel_call(S0, K, r, sigma, T, n_paths, n_steps, seed, thread_counts[i]);
        double t1 = (double)clock() / CLOCKS_PER_SEC;
        times[i]  = t1 - t0;
        prices[i] = as<double>(res["price"]);
    }

    return DataFrame::create(
        Named("n_threads")   = thread_counts,
        Named("elapsed_sec") = times,
        Named("price")       = prices
    );
}
