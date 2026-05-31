#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace mcmc {

// ── Effective Sample Size (ESS) ───────────────────────────────────────────────
// Uses autocorrelation summing: ESS = N / (1 + 2 * sum_k rho_k)
// Stops at first negative autocorrelation (Geyer's monotone rule).
inline double ess(const std::vector<double>& chain) {
    int n = chain.size();
    if (n < 4) return (double)n;

    double mean_val = 0.0;
    for (double x : chain) mean_val += x;
    mean_val /= n;

    double var_val = 0.0;
    for (double x : chain) var_val += (x - mean_val) * (x - mean_val);
    var_val /= n;
    if (var_val < 1e-14) return (double)n;

    double rho_sum = 0.0;
    for (int lag = 1; lag < n / 2; ++lag) {
        double rho = 0.0;
        for (int t = 0; t < n - lag; ++t)
            rho += (chain[t] - mean_val) * (chain[t + lag] - mean_val);
        rho /= (n * var_val);
        if (rho < 0.0) break;
        rho_sum += rho;
    }
    return (double)n / (1.0 + 2.0 * rho_sum);
}

// ── Gelman-Rubin R-hat (potential scale reduction factor) ────────────────────
// Requires at least 2 chains. Values < 1.1 indicate convergence.
inline double r_hat(const std::vector<std::vector<double>>& chains) {
    int m = chains.size();       // number of chains
    int n = chains[0].size();    // length of each chain
    if (m < 2 || n < 2) throw std::runtime_error("r_hat needs >= 2 chains of length >= 2");

    // Chain means
    std::vector<double> chain_mean(m, 0.0);
    for (int j = 0; j < m; ++j) {
        for (double x : chains[j]) chain_mean[j] += x;
        chain_mean[j] /= n;
    }

    // Grand mean
    double grand_mean = 0.0;
    for (double cm : chain_mean) grand_mean += cm;
    grand_mean /= m;

    // Between-chain variance B
    double B = 0.0;
    for (int j = 0; j < m; ++j)
        B += (chain_mean[j] - grand_mean) * (chain_mean[j] - grand_mean);
    B *= (double)n / (m - 1);

    // Within-chain variance W
    double W = 0.0;
    for (int j = 0; j < m; ++j) {
        double s2 = 0.0;
        for (double x : chains[j])
            s2 += (x - chain_mean[j]) * (x - chain_mean[j]);
        W += s2 / (n - 1);
    }
    W /= m;

    double var_plus = ((double)(n - 1) / n) * W + B / n;
    return std::sqrt(var_plus / W);
}

// ── Posterior summary statistics ──────────────────────────────────────────────
struct PosteriorSummary {
    double mean;
    double sd;
    double median;
    double q025;   // 2.5th percentile
    double q975;   // 97.5th percentile
    double ess_val;
};

inline PosteriorSummary summarise(const std::vector<double>& chain) {
    int n = chain.size();
    double m = 0.0;
    for (double x : chain) m += x;
    m /= n;

    double s = 0.0;
    for (double x : chain) s += (x - m) * (x - m);
    s = std::sqrt(s / (n - 1));

    std::vector<double> sorted = chain;
    std::sort(sorted.begin(), sorted.end());

    auto quantile = [&](double p) {
        double idx = p * (n - 1);
        int lo = (int)idx;
        int hi = lo + 1;
        if (hi >= n) return sorted[n - 1];
        return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
    };

    return {m, s, quantile(0.5), quantile(0.025), quantile(0.975), ess(chain)};
}

// ── Acceptance rate from binary accept vector ─────────────────────────────────
inline double acceptance_rate(const std::vector<int>& accepted) {
    if (accepted.empty()) return 0.0;
    double sum = 0.0;
    for (int a : accepted) sum += a;
    return sum / accepted.size();
}

} // namespace mcmc
