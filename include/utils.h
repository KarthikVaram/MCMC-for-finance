#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>

namespace mcf {

// Cholesky decomposition of a 2x2 correlation matrix
// Returns lower triangular factor L such that L * L^T = [[1, rho],[rho, 1]]
inline std::pair<double,double> chol2x2(double rho) {
    // L = [[1, 0], [rho, sqrt(1 - rho^2)]]
    return {rho, std::sqrt(1.0 - rho * rho)};
}

// Black-Scholes European call price
inline double bs_call(double S, double K, double r, double sigma, double T) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    // Standard normal CDF via erfc
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

// Black-Scholes European put price
inline double bs_put(double S, double K, double r, double sigma, double T) {
    return bs_call(S, K, r, sigma, T) - S + K * std::exp(-r * T);
}

} // namespace mcf
