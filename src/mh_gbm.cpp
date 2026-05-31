// [[Rcpp::depends(Rcpp)]]
// =============================================================================
//  mh_gbm.cpp  —  Metropolis-Hastings sampler for GBM parameters
//
//  METROPOLIS-HASTINGS ALGORITHM (random-walk variant)
//  ────────────────────────────────────────────────────
//  Goal: draw samples from π(θ | data) without knowing its normalising constant.
//
//  At iteration i, current state θ = (μ, σ):
//
//    1. PROPOSE:
//         θ*  ~  q(θ* | θ)  =  N(θ, Σ_prop)
//
//       Here we use a diagonal Gaussian proposal — each parameter is
//       perturbed independently:
//           μ*  =  μ  +  ε_μ,    ε_μ  ~ N(0, s_μ²)
//           σ*  =  σ  +  ε_σ,    ε_σ  ~ N(0, s_σ²)
//
//       The proposal is symmetric: q(θ*|θ) = q(θ|θ*),
//       so the Hastings ratio cancels and we only need the posterior ratio.
//
//    2. ACCEPTANCE PROBABILITY:
//
//         α  =  min(1,  π(θ* | data) / π(θ | data) )
//
//       In log space (more numerically stable):
//
//         log α  =  log π(θ*|data)  −  log π(θ|data)
//
//       If log α ≥ 0 ⟹  α = 1 (always accept).
//       If log α < 0 ⟹  accept with probability exp(log α).
//
//    3. UPDATE:
//         u ~ Uniform(0, 1)
//         if log(u) < log α:
//             θ ← θ*          (accept)
//         else:
//             θ ← θ           (reject — stay, duplicate current state)
//
//    4. After burn-in, store θ_i in the chain.
//
//  PROPOSAL TUNING
//  ───────────────
//  Optimal acceptance rate for a 2D Gaussian target ≈ 44% (Roberts et al.)
//  If acceptance too high → proposals too small → slow mixing.
//  If acceptance too low  → proposals too large → nearly always rejected.
//
//  BURN-IN
//  ───────
//  First `burnin` samples discarded — chain hasn't reached stationary dist yet.
// =============================================================================

#include <Rcpp.h>
#include <random>
#include <cmath>
#include <vector>
#include "../include/gbm_bayes.h"
#include "../include/diagnostics.h"
using namespace Rcpp;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helper: run one MH chain
// ─────────────────────────────────────────────────────────────────────────────

struct MHChain {
    std::vector<double> mu, sigma, log_post;
    double accept_rate;
};

static MHChain run_mh_chain(const std::vector<double>& log_returns,
                              double dt,
                              int n_iter, int burnin,
                              double prop_sd_mu, double prop_sd_sigma,
                              double mu_init, double sigma_init,
                              double prior_sd_mu, double prior_sd_sigma,
                              int seed) {
    std::mt19937 engine(seed);
    std::normal_distribution<double>       Z(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    double mu_cur    = mu_init;
    double sigma_cur = sigma_init;
    double lp_cur    = gbm_bayes::log_posterior(log_returns, mu_cur, sigma_cur,
                                                 dt, prior_sd_mu, prior_sd_sigma);

    int keep = n_iter - burnin;
    MHChain chain;
    chain.mu.resize(keep);
    chain.sigma.resize(keep);
    chain.log_post.resize(keep);

    int n_accepted = 0;

    for (int i = 0; i < n_iter; ++i) {

        // Step 1: Propose
        double mu_prop    = mu_cur    + prop_sd_mu    * Z(engine);
        double sigma_prop = sigma_cur + prop_sd_sigma * Z(engine);

        // Step 2: Compute log acceptance ratio
        if (sigma_prop > 0.0) {
            double lp_prop = gbm_bayes::log_posterior(log_returns, mu_prop,
                                                       sigma_prop, dt,
                                                       prior_sd_mu, prior_sd_sigma);
            // log α = log π(θ_prop) − log π(θ_cur)
            // Proposal is symmetric so Hastings ratio = 1 (drops out).
            double log_alpha = lp_prop - lp_cur;

            // Step 3: Accept / reject
            // Draw u ~ U(0,1). Accept iff log(u) < log(α).
            if (std::log(U(engine)) < log_alpha) {
                mu_cur    = mu_prop;
                sigma_cur = sigma_prop;
                lp_cur    = lp_prop;
                ++n_accepted;
            }
        }
        // sigma_prop <= 0: prior = -inf, always reject.

        // Step 4: Store after burn-in
        if (i >= burnin) {
            int k = i - burnin;
            chain.mu[k]       = mu_cur;
            chain.sigma[k]    = sigma_cur;
            chain.log_post[k] = lp_cur;
        }
    }

    chain.accept_rate = (double)n_accepted / n_iter;
    return chain;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Exported: single-chain MH sampler
// ─────────────────────────────────────────────────────────────────────────────

//' Metropolis-Hastings sampler for GBM parameters (mu, sigma)
//'
//' Draws from the posterior p(mu, sigma | log_returns) via random-walk MH.
//'
//' The GBM log-return likelihood is:
//'   r_t | mu, sigma ~ N( (mu - sigma^2/2)*dt,  sigma^2*dt )
//'
//' Priors:
//'   mu    ~ N(0, prior_sd_mu^2)
//'   sigma ~ Half-Normal(0, prior_sd_sigma^2)
//'
//' Target acceptance rate: 30-50% for a 2D sampler.
//'
//' @param log_returns   NumericVector: log(S_t / S_{t-1})
//' @param dt            Time step in years (default 1/252)
//' @param n_iter        Total iterations (including burn-in)
//' @param burnin        Burn-in iterations to discard
//' @param prop_sd_mu    Proposal std for mu
//' @param prop_sd_sigma Proposal std for sigma
//' @param mu_init       Initial mu
//' @param sigma_init    Initial sigma
//' @param prior_sd_mu   Prior std for mu    (default 0.5)
//' @param prior_sd_sigma Prior std for sigma (default 0.5)
//' @param seed          Random seed
//' @return List: mu_chain, sigma_chain, lp_chain, accept_rate, summary
//' @export
// [[Rcpp::export]]
List mh_gbm(NumericVector log_returns,
             double dt             = 1.0 / 252.0,
             int    n_iter         = 30000,
             int    burnin         = 5000,
             double prop_sd_mu     = 0.02,
             double prop_sd_sigma  = 0.01,
             double mu_init        = 0.05,
             double sigma_init     = 0.20,
             double prior_sd_mu    = 0.50,
             double prior_sd_sigma = 0.50,
             int    seed           = 42) {

    if (n_iter <= burnin) stop("n_iter must be greater than burnin.");
    if (dt <= 0.0)        stop("dt must be positive.");

    std::vector<double> ret(log_returns.begin(), log_returns.end());
    MHChain ch = run_mh_chain(ret, dt, n_iter, burnin,
                               prop_sd_mu, prop_sd_sigma,
                               mu_init, sigma_init,
                               prior_sd_mu, prior_sd_sigma, seed);

    auto s_mu  = mcmc::summarise(ch.mu);
    auto s_sig = mcmc::summarise(ch.sigma);

    DataFrame summary = DataFrame::create(
        Named("parameter") = CharacterVector{"mu", "sigma"},
        Named("mean")      = NumericVector{s_mu.mean,    s_sig.mean},
        Named("sd")        = NumericVector{s_mu.sd,      s_sig.sd},
        Named("q025")      = NumericVector{s_mu.q025,    s_sig.q025},
        Named("median")    = NumericVector{s_mu.median,  s_sig.median},
        Named("q975")      = NumericVector{s_mu.q975,    s_sig.q975},
        Named("ess")       = NumericVector{s_mu.ess_val, s_sig.ess_val}
    );

    return List::create(
        Named("mu_chain")    = ch.mu,
        Named("sigma_chain") = ch.sigma,
        Named("lp_chain")    = ch.log_post,
        Named("accept_rate") = ch.accept_rate,
        Named("summary")     = summary
    );
}

//' Run two independent MH chains and compute Gelman-Rubin R-hat
//'
//' R-hat measures between-chain vs within-chain variance.
//' Values < 1.01 = good convergence. > 1.1 = run longer.
//'
//' @param log_returns  Log-returns
//' @param dt           Time step
//' @param n_iter       Iterations per chain
//' @param burnin       Burn-in per chain
//' @param prop_sd_mu   Proposal sd for mu
//' @param prop_sd_sigma Proposal sd for sigma
//' @return List: r_hat_mu, r_hat_sigma, converged, summary, chain1, chain2
//' @export
// [[Rcpp::export]]
List mh_gbm_convergence(NumericVector log_returns,
                          double dt            = 1.0 / 252.0,
                          int    n_iter        = 30000,
                          int    burnin        = 5000,
                          double prop_sd_mu    = 0.02,
                          double prop_sd_sigma = 0.01) {

    std::vector<double> ret(log_returns.begin(), log_returns.end());

    // Chain 1: typical starting values
    MHChain c1 = run_mh_chain(ret, dt, n_iter, burnin,
                               prop_sd_mu, prop_sd_sigma,
                               0.05, 0.20, 0.5, 0.5, 42);

    // Chain 2: deliberately different start to stress-test convergence
    MHChain c2 = run_mh_chain(ret, dt, n_iter, burnin,
                               prop_sd_mu, prop_sd_sigma,
                               0.20, 0.40, 0.5, 0.5, 99);

    // R-hat = sqrt(V+ / W)
    //   W   = mean within-chain variance
    //   V+  = pooled variance estimate
    // Values near 1.0 → chains have mixed to the same distribution.
    double rhat_mu    = mcmc::r_hat({c1.mu,    c2.mu});
    double rhat_sigma = mcmc::r_hat({c1.sigma, c2.sigma});

    // Pooled summaries
    std::vector<double> mu_pool(c1.mu);
    mu_pool.insert(mu_pool.end(), c2.mu.begin(), c2.mu.end());
    std::vector<double> sig_pool(c1.sigma);
    sig_pool.insert(sig_pool.end(), c2.sigma.begin(), c2.sigma.end());

    auto s_mu  = mcmc::summarise(mu_pool);
    auto s_sig = mcmc::summarise(sig_pool);

    DataFrame summary = DataFrame::create(
        Named("parameter") = CharacterVector{"mu", "sigma"},
        Named("mean")      = NumericVector{s_mu.mean,    s_sig.mean},
        Named("sd")        = NumericVector{s_mu.sd,      s_sig.sd},
        Named("q025")      = NumericVector{s_mu.q025,    s_sig.q025},
        Named("median")    = NumericVector{s_mu.median,  s_sig.median},
        Named("q975")      = NumericVector{s_mu.q975,    s_sig.q975},
        Named("ess")       = NumericVector{s_mu.ess_val, s_sig.ess_val}
    );

    auto pack = [](const MHChain& c) {
        return List::create(
            Named("mu_chain")    = c.mu,
            Named("sigma_chain") = c.sigma,
            Named("lp_chain")    = c.log_post,
            Named("accept_rate") = c.accept_rate
        );
    };

    return List::create(
        Named("r_hat_mu")    = rhat_mu,
        Named("r_hat_sigma") = rhat_sigma,
        Named("converged")   = (rhat_mu < 1.01 && rhat_sigma < 1.01),
        Named("summary")     = summary,
        Named("chain1")      = pack(c1),
        Named("chain2")      = pack(c2)
    );
}

//' Adaptive Metropolis-Hastings for GBM
//'
//' Automatically tunes proposal standard deviations during burn-in to
//' target a 44% acceptance rate (optimal for 2D Gaussian targets).
//'
//' Adaptation rule (Haario et al. 2001 simplified):
//'   Every adapt_every steps:
//'     s ← s * exp( adapt_rate * (observed_rate - target_rate) )
//'   observed > target → s grows → wider proposals → lower acceptance
//'   observed < target → s shrinks → narrower proposals → higher acceptance
//'
//' Adaptation is frozen after burn-in ends.
//'
//' @param log_returns   Log-returns
//' @param dt            Time step
//' @param n_iter        Total iterations
//' @param burnin        Burn-in (adaptation phase)
//' @param target_rate   Target acceptance rate (default 0.44)
//' @param adapt_every   Adapt every N steps (default 100)
//' @param adapt_rate    Adaptation step size (default 0.5)
//' @param seed          Random seed
//' @return Same as mh_gbm, plus: adapted_prop_sd_mu, adapted_prop_sd_sigma
//' @export
// [[Rcpp::export]]
List mh_gbm_adaptive(NumericVector log_returns,
                      double dt           = 1.0 / 252.0,
                      int    n_iter       = 30000,
                      int    burnin       = 5000,
                      double target_rate  = 0.44,
                      int    adapt_every  = 100,
                      double adapt_rate   = 0.5,
                      int    seed         = 42) {

    std::vector<double> ret(log_returns.begin(), log_returns.end());

    std::mt19937 engine(seed);
    std::normal_distribution<double>       Z(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    double s_mu = 0.02, s_sig = 0.01;
    double mu_cur = 0.05, sigma_cur = 0.20;
    double lp_cur = gbm_bayes::log_posterior(ret, mu_cur, sigma_cur, dt);

    int keep = n_iter - burnin;
    std::vector<double> mu_ch(keep), sig_ch(keep), lp_ch(keep);

    int n_accepted = 0, window_accepted = 0;

    for (int i = 0; i < n_iter; ++i) {

        double mu_prop    = mu_cur    + s_mu  * Z(engine);
        double sigma_prop = sigma_cur + s_sig * Z(engine);

        if (sigma_prop > 0.0) {
            double lp_prop = gbm_bayes::log_posterior(ret, mu_prop, sigma_prop, dt);
            if (std::log(U(engine)) < lp_prop - lp_cur) {
                mu_cur = mu_prop; sigma_cur = sigma_prop; lp_cur = lp_prop;
                ++n_accepted; ++window_accepted;
            }
        }

        // Adaptation step (only during burn-in)
        if (i < burnin && (i + 1) % adapt_every == 0) {
            double obs_rate = (double)window_accepted / adapt_every;
            double factor   = std::exp(adapt_rate * (obs_rate - target_rate));
            s_mu  = std::max(1e-5, std::min(s_mu  * factor, 1.0));
            s_sig = std::max(1e-5, std::min(s_sig * factor, 0.5));
            window_accepted = 0;
        }

        if (i >= burnin) {
            int k = i - burnin;
            mu_ch[k] = mu_cur; sig_ch[k] = sigma_cur; lp_ch[k] = lp_cur;
        }
    }

    auto s_mu_s  = mcmc::summarise(mu_ch);
    auto s_sig_s = mcmc::summarise(sig_ch);

    DataFrame summary = DataFrame::create(
        Named("parameter") = CharacterVector{"mu", "sigma"},
        Named("mean")      = NumericVector{s_mu_s.mean,    s_sig_s.mean},
        Named("sd")        = NumericVector{s_mu_s.sd,      s_sig_s.sd},
        Named("q025")      = NumericVector{s_mu_s.q025,    s_sig_s.q025},
        Named("median")    = NumericVector{s_mu_s.median,  s_sig_s.median},
        Named("q975")      = NumericVector{s_mu_s.q975,    s_sig_s.q975},
        Named("ess")       = NumericVector{s_mu_s.ess_val, s_sig_s.ess_val}
    );

    return List::create(
        Named("mu_chain")              = mu_ch,
        Named("sigma_chain")           = sig_ch,
        Named("lp_chain")              = lp_ch,
        Named("accept_rate")           = (double)n_accepted / n_iter,
        Named("adapted_prop_sd_mu")    = s_mu,
        Named("adapted_prop_sd_sigma") = s_sig,
        Named("summary")               = summary
    );
}
