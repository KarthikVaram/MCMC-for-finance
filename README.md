# Monte Carlo Finance Engine

A high-performance computational finance engine built with C++ and R.

## Project Structure

```
project/
├── src/          # C++ source files (compiled via Rcpp)
├── include/      # C++ header files
├── R/            # R scripts for analysis and visualization
├── tests/        # Unit tests
└── README.md
```

## Stages

| Stage | Description |
|-------|-------------|
| 1 | R/C++ Infrastructure (Rcpp) |
| 2 | RNG + Brownian Motion + GBM |
| 3 | Monte Carlo Option Pricing |
| 4 | Variance Reduction Techniques |
| 5 | Parallel Computing (OpenMP / RcppParallel) |
| 6 | Exotic Option Pricing |
| 7 | Heston Stochastic Volatility Model |

## Setup

```r
install.packages(c("Rcpp", "RcppParallel", "ggplot2", "microbenchmark"))
```

Compile all C++ files from R:
```r
library(Rcpp)
sourceCpp("src/rng.cpp")
sourceCpp("src/brownian.cpp")
sourceCpp("src/gbm.cpp")
sourceCpp("src/mc_pricer.cpp")
sourceCpp("src/variance_reduction.cpp")
sourceCpp("src/parallel_mc.cpp")
sourceCpp("src/exotic_options.cpp")
sourceCpp("src/heston.cpp")
```
