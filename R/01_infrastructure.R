# Stage 1: R/C++ Infrastructure Test
# Run this after setting your working directory to the project root.

library(Rcpp)
library(ggplot2)
library(microbenchmark)

# Compile
sourceCpp("src/hello_rcpp.cpp")

# Test vector addition
x <- 1:5 * 1.0
y <- 6:10 * 1.0

result <- add_vectors(x, y)
stopifnot(all.equal(result, x + y))
cat("add_vectors OK:", result, "\n")

# Test dot product
dp <- dot_product(x, y)
cat("dot_product OK:", dp, "(expected", sum(x * y), ")\n")

# Benchmark R vs C++
bm <- microbenchmark(
  R_add    = x + y,
  cpp_add  = add_vectors(x, y),
  times    = 1000L
)
print(bm)
