// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;

//' Add two numeric vectors element-wise (Stage 1 integration test)
//'
//' @param x NumericVector
//' @param y NumericVector
//' @return NumericVector x + y
//' @export
// [[Rcpp::export]]
NumericVector add_vectors(NumericVector x, NumericVector y) {
    if (x.size() != y.size())
        stop("Vectors must be the same length.");
    return x + y;
}

//' Inner product of two vectors
//'
//' @param x NumericVector
//' @param y NumericVector
//' @return scalar double
//' @export
// [[Rcpp::export]]
double dot_product(NumericVector x, NumericVector y) {
    if (x.size() != y.size())
        stop("Vectors must be the same length.");
    return sum(x * y);
}
