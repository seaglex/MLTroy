function[m, v] = linear_gp_regression(X, T, x, sigma2)
N = length(T);
C = linear_cov(X, X) + eye(N, N)*sigma2;
K = linear_cov(X, x);
c = diag(linear_cov(x, x)) + sigma2;
[m, v] = gaussian_process_regression(C, T, K, c);

function[cov] = linear_cov(X, Y)
cov = X*Y';
