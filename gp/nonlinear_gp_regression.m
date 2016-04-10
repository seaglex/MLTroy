% T = f(X)
% theta/sigma2 is the super-parameters
% return: m(f(x)), var(f(x)), and m/v after optimizing theta(optional
% sigma2)
function[m0, v0] = nonlinear_gp_regression(X, T, x, theta, sigma2)

N = length(T);
C = nonlinear_kernel(X, X, theta) + eye(N, N)*sigma2;
K = nonlinear_kernel(X, x, theta);
c = diag(nonlinear_kernel(x, x, theta)) + sigma2;
[m0, v0] = gaussian_process_regression(C, T, K, c);