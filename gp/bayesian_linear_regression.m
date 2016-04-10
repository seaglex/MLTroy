function[my, vy] = bayesian_linear_regression(X, t, x)
beta = 1;
[N, D] = size(X);
m0 = zeros(D, 1);
S0 = eye(D);

SN = pinv(pinv(S0) + beta*X'*X);
mN = SN * (pinv(S0)*m0 + beta*X'*t);

my = x * mN;
vy = diag(x * SN * x') + beta;
