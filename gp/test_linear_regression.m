function[] = test_linear_regression()
N = 10;
X = ([1:N]' - (N+1)/2)*(10/N); % -5: 5
X = X + randn(N, 1) * 1;
X = sort(X);
x = [-7:1:7]';
theta = [1, 1, 1, 1]';
sigma2 = 1;
opt_sigma2 = 1;

% linear data
[T, t] = get_linear_data(X, x);
m0 = x*inv(X'*X)*(X'*T);

[m1, v1] = linear_gp_regression(X, T, x, sigma2);
% [y, m20, m, v]
figure(1);
title('linear - linear-gp')
plot(x, t, 'r.-', x, m0, 'g.-');
hold on;
errorbar(x, m1, sqrt(v1), 'b');
hold off;
box off;
grid on;
legend('test', 'test-lr', 'test-linear-gp')

[m2, v2] = nonlinear_gp_regression(X, T, x, theta, sigma2);
if opt_sigma2
    [theta, sigma2] = optimize_gp(X, T, theta, sigma2);
else
    theta = optimize_gp(X, T, theta, sigma2);
end
[m3, v3] = nonlinear_gp_regression(X, T, x, theta, sigma2);

% [y, m20, m2, v2, m3, v3]
figure(2);
title('linear - nonlinear')
plot(x, t, 'r.-');
hold on;
errorbar(x, m2, sqrt(v2), 'g');
errorbar(x, m3, sqrt(v3), 'b');
hold off;
box off;
grid on;
legend('test', 'test-ngp-init', 'test-ngp-opt');

[m4, v4] = bayesian_linear_regression(X, T, x);
figure(3);
title('linear - Bayesian')
plot(x, t, 'r.-');
hold on;
errorbar(x, m4, sqrt(v4), 'b');
hold off;
box off;
grid on;
legend('real', 'test-Bayesian');


% linear case y = alpha*(x'*ones) + sigma2*random
% x should be symmetric, y is 0-mean in nature
function[T, t] = get_linear_data(X, x)
w0 = 1;
sigma = 0.5;
D = size(X, 2);
w = w0*ones(D, 1);
N = size(X, 1);
Y = X * w;
T = Y + sigma*randn(N, 1);
y = x * w;
t = y + sigma*randn(length(y), 1);

