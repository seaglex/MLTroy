function[] = test_nonlinear_regression()
N = 10;
X = ([1:N]' - (N+1)/2)*(10/N); % -5: 5
X = X + randn(N, 1) * 1;
X = sort(X);
x = [-7:1:7]';
theta = [1, 1, 1, 1]';
sigma2 = 1;
opt_sigma2 = 1;

% nonlinear data
[T, t] = get_nonlinear_data(X, x);

[m, v] = linear_gp_regression(X, T, x, sigma2);
m00 = x * inv(X'*X)*(X'*T);
% [y, m, v]
figure(3);
title('Nonlinear - linear')
plot(x, t, 'r.-', x, m00, 'g.-');
hold on;
errorbar(x, m, sqrt(v), 'b');
hold off;
box off;
grid on;
legend('test', 'test-lr', 'test-gp')

[m0, v0] = nonlinear_gp_regression(X, T, x, theta, sigma2);
if opt_sigma2
    [theta, sigma2] = optimize_gp(X, T, theta, sigma2);
else
    theta = optimize_gp(X, T, theta, sigma2);
end
[m1, v1] = nonlinear_gp_regression(X, T, x, theta, sigma2);

% [y, m0, v0, m1, v1]
figure(4);
title('Nonlinear - nonlinear')
plot(x, t, 'r.-');
hold on;
errorbar(x, m0, sqrt(v0), 'g');
errorbar(x, m1, sqrt(v1), 'b');
hold off;
box off;
grid on;
legend('test', 'test-ngp-init', 'test-ngp-opt');


% non-linear case y = alpha*x(1)+alpha*x(2)^2 + sigma2*random - mean
function[T, t] = get_nonlinear_data(X, x)
alpha = 1;
sigma = 0.5;
[N, D] = size(X);
Y = zeros(N, 1);
s = 1;
y = zeros(size(x, 1), 1);
for d = 1:D
    Y = Y + s * alpha * X(:,d).^2;
    y = y + s * alpha * x(:,d).^2;
    s = -s;
end
meanY = sum(Y)/N;
Y = Y - meanY;
y = y - meanY;
T = Y + sigma*randn(N, 1);
t = y + sigma*randn(length(y), 1);
