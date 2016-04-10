% Cij = k(xi, xj) + I(i==j)*sigma
% k = k(xi, x)
% c = k(x, x) + sigma
% t : label
function[m, v] = gaussian_process_regression(C, t, K, c)
invC = inv(C);
num = length(c);
m = zeros(num, 1);
v = zeros(num, 1);
m = K' * (invC * t);
for n = 1:num
    v(n) = c(n) - K(:, n)'*invC*K(:, n);
end
