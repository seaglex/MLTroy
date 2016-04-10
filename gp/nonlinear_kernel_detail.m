function[Cdist, Cdot] = nonlinear_kernel_detail(X, x)
N = size(X, 1);
M = size(x, 1);
Cdist = zeros(N, M);
for m = 1:M
    Cdist(:, m) = sum((X-ones(N, 1)*x(m, :)).^2, 2);
end
Cdot = X * x';
