% conclusion: 
N = 100;
X = ([1:N]' - (N+1)/2)*(10/N); % -5: 5
X = X + randn(N, 1) * 1;
X = sort(X);
w = 1;
t = X * w + randn(N, 1);
mt = sum(t)/N;

Cdot = X*X';
sigmas = [0.01:0.01:2];
costs = zeros(size(sigmas));
for n = 1:length(sigmas)
    sigma = sigmas(n);
    C = Cdot + eye(N)*sigma;
    invC = pinv(C);
    costs(n) = -0.5 * (log(det(C)) + (t-mt)'*invC*(t-mt) + N*log(2*pi));
end
plot(sigmas(50:end), costs(50:end));
box off;
grid on;
[x, I] = max(costs);
min_sigma_cost = [sigmas(I), costs(I)]