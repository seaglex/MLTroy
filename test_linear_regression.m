function[w]=test_linear_regression()
N = 1000;
pos = [rand(N, 1)*4-3, rand(N, 1)];
neg = [rand(N, 1)*4-1, -rand(N, 1)];
X = [pos; neg];
y = [ones(N,1); -ones(N, 1)];

exx = (1/N/2) * X'*X;
exy = (1/N/2) * X'*y;
invEXX = inv(exx);
w = invEXX * exy;

figure(1);
scatter(X(1:N, 1), X(1:N, 2), '*');
hold on;
scatter(X(N+1:end, 1), X(N+1:end, 2), '^');
hold on;
plot([0, w(1)], [0, 0], 'r') 
plot([0, 0], [0, w(2)], 'r') 
hold off;

% exx = U*S*V' (U==V)
% S = U'*exx*V
% nX = X*U*S^-0.5
% nX'*nX = S^-0.5*U'*X'*X*U*S^0.5 = S^-0.5*U' * U*S*U' * U*S^-0.5 = I
[U, S, V] = svd(exx);
T = U*(S^-0.5);
nX = X*T;
nEXX = (1/N/2) * nX'*nX;
nEXY = (1/N/2) * nX' * y;
nw = inv(nEXX) * nEXY;
w01 = [1, 0]*T - [-1, 0]*T;
w01 = w01 ./ sqrt(w01*w01');
pnw = (w01 * nw) * w01';
dnw = nw - pnw;

figure(2);
scatter(nX(1:N, 1), nX(1:N, 2), '*');
hold on;
scatter(nX(N+1:end, 1), nX(N+1:end, 2), '^');
hold on;
plotpoint(w01, 'k');
plotpoint(nw, 'r');
plotpoint(pnw, 'r');
plotpoint(dnw, 'r');
hold off;

function[] = plotpoint(x, fmt)
plot([0, x(1)], [0, x(2)], fmt);
