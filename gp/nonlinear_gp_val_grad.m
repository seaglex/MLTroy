function[cost, grad] = nonlinear_gp_val_grad(T, Cdist, Cdot, theta, sigma2)
Dtheta = length(theta);
if nargin==4
    sigma2 = theta(5);
end
C = theta(1)*exp(-0.5*theta(2)*Cdist) + theta(3) + theta(4) * Cdot;
[N, M] = size(C);
C = eye(N, M)*sigma2 + C;
invC = pinv(C);
dCdtheta = zeros(N, M, Dtheta);
dCdtheta(:, :, 1) = exp(-0.5*theta(2)*Cdist);
dCdtheta(:, :, 2) = theta(1)*exp(-0.5*theta(2)*Cdist).*(-0.5*Cdist);
dCdtheta(:, :, 3) = ones(N, M);
dCdtheta(:, :, 4) = Cdot;
dCdtheta(:, :, 5) = eye(N, N);
cost = 0.5*( log(det(C)) + T'*invC*T + N*log(2*pi) );
grad = zeros(Dtheta, 1);
for n = 1:Dtheta
    grad(n) = 0.5*trace(invC * dCdtheta(:, :, n)) - 0.5*T'*invC*dCdtheta(:, :, n)*invC*T;
end