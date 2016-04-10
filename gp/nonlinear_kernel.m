function[C] = nonlinear_kernel(X, x, theta)
[Cdist, Cdot] = nonlinear_kernel_detail(X, x);
C = theta(1)*exp(-0.5*theta(2)*Cdist) + theta(3) + theta(4) * Cdot;
