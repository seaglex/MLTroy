function[theta, sigma2] = optimize_gp(X, T, theta, sigma2)
if nargout == 2
    opt_sigma2 = 1;
elseif nargout == 1
    opt_sigma2 = 0;
else
    disp('usage: [theta [,sigma2]] = optimize_gp(X, T, theta, sigma2)')
    opt_sigma2 = 1;
end

[Cdist, Cdot] = nonlinear_kernel_detail(X, X);
disp('optimizing');
if ~opt_sigma2
    Dtheta = 4;
    opt_theta = fmincon(@(arg) nonlinear_gp_val_grad(T, Cdist, Cdot, arg, sigma2), ...
                theta, ...
                [], [], [], [], zeros(Dtheta, 1), [], [], ...
                optimset('Gradobj', 'on'));
else
    Dtheta = 5;
    opt_theta = fmincon(@(arg) nonlinear_gp_val_grad(T, Cdist, Cdot, arg), ...
                [theta; sigma2], ...
                [], [], [], [], zeros(Dtheta, 1), [], [], ...
                optimset('Gradobj', 'on'));
end

disp(opt_theta);
theta = opt_theta(1:4);
if opt_sigma2
    sigma2 = opt_theta(5);
end