% train head logistic regression
% Input:
% X[N, D], X(n, :) is the nth feature vector
% T[N, 1], the label vector (-1/1)
% c, regularization
% Output:
% model, standard linear model w'*x + b
% description: penalty for mistake is suppressed, more robust
function[model] = train_head_logistic_regression(X, T, c)
eps_n = 0.1;
eps_p = 0;
dim = size(X, 2);
wb = fminunc(@(wb) cost_grad(X, T, wb, c, eps_n, eps_p), ...
    [ones(1, dim), 0]', ...
    optimset('GradObj', 'on'));
model = struct('name', 'glm', ...
    'w', wb(1:end-1), 'b', wb(end), 'active_func', @sigmoid);

function[s] = sigmoid(x)
s = 1.0 ./ (1.0 + exp(-x));

% d s(x) = s(x)(1-s(x))
% d ln(s(x)) = 1-s(x)
% d ln(s(xt)) = t*(1-s(xt))
% observed positive
% d ln(eps_n*(1-s)+(1-eps_p)*s) = d ln(eps_n + (1-eps_p-eps_n)*s)
% = d ln(eta) = 1/eta * [(1-eps_p-eps_n)*s*(1-s)]
% observed negative
% d ln((1-eps_n)*(1-s)+eps_p*s) = d ln((1-eps_n) - (1-eps_p-eps_n)*s)
% = d ln(eta) = 1/eta * [(1-eps_p-eps_n)*s*(1-s)]
% eps_n: p(observed pos | neg), eps_p: p(observed neg | pos)
function[cost, grad] = cost_grad(X, T, wb, c, eps_n, eps_p)
b = wb(end);
w = wb(1:end-1);
[num, dim] = size(X);
s = sigmoid(X*w+b);
s = max(1e-10, min(1 - 1e-10, s));
% cost
negI = T < 0;
cost = -sum(log(eps_n+(1-eps_n-eps_p)*s(~negI)));
cost = cost - sum(log(1-eps_n-(1-eps_n-eps_p)*s(negI)));
% grad
grad = -sum(1./(eps_n+(1-eps_n-eps_p)*s(~negI)).* ...
    ((1-eps_n-eps_p)*s(~negI).*(1-s(~negI))) * ...
    ones(1, dim+1) .* ...
    [X(~negI,:), ones(sum(~negI), 1)]);
grad = grad - sum(1./(1-eps_n-(1-eps_n-eps_p)*s(negI)).* ...
    (-(1-eps_n-eps_p)*s(negI).*(1-s(negI))) * ...
    ones(1, dim+1) .* ...
    [X(negI,:), ones(sum(negI), 1)]);
grad = grad';
% regularization
cost = cost  + 0.5*c*w'*w;
grad = grad + c*[w; 0];
return