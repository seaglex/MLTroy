% train logistic regression
% Input:
% X[N, D], X(n, :) is the nth feature vector
% T[N, 1], the label vector (-1/1)
% c, regularization
% Output:
% model, standard linear model w'*x + b
function[model] = train_logistic_regression(X, T, c)
dim = size(X, 2);
wb = fminunc(@(wb) cost_grad(X, T, wb, c), [0, 1, 0]', ...
    optimset('GradObj', 'on'));
model = struct('name', 'glm', ...
    'w', wb(1:end-1), 'b', wb(end), 'active_func', @sigmoid);

function[s] = sigmoid(x)
s = 1.0 ./ (1.0 + exp(-x));

% d s(x) = s(x)(1-s(x))
% d log(s(x)) = 1-s(x)
% d log(s(xt)) = t*(1-s(xt))
function[cost, grad] = cost_grad(X, T, wb, c)
b = wb(end);
w = wb(1:end-1);
[num, dim] = size(X);
pr = sigmoid(T.*(X*w+b));
pr = max(1e-10, min(1 - 1e-10, pr));
cost = -sum(log(pr)) + 0.5*c*w'*w;
grad = -sum( (T.*(1-pr))*ones(1, dim+1) .* [X, ones(num, 1)], 1)' ...
    + c*[w; 0];
return