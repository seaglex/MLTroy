% linear regression
% Input:
% model, standard linear model w'*x + b
% x[N, D], feature vector
% Output:
% scores[N, 1]
function[scores] = linear_model(model, X)
scores = X*model.w + model.b;
if isfield(model, 'active_func')
    scores = model.active_func(scores);
end