% generalized linear model (glm)
% Input:
% model, standard linear model active_fuction(w'*x + b)
% x[N, D], feature vector
% Output:
% scores[N, 1]
function[scores] = linear_model(model, X)
if model.name ~= 'glm'
    throw(MException('linear_model:bad_name', 'model.name!=lr'))
end
scores = X*model.w + model.b;
if isfield(model, 'active_func')
    scores = model.active_func(scores);
end