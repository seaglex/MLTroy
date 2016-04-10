% svm classifier
% Input:
% model, kernel svm model \sum(alpha_i*y_i*k(sv_i, x)) + b
% x[N, D], feature vector
% Output:
% scores[N, 1]
function[scores] = svm_model(model, X)
if model.name ~= 'svm'
    throw(MException('svm_model:name_mismatch', 'model.name!=svm'));
end
num = size(X, 1);
scores = zeros(num, 1);
for n = 1:num
    scores(n) = (model.alpha .* model.label) .* ...
        model.kernel_function(model.sv, X(n, :)');
end
