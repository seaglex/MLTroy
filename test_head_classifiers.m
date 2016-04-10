function[] = test_head_classifier()
c = 1;
model_trainers = {
    @(X, T) train_linear_regression(X, T, c);
    @(X, T) train_logistic_regression(X, T, c);
    @(X, T) train_head_logistic_regression(X, T, c);
    % @(X, T) train_svm(X, T, c);
    };
fmts = {'g', 'm', 'k', 'y'};
names = {'linear regression', ...
    'logistic regression', 'Head LR'};
data_generators = {
    @() gen_asymmetric();
    %@() anti_fisher_discriminant();
    %@() anti_logistic_regression1();
    %@() anti_logistic_regression2();
    };
titles = {'data1', 'data2', 'data3', 'data4'};

close all;
AUCs = zeros(1, length(model_trainers));
for d = 1:length(data_generators)
    generator = data_generators{d};
    [X, labels] = generator();
    I = labels>0;
    pos = X(I, :);
    neg = X(~I, :);
    figure(d);
    scatter(pos(:, 1), pos(:, 2), 'r.');
    hold on;
    scatter(neg(:, 1), neg(:, 2), 'b.');
    for m = 1:length(model_trainers)
        trainer = model_trainers{m};
        model = trainer(X, labels);
        plot_linear_model(model, 1, fmts{m});
        AUCs(d, m) = auc(labels, linear_model(model, X));
    end
    legend([{'Pos', 'Neg'}, names]);
    title(titles(d));
    grid on;
    box off;
    hold off;
end
AUCs


function[X, labels] = gen_two_directions()
num_bar = 1000;
bar1 = [-2:4/num_bar:2;
    -2:4/num_bar:2;
    ]';
bar2 = [2:-4/num_bar:-2;
    -2:4/num_bar:2]';
num1 = size(bar1, 1);
num2 = size(bar2, 1);
bar1 = bar1 + (rand(num1, 2)-0.5)*1;
bar2 = bar2 + (rand(num2, 2)-0.5)*1;
X = [bar1; bar2];
T1 = 1/num1*[0:(num1-1)]' >= rand(num1, 1);
T2 = (0.6/num2*[0:(num2-1)]' + 0.2) >= rand(num2, 1);
T = [T1; T2];
labels = zeros(size(T));
labels(T) = 1;
labels(~T) = -1;

function[X, labels] = gen_asymmetric()
num_square = 1000;
X = [
    rand(num_square, 2);
    rand(num_square, 2)+ones(num_square, 1)*[0, -1];
    rand(num_square, 2)+ones(num_square, 1)*[-1, -1];];
labels = ones(num_square*3, 1);
T = rand(num_square*3, 1) < 0.9;
labels(T) = -1;
labels(1:num_square) = 1;