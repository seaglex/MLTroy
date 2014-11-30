function[] = plat_linear_model(model, scale, fmt)
if length(model.w) ~= 2
    disp('Only 2-D linear model supported');
    return
end

if model.w(2) ~= 0
    x0 = [0, -model.b/model.w(2)]';
else
    x0 = [-model.b/model.w(1), 0]';
end
w = model.w;
w = w / norm(w);
% rotate pi/2
w = [0, -1; 1, 0] * w;
x1 = x0 - scale*w;
x2 = x0 + scale*w;
plot([x1(1), x2(1)], [x1(2), x2(2)], fmt, 'LineWidth', 3);
