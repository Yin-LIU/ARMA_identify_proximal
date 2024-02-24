d = 5;
num_samp = 10;
P=3;
Q=2;
T=4000;
generate_ARMA_data(P, Q, d, num_samp, T)


load("Data/ARMA_sample_P3_Q2.mat")


%% Solve the problem
stepsize = 1e-6;
lambda_list = [0.1,0.5,1,2,3,5,10];
Percent_lambda = 0.9;
ite_max = 6000;

for j = 1:length(lambda_list)
    lambda_0 = lambda_list(j);

    X_save = cell(10, 1);
    err = cell(10,1);
    for i = 1:10

        [X_save{i}, err{i}, exitflag] = est_HS_ARMA(Y(:, i), d, lambda_0, Percent_lambda, stepsize, ite_max);

    end

    filename = sprintf('Results/lambda%.1f.mat',lambda_0);
    save(filename)
end


%% Calculate error
X = zeros(10, 7, 10);

for j = 1:length(lambda_list)
    lambda_0 = lambda_list(j);
    lambda_filename = sprintf('Results/lambda%.1f.mat', lambda_0);
    load(lambda_filename);

    for i = 1:10
        temp = X_save{i};
        X(:, j, i) = temp(:, end);
    end
end

erro = zeros(10, 7);
for i = 1:10
    erro(i, :) = sqrt(sum((X(:, :, i) - ARMA_samp(:, i)).^2, 1));
end

% Reshape X
X_tall = [];
for i = 1:10
    X_tall = [X_tall; X(:, :, i)];
end



%% Compare to ARIMA toolbox result

% Estimate ARMA using Toolbox.
X_toolbox = zeros(10, 10);

Mdl = arima(3, 0, 2);
Mdl.Constant = 0;
Mdl.Variance = 1;
EstMdl = cell(10, 1);
for i = 1:10
    EstMdl{i} = estimate(Mdl, Y(:, i));
    X_toolbox(1:3, i) = cell2mat(EstMdl{i}.AR)';
    X_toolbox(6:7, i) = cell2mat(EstMdl{i}.MA)';

end

erro_toolbox = (sqrt(sum((X_toolbox - ARMA_samp).^2, 1)));

Mdl_under1 = arima(2, 0, 2);
Mdl_under1.Constant = 0;
Mdl_under1.Variance = 1;
EstMdl_under1 = cell(10, 1);
for i = 1:10
    EstMdl_under1{i} = estimate(Mdl_under1, Y(:, i));
    X_toolbox(1:2, i) = cell2mat(EstMdl_under1{i}.AR)';
    X_toolbox(6:7, i) = cell2mat(EstMdl_under1{i}.MA)';

end

erro_under1 = (sqrt(sum((X_toolbox - ARMA_samp).^2, 1)));


Mdl_under2 = arima(3, 0, 1);
Mdl_under2.Constant = 0;
Mdl_under2.Variance = 1;
EstMdl_under2 = cell(10, 1);
for i = 1:10
    EstMdl_under2{i} = estimate(Mdl_under2, Y(:, i));
    X_toolbox(1:3, i) = cell2mat(EstMdl_under2{i}.AR)';
    X_toolbox(6, i) = cell2mat(EstMdl_under2{i}.MA)';

end

erro_under2 = mean(sqrt(sum((X_toolbox - ARMA_samp).^2, 1)));


Mdl_over1 = arima(5, 0, 5);
Mdl_over1.Constant = 0;
Mdl_under2.Variance = 1;
EstMdl_over1 = cell(10, 1);
for i = 1:10
    EstMdl_over1{i} = estimate(Mdl_over1, Y(:, i));
    X_toolbox(1:5, i) = cell2mat(EstMdl_over1{i}.AR)';
    X_toolbox(6:10, i) = cell2mat(EstMdl_over1{i}.MA)';

end

erro_over1 = mean(sqrt(sum((X_toolbox - ARMA_samp).^2, 1)));


Mdl_over2 = arima(3, 0, 3);
Mdl_over2.Constant = 0;
Mdl_over2.Variance = 1;
EstMdl_over2 = cell(10, 1);
for i = 1:10
    EstMdl_over2{i} = estimate(Mdl_over2, Y(:, i));
    X_toolbox(1:3, i) = cell2mat(EstMdl_over2{i}.AR)';
    X_toolbox(6:8, i) = cell2mat(EstMdl_over2{i}.MA)';

end

erro_over2 = mean(sqrt(sum((X_toolbox - ARMA_samp).^2, 1)));

%%
clf

plot([0.1, 0.5, 1, 2, 3, 5, 10], mean(erro, 1), 'o-', 'LineWidth', 3, 'Color', [0,0,0])
yline(mean(erro_under1), 'r-', 'LineWidth', 3, 'Color', 'r')
yline(mean(erro_toolbox), '-', 'LineWidth', 3, 'Color', [0, 0.5, 0])
for i = 1:10
    hold on

    plot([0.1, 0.5, 1, 2, 3, 5, 10], erro(i, :), 'Color', [0.5, 0.5, 0.5]) % HS-ARMA sample
    yline(erro_toolbox(i), '-', 'LineWidth', 0.5, 'Color', [0, 0.5, 0])

    yline(erro_under1(i), 'r-', 'LineWidth', 0.5, 'Color', 'r')

end


%yline(erro_under2, 'r-', 'Under-estimated model ARMA(3,1)', 'LineWidth', 3, 'Color', [0.8500, 0.3250, 0.0980])

xlabel('\lambda_0')
ylabel('$$\epsilon = ||(\hat{\phi},\hat{\theta})-(\phi^*,\theta^*)||_2$$','Interpreter','Latex')
legend('Average of HS-ARMA estimation', ...
    'Average of misidentified model ARMA(2,2)', 'Average of correctly identified model ARMA(3,2)')

%% Compare prediction difference
F = zeros(20, 10);
F_under1 = zeros(size(F));
F_under2 = zeros(size(F));
F_over1 = zeros(size(F));
F_over2 = zeros(size(F));
F_t = zeros(size(F));

F_HS = zeros([size(F), 7]);
for i = 1:10
    F_under1(:, i) = forecast(EstMdl_under1{i}, 20, Y(:, i));
    F(:, i) = forecast(arima('AR', ARMA_samp(1:5, i), 'MA', ARMA_samp(6:end, i), ...
        'Constant', 0, 'Variance', 1), 20, Y(:, i));
    F_under2(:, i) = forecast(EstMdl_under2{i}, 20, Y(:, i));
    F_over1(:, i) = forecast(EstMdl_over1{i}, 20, Y(:, i));
    F_over2(:, i) = forecast(EstMdl_over2{i}, 20, Y(:, i));
    F_t(:, i) = forecast(EstMdl{i}, 20, Y(:, i));
end

for j = 1:7 % over lambda
    for i = 1:10 % over sample
        F_HS(:, i, j) = forecast(arima('AR', X(1:5, j, i), 'MA', X(6:10, j, i), ...
            'Constant', 0, 'Variance', 1), 20, Y(:, i));

    end
end


pred_err_over1 = sqrt(mean((F_over1 - F).^2, 2));
pred_err_over2 = sqrt(mean((F_over2 - F).^2, 2));
pred_err_under1 = sqrt(mean((F_under1 - F).^2, 2));
pred_err_under2 = sqrt(mean((F_under2 - F).^2, 2));

pred_err_HS = sqrt(mean((F_HS - F).^2, 2));
pred_err_t = sqrt(mean((F_t - F).^2, 2));

%%
figure()
semilogy(pred_err_t, 'Color', [0.4660, 0.6740, 0.1880], 'LineWidth', 2)
hold on
%semilogy(pred_err_over1, '+-', 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2)
%semilogy(pred_err_over2, 'o-', 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2)
semilogy(pred_err_under1, '-', 'Color', 'r', 'LineWidth', 2)
%semilogy(pred_err_under2, 'o-', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2)

semilogy(mean(pred_err_HS(:, :, 1:6), 3), 'LineWidth', 3, 'Color', [0,0,0])
for j = 1:6
    semilogy(pred_err_HS(:, :, j), 'Color', [0.5, 0.5, 0.5])
end

legend('Correctly identified model ARMA(3,2)', 'Misidentified model ARMA(2,2)', 'Average of predictions of HS-ARMA', 'Predictions of HS-ARMA with different \lambda_0')

xlim([1, 20])
xlabel('time')
ylabel('RMSE')




