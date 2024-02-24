%% Generate Data
PQ_combinations = [
    3, 2;
    3, 3;
    2, 6;
    6, 6;
    8, 5
    ];

d = 10;
num_samp = 20;
T = 4000;

% Create a loop to generate samples for each PQ combination
for i = 1:size(PQ_combinations, 1)
    P = PQ_combinations(i, 1);
    Q = PQ_combinations(i, 2);
    generate_ARMA_data(P, Q, d, num_samp, T);
end


%% Solve the problem

stepsizes =  [1e-5, 1e-5, 2.5e-6,1e-5,5e-6];
lambda_0_list = [0.5, 1,2,3,5,10];

Percent_lambda = 1;
ite_max = 6000;

for i=1:size(PQ_combinations,1)
    P = PQ_combinations(i,1);
    Q = PQ_combinations(i,2);
    stepsize = stepsizes(i);
    filename = sprintf('Data/ARMA_sample_P%d_Q%d.mat', P, Q);
    load(filename)

    for k = 1:size(lambda_0_list,2)
        lambda_0 = lambda_0_list(k);

        X_save = cell(num_samp, 1);
        err = cell(num_samp,1);
        for j = 1:num_samp
            [X_save{j}, err{j}, exitflag] = est_HS_ARMA(Y(:, j), d, lambda_0, Percent_lambda, stepsize, ite_max);
        end

        filename = sprintf('Results/P%d_Q%d_lambda%.1f.mat',P,Q,lambda_0);
        save(filename)
    end

end




%% Calculate error


% Initialize mean_err and std_err arrays to store results for all combinations
mean_err_all = zeros(size(PQ_combinations, 1), 6);
std_err_all = zeros(size(PQ_combinations, 1), 6);

for pQ_idx = 1:size(PQ_combinations, 1)
    P = PQ_combinations(pQ_idx, 1);
    Q = PQ_combinations(pQ_idx, 2);

    % Load data for the current P and Q combination
    filename = sprintf('Results/P%d_Q%d_lambda%.1f.mat', P, Q, lambda_0_list(1)); % Assuming lambda_0_list(1) is used
    load(filename);

    for lambda_idx = 1:length(lambda_0_list)
        lambda_0 = lambda_0_list(lambda_idx);

        % Load data for the current lambda
        filename = sprintf('Results/P%d_Q%d_lambda%.1f.mat', P, Q, lambda_0);
        load(filename);

        for samp_idx = 1:20
            temp = X_save{samp_idx};
            X(:, lambda_idx, samp_idx) = temp(:, end);
        end
    end

    % Calculate error for the current P and Q combination
    erro = zeros(20, 6);
    for samp_idx = 1:20
        erro(samp_idx, :) = sqrt(sum((X(:,:,samp_idx) - ARMA_samp(:,samp_idx)).^2, 1));
    end

    mean_err = mean(erro, 1);
    std_err = std(erro, 1);

    % Store the results in the respective arrays
    mean_err_all(pQ_idx, :) = mean_err;
    std_err_all(pQ_idx, :) = std_err;
end

% Display or save the mean_err_all and std_err_all arrays as needed
disp("Mean Error for All PQ Combinations:");
disp(mean_err_all);
disp("Standard Deviation of Error for All PQ Combinations:");
disp(std_err_all);


