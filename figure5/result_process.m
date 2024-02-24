filePattern = fullfile('./Result/', '*.mat'); % Change to whatever pattern you need.

file_list = dir(filePattern);

%%
pred_erro1 = zeros(length(file_list), 1);
PQ1 = cell(length(file_list), 1);

pred_erro2 = zeros(length(file_list), 1);
PQ2 = cell(length(file_list), 1);

%% Read matlab result
for k = 1:length(file_list)
    filename = fullfile(file_list(k).folder, file_list(k).name);

    load(filename, 'MSE', 'P_est', 'Q_est')
    pred_erro1(k) = MSE;
    PQ1{k} = [P_est; Q_est];
end

%% Read R result

filePattern = fullfile('./R_Result/', 'PQ*.csv'); % Change to whatever pattern you need.

file_list = dir(filePattern);

for k = 1:length(file_list)
    filename = fullfile(file_list(k).folder, file_list(k).name);
    temp = readmatrix(filename);
    temp(:, 1) = [];
    PQ2{k} = temp;
end
pred_erro2 = readmatrix('./R_Result/Herro.csv')';
pred_erro2(1) = [];


pred_erro3 = readmatrix('./R_Result/Lerro.csv')';
pred_erro3(1) = [];

%%
pred_erro1 = sqrt(pred_erro1);
pred_erro2 = sqrt(pred_erro2);
pred_erro3 = sqrt(pred_erro3);
temp = reshape(pred_erro1, 4, []);
temp = mean(temp, 1);
e1 = temp;

temp = reshape(pred_erro2, 4, []);
temp = mean(temp, 1);
e2 = temp;

temp = reshape(pred_erro3, 4, []);
temp = mean(temp, 1);
e3 = temp;
label = categorical({'(2,2)', '(2,5)', '(2,8)', '(5,2)', '(5,5)', '(5,8)', '(8,2)', '(8,5)', '(8,8)'});
label = reordercats(label, {'(2,2)', '(2,5)', '(2,8)', '(5,2)', '(5,5)', '(5,8)', '(8,2)', '(8,5)', '(8,8)'});
bar(label, [e1', e2', e3'])
xlabel('(p*,q*)')
ylabel('RMSE')
legend('HS-ARMA(Proposed)', 'H-Lag penalty', 'L1penalty')
