d = 10;

T = 501;
num_samp = 20;
ARMA_samp = zeros(2*d, 1);
root_max = [0.5, 0.99];


for P = [5, 8]
    for Q = [2, 5, 8]

        for caseAR = 1:2
            for caseMA = 1:2
                root_max_AR = root_max(caseAR);
                root_max_MA = root_max(caseMA);

                %% Generate ARMA model
                rng(10)


                while (1)
                    X_AR = 0.1 + (root_max_AR - 0.1) * rand(P, 1);
                    sign = randi([0, 1], P, 1) * 2 - 1; % random sign
                    X_AR = sign .* X_AR;
                    X_AR(1) = root_max_AR;
                    temp = poly(X_AR);
                    ARMA_samp(1:P) = -temp(2:end);

                    X_MA = 0.1 + (root_max_MA - 0.1) * rand(Q, 1);
                    sign = randi([0, 1], Q, 1) * 2 - 1; % random sign
                    X_MA = sign .* X_MA;
                    X_MA(1) = root_max_MA;
                    temp = poly(X_MA);
                    ARMA_samp(d+1:d+Q) = temp(2:end);


                    temp = abs(ARMA_samp(ARMA_samp ~= 0));
                    if max(temp) < 1.5 && min(temp) > 0.001
                        break

                    end
                end

                %% Generate time series and discard first 300 points
                Y = zeros(T, num_samp);
                for i = 1:num_samp
                    Mdl = arima('AR', ARMA_samp(1:P), 'MA', ARMA_samp(d+1:d+Q), ...
                        'Constant', 10, 'Variance', 1);

                    Y(:, i) = simulate(Mdl, T);
                end

                % Discard first 300 points
                Y = Y(301:end, :);
                Y_train = Y(1:end-1, :);
                Y_test = Y(end, :);

                %% Estimate ARMA
                result = cell(num_samp, 1);
                BIC = cell(num_samp, 1);
                idx_erro = [];
                for i = 1:num_samp
                    try
                        tic
                        [result{i}, BIC{i}] = est_HSARMA(Y_train(:, i));
                        disp(i);
                        toc
                    catch
                        idx_erro = [idx_erro, i];
                        disp(['erro', num2str(i)]);
                    end
                end
                Y(:, idx_erro) = [];
                Y_train(:, idx_erro) = [];
                Y_test(idx_erro) = [];
                result(idx_erro) = [];
                filename = ['P', num2str(P), '_Q', num2str(Q), '_AR', num2str(caseAR), '_MA', num2str(caseMA)]
                writematrix(Y, ['Data/', filename, '.csv'])

                %% Predict
                y_pred = zeros(size(Y_test));
                P_est = zeros(size(Y_test));
                Q_est = zeros(size(Y_test));
                for i = 1:length(Y_test)
                    fitresult = result{i};
                    estMdl = arima('AR', fitresult.AR, 'MA', fitresult.MA, ...
                        'Constant', fitresult.const, 'Variance', 1);
                    y_pred(i) = forecast(estMdl, 1, Y_train(:, i));
                    P_est(i) = estMdl.P;
                    Q_est(i) = estMdl.Q;

                end
                MSE = mean((y_pred - Y_test).^2)

                %%
                save(['Result/', filename])
            end
        end
    end
end