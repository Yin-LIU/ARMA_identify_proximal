
function generate_ARMA_data(P, Q, d, num_samp, T)

    ARMA_samp = zeros(2*d, num_samp);

    rng(8)
    for i = 1:num_samp
        while 1

            X_AR = 0.1 + 0.9 * rand(P, 1);

            sign = randi([0, 1], P, 1) * 2 - 1; % random sign

            X_AR = sign .* X_AR;

            X_MA = 0.1 + 0.9 * rand(Q, 1);

            sign = randi([0, 1], Q, 1) * 2 - 1; % random sign

            X_MA = sign .* X_MA;

            AR_check = roots([1, -X_AR']);
            MA_check = roots([1, X_MA']);

            c = [max(abs(AR_check)); max(abs(MA_check))] - 0.9;
            if c < 0
                break
            end
        end
        ARMA_samp(1:P, i) = X_AR;
        ARMA_samp(d+1:d+Q, i) = X_MA;
    end

    Y = zeros(T, num_samp);
    for i = 1:num_samp
        Mdl = arima('AR', ARMA_samp(1:P, i), 'MA', ARMA_samp(d+1:d+Q, i), ...
                'Constant', 0, 'Variance', 1);

        Y(:, i) = simulate(Mdl, T);
    end
    
    filename = sprintf('Data/ARMA_sample_P%d_Q%d.mat', P, Q);
    save(filename, 'ARMA_samp', 'Y');

end
