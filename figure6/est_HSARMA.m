
function [result,AIC,BIC,X_all,const_all,ic] = est_HSARMA(y,opts)

%% Estimate ARMA model with best pair of lambda and phi selected by cross-validation





%% Initialize
N = length(y);

defaultopts.Pmax = floor(0.75*sqrt(N));
defaultopts.Qmax = floor(0.75*sqrt(N));
defaultopts.itermax = 100;
defaultopts.stepsize = 0.01;
defaultopts.beta = 0.8;
defaultopts.stopcrt = 1e-4;
iteration_BCD = 10;

if nargin < 2
    opts = defaultopts;
    
else
    opts = load_struct_vars(opts, defaultopts);
end


lambdaAR_seq = logspace(log10(100),log10(1),10);
lambdaMA_seq =  logspace(log10(100),log10(1),10);



%options_fmincon = optimoptions('fmincon', 'Algorithm', 'sqp', 'ConstraintTolerance', 1e-7, 'Display', 'off');


E = zeros(N, 1); % residual vector


% LOG Penalty
M = triu(ones(opts.Pmax));
n_para = size(M, 1);

indx_AR = {n_para, 1};
for i = 1:n_para
    indx_AR{i} = find(M(:, i));
end

w_AR = sqrt(sum(M, 1))';


M = triu(ones(opts.Qmax));
n_para = size(M, 1);

indx_MA = {n_para, 1};
for i = 1:n_para
    indx_MA{i} = find(M(:, i));
end

w_MA = sqrt(sum(M, 1))';



%% Use BIC to determine the best lambda
BIC = zeros(length(lambdaAR_seq),length(lambdaMA_seq));
AIC = zeros(length(lambdaAR_seq),length(lambdaMA_seq));
ic = cell(size(BIC));
X_all =cell(size(BIC));
const_all =cell(size(BIC));

for i=1:length(lambdaAR_seq)
    X = zeros(opts.Pmax+opts.Qmax,1);
    for j=1:length(lambdaMA_seq)
        X_pre = X;
        [X,const,info,~] = FIT_SARMA(y,lambdaAR_seq(i),lambdaMA_seq(j),X_pre);

        X(abs(X)<1e-3) =0;
        estMdl = arima('AR', X(1:opts.Pmax), 'MA', X(opts.Pmax+1:opts.Pmax+opts.Qmax), ...
            'Constant', const, 'Variance', 1);
        [~,~,LogL] = estimate(estMdl,y,'Display','off');
        [AIC(i,j),BIC(i,j),ic{i,j}] = aicbic(LogL,estMdl.P+estMdl.Q+1,length(y));
        
        X_all{i,j} = X;
        const_all{i,j} = const;
    end
end

[i,j] = find(BIC ==min(BIC(:)));
%[i,j] = find(AIC ==min(AIC(:)));

[~,idx] = min(i+j);
i = i(idx);
j = j(idx);
lambdaAR = lambdaAR_seq(i);
lambdaMA = lambdaMA_seq(j);

X_final = X_all{i,j};
result.AR = X_final(1:opts.Pmax);
result.MA = X_final(opts.Pmax+1:opts.Pmax+opts.Qmax);
result.lambdaAR = lambdaAR;
result.lambdaMA = lambdaMA;
result.const = const_all{i,j};



% % %% Cross-validation
% % T_cv =floor(0.9*N);
% % y_CV = y(1:T_cv);
% % MSFE = zeros(length(lambdaAR_seq),length(lambdaMA_seq));
% % STD = zeros(length(lambdaAR_seq),length(lambdaMA_seq));
% % X_ini= cell(length(lambdaAR_seq),length(lambdaMA_seq));
% % X = zeros(opts.Pmax+opts.Qmax,1);
% % for i=1:length(lambdaAR_seq)
% %     for j=1:length(lambdaMA_seq)
% %         X_ini{i,j} = X;
% %         X_pre = X;
% %         opts_cv.Pmax=15;
% %         opts_cv.Qmax=15;
% %         [X,const,info,~] = FIT_SARMA(y_CV,lambdaAR_seq(i),lambdaMA_seq(j),X_pre,opts_cv);
% %         
% %         estMdl = arima('AR', X(1:opts.Pmax), 'MA', X(opts.Pmax+1:opts.Pmax+opts.Qmax), ...
% %             'Constant', const, 'Variance', 1);
% %         
% %         
% %         y_pred = zeros(N-T_cv,1);
% %         for t = 0:N-T_cv-1
% %             y_pred(t+1) = forecast(estMdl,1,y(1:T_cv+t));
% %         end
% %         SFE = (y_pred - y(T_cv+1:N)).^2;
% %         MSFE(i,j) = mean(SFE);
% %         STD(i,j) = std(SFE);
% %         
% %         
% %         
% %         
% %     end
% % end
% % %Find the best lambda
% % [i,j] = find(MSFE ==min(MSFE(:)));
% % [~,idx] = min(i+j);
% % i = i(idx);
% % j = j(idx);
% % lambdaAR = lambdaAR_seq(i);
% % lambdaMA = lambdaMA_seq(j);
% % 
% % [X,const,info,~] = FIT_SARMA(y,lambdaAR,lambdaMA,X_ini{i,j});
% % 
% % result.AR = X(1:opts.Pmax);
% % result.MA = X(opts.Pmax+1:opts.Pmax+opts.Qmax);
% % result.lambdaAR = lambdaAR;
% % result.lambdaMA = lambdaMA;
% % result.const = const;
