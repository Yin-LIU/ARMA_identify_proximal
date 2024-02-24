function [X_save,err,exitflag] = est_HS_ARMA(y, d, lambda_0,Percent_lambda, stepsize, ite_max)

% assume constant of ARMA is 0
%
% Input
%   Y(N,1)   sample data
%   P        Highest order of AR
%   Q        Highest order of MA
%   lambda   Penalty parameter
%   stepsize stepsize for PGM
%   ite_max  Maximum iteration numbers for PGM
%   var=1      Known variable of ARMA model
%
% Output
%   X(P+Q,1)    estimated parameters


%% Initialize
N = length(y);
E = zeros(N, 1);
X0 = zeros(2*d, 1); % initial point
iteration_BCD = 10;
err = zeros(ite_max);
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'ConstraintTolerance', 1e-7, 'Display', 'off');

X_save=zeros(2*d,ite_max);
crt_stop=5e-6;

lambda_AR = lambda_0*sqrt(N);
lambda_MA = Percent_lambda*lambda_AR;


% LOG Penalty
M = zeros(d);
M = triu(ones(d));
n_para = size(M, 1);

indx = {n_para, 1};
for i = 1:n_para
    indx{i} = find(M(:, i));
end

w = sqrt(sum(M, 1))';

%%
X=X0;
for ite = 1:ite_max

    X_pre = X;

    %% Update AR

    % Calculate eps
    phi = X(1:d);
    theta = X(d+1:end);

    for n = d + 1:N
        E(n) = y(n) - phi' * y(n-1:-1:n-d) - theta' * E(n-1:-1:n-d);
    end

    grad = zeros(2*d, N);
    for n = d+1 + 1:N

        grad(1:d, n) = -y(n-1:-1:n-d) - grad(1:d, n-1:-1:n-d) * theta;


    end
    grad_MLE = grad(1:d, :) * E;


    X_PGM = X(1:d) - stepsize * grad_MLE;


    [X_prox, ~, ~] = BCD_func(indx, X_PGM, lambda_AR*stepsize, w, iteration_BCD);
    X_prox = X_prox(:, end);

    % Project to stationary
    p = find(X_prox, 1, 'last'); % index of last nonzero element
    if isempty(p)
        p = d;
    end

    a = d - p; % zero AR parts
    Aeq = zeros(a, d);
    beq = zeros(a, 1);

    for j = 1:a

        Aeq(j, p+j) = 1;
    end

    A = [];
    b = [];
    lb = -ones(d, 1) * 1.5;
    ub = ones(d, 1) * 1.5;

    nonlcon = @(x) Const_Sta(x, p);
    [X_proj, ~, exitflag, ~] = fmincon(@(x) norm(x-X_prox)^2, X_prox, A, b, Aeq, beq, lb, ub, nonlcon, options);

    X(1:d) = X_proj;

    %% Update MA

    % Calculate eps
    phi = X(1:d);
    theta = X(d+1:end);

    for n = d + 1:N
        E(n) = y(n) - phi' * y(n-1:-1:n-d) - theta' * E(n-1:-1:n-d);
    end

    grad = zeros(2*d, N);
    for n = d+1 + 1:N

        grad(d+1:2*d, n) = -E(n-1:-1:n-d) - grad(d+1:2*d, n-1:-1:n-d) * theta;


    end
    grad_MLE = grad(d+1:end, :) * E;


    X_PGM = X(d+1:end) - stepsize * grad_MLE;


    [X_prox, ~, ~] = BCD_func(indx, X_PGM, lambda_MA*stepsize, w, iteration_BCD);
    X_prox = X_prox(:, end);

    % Project to invertible
    q = find(X_prox, 1, 'last'); % index of last nonzero element
    if isempty(q)
        q = d;
    end

    a = d - q; % zero MA parts
    Aeq = zeros(a, d);
    beq = zeros(a, 1);

    for j = 1:a

        Aeq(j, q+j) = 1;
    end

    A = [];
    b = [];
    lb = -ones(d, 1) * 1.5;
    ub = ones(d, 1) * 1.5;

    nonlcon = @(x) Const_Inv(x, q);
    [X_proj, ~, exitflag, ~] = fmincon(@(x) norm(x-X_prox)^2, X_prox, A, b, Aeq, beq, lb, ub, nonlcon, options);
    X(d+1:2*d) = X_proj;
        X_save(:,ite)=X;

    if norm(X-X_pre) < crt_stop
        X_save=X_save(:,1:ite);
        err = sqrt(sum((X_save-X_save(:,end)).^2,1));
        break
    end

end
        err = sqrt(sum((X_save-X_save(:,end)).^2,1));


%% ------------------------------------------------------------------------
% Constraint of stationary
    function [c, ceq] = Const_Sta(X, p)
        AR = roots([1, -X(1:p)']);

        c = max(abs(AR)).^2 - 1;

        ceq = [];

    end

%% ------------------------------------------------------------------------
% Constraint of invertible
    function [c, ceq] = Const_Inv(X, q)
        MA = roots([1, X(1:q)']);

        c = max(abs(MA)).^2 - 1;

        ceq = [];

    end

end
