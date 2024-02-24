function [X,const, info, opts] = FIT_SARMA(y, lambda_AR, lambda_MA, X0, opts)

% assume constant of ARMA is 0
%
% Input
%   Y(N,1)   sample data
%   lambda_AR   Penalty parameter for AR
%   lambda_MA   Penalty parameter for MA
%   X0  initial point
%   opts.Pmax   maximum order of AR  default set is 0.75*sqrt(N)
%   opts.Qmax   maximum order of MA
%   opts.itermax    maximum iteration number
%   opts.stepsize   initial stepsize
%   stepsize stepsize for PGM

%
% Output
%   X   estimated ARMA parameter

%% Initialize
N = length(y);

defaultopts.Pmax = floor(0.75*sqrt(N));
defaultopts.Qmax = floor(0.75*sqrt(N));
defaultopts.itermax = 500;
defaultopts.stepsize = 0.01;
defaultopts.beta = 0.8;
defaultopts.stopcrt = 1e-4;
iteration_BCD = 10;

if nargin < 5
    opts = defaultopts;
    if nargin < 4
        X0 = [zeros(defaultopts.Pmax, 1); zeros(defaultopts.Qmax, 1)];
    end
else
    opts = load_struct_vars(opts, defaultopts);
end
options_fmincon = optimoptions('fmincon', 'Algorithm', 'sqp', 'ConstraintTolerance', 1e-7, 'Display', 'off');


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

%% calculate y_mu and subtract it from y
y_mu = mean(y);
y = y-y_mu;
%% PGM
X = X0;
t_AR = opts.stepsize;
t_MA = opts.stepsize;

for iter = 1:opts.itermax
    X_pre = X;

    %% Update AR
    phi = X(1:opts.Pmax);
    theta = X(opts.Pmax+1:opts.Pmax+opts.Qmax);
    E = zeros(N, 1);
    for n = opts.Pmax + 1:N

        E(n) = y(n) - phi' * y(n-1:-1:n-opts.Pmax) - theta' * E(n-1:-1:n-opts.Pmax);
    end
    grad = zeros(opts.Pmax+opts.Qmax, N);
    for n = opts.Pmax + 1 + 1:N

        grad(1:opts.Pmax, n) = -y(n-1:-1:n-opts.Pmax) - grad(1:opts.Pmax, n-1:-1:n-opts.Pmax) * theta;


    end
    grad_MLE = grad(1:opts.Pmax, :) * E;

    % line search
    f = norm(E)^2 / 2;

    while (1)

        X_PGM = X(1:opts.Pmax) - t_AR * grad_MLE;
        [X_prox, ~, ~] = BCD_func(indx_AR, X_PGM, lambda_AR*t_AR, w_AR, iteration_BCD);
        X_prox = X_prox(:, end);


        Gt = (X(1:opts.Pmax) - X_prox) / t_AR;
        E_new = zeros(N, 1);
        for n = opts.Pmax + 1:N
            E_new(n) = y(n) - X_prox' * y(n-1:-1:n-opts.Pmax) - theta' * E_new(n-1:-1:n-opts.Pmax);
        end
        f_new = norm(E_new)^2 / 2;
        if f_new <= f - t_AR * grad_MLE' * Gt + t_AR / 2 * norm(Gt)^2
            info.g_AR(iter) = f_new;
            info.stepsize(iter) = t_AR;
            info.grad(iter) = norm(grad_MLE);
            break
        else
            t_AR = t_AR * opts.beta;
        end
        if norm(t_AR*grad_MLE) < 1e-6
            break
        end


    end

    % Project to stationary
    p = find(X_prox, 1, 'last'); % index of last nonzero element
    if isempty(p)
        p = opts.Pmax;
    end

    a = opts.Pmax - p; % zero AR parts
    Aeq = zeros(a, opts.Pmax);
    beq = zeros(a, 1);

    for j = 1:a

        Aeq(j, p+j) = 1;
    end

    A = [];
    b = [];
    lb = -ones(opts.Pmax, 1) * 1.5;
    ub = ones(opts.Pmax, 1) * 1.5;

    nonlcon = @(x) Const_Sta(x, p);
    [X_proj, ~, exitflag_AR, ~] = fmincon(@(x) norm(x-X_prox)^2, X_prox, A, b, Aeq, beq, lb, ub, nonlcon, options_fmincon);
    % X_proj = X_prox;
    X(1:opts.Pmax) = X_proj;

    %% Update MA

    phi = X(1:opts.Pmax);
    theta = X(opts.Pmax+1:opts.Pmax+opts.Qmax);
    E = zeros(N, 1);
    for n = opts.Pmax + 1:N
        E(n) = y(n) - phi' * y(n-1:-1:n-opts.Pmax) - theta' * E(n-1:-1:n-opts.Pmax);
    end

    grad = zeros(opts.Pmax+opts.Qmax, N);
    for n = opts.Pmax + 1 + 1:N

        grad(opts.Pmax+1:2*opts.Pmax, n) = -E(n-1:-1:n-opts.Pmax) - grad(opts.Pmax+1:2*opts.Pmax, n-1:-1:n-opts.Pmax) * theta;


    end
    grad_MLE = grad(opts.Pmax+1:end, :) * E;
    f = norm(E)^2 / 2;
    while (1)

        X_PGM = X(opts.Pmax+1:end) - t_MA * grad_MLE;

        [X_prox, ~, ~] = BCD_func(indx_MA, X_PGM, lambda_MA*t_MA, w_MA, iteration_BCD);
        X_prox = X_prox(:, end);


        Gt = (X(opts.Pmax+1:end) - X_prox) / t_MA;
        E_new = zeros(N, 1);
        for n = opts.Pmax + 1:N
            E_new(n) = y(n) - phi' * y(n-1:-1:n-opts.Pmax) - X_prox' * E_new(n-1:-1:n-opts.Pmax);
        end
        f_new = norm(E_new)^2 / 2;
        if f_new <= f - t_MA * grad_MLE' * Gt + t_MA / 2 * norm(Gt)^2
            info.g_MA(iter) = f_new;
            info.stepsize_MA(iter) = t_MA;
            info.grad_MA(iter) = norm(grad_MLE);

            break
        else
            t_MA = t_MA * opts.beta;
        end

        if norm(t_AR*grad_MLE) < 1e-6
            break
        end
    end
    % Project to invertible
    q = find(X_prox, 1, 'last'); % index of last nonzero element
    if isempty(q)
        q = opts.Qmax;
    end

    a = opts.Qmax - q; % zero MA parts
    Aeq = zeros(a, opts.Qmax);
    beq = zeros(a, 1);

    for j = 1:a

        Aeq(j, q+j) = 1;
    end

    A = [];
    b = [];
    lb = -ones(opts.Qmax, 1) *1.5;
    ub = ones(opts.Qmax, 1) * 1.5;

    nonlcon = @(x) Const_Inv(x, q);
    [X_proj, ~, exitflag_MA, ~] = fmincon(@(x) norm(x-X_prox)^2, X_prox, A, b, Aeq, beq, lb, ub, nonlcon, options_fmincon);
    %X_proj = X_prox;
    X(opts.Pmax+1:end) = X_proj;

    if norm(X-X_pre) < 1e-3
        break
    end
end
info.exitflag = [exitflag_AR, exitflag_MA];
const = y_mu*(1-sum(X(1:opts.Pmax)));

%% Constraint of stationary
    function [c, ceq] = Const_Sta(X, p)
        AR = roots([1, -X(1:p)']);

        c = max(abs(AR)) - 0.95;

        ceq = [];

    end

%% ------------------------------------------------------------------------
% Constraint of invertible
    function [c, ceq] = Const_Inv(X, q)
        MA = roots([1, X(1:q)']);

        c = max(abs(MA)) - 0.95;

        ceq = [];

    end

end