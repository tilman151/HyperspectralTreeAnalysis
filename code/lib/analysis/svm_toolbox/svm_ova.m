function [f,SX,SY,SA,t] = svm_ova(X,Y,N,C1,C2,k)
% SVM_OVA Solves the multiclass problem with SVMs (one vs all).
%
% Input:
%       X(n,d)  = Input data points
%       Y(n,1)  = Input data labels
%       N(~,d)  = Input data points (neutral)
%       C1(1,1) = Misclassification penalty (labeled data)
%       C2(1,1) = Misclassification penalty (neutral data)
%       k = Kernel function handle
%
% Output:
%       f = Classification function handle
%       SX(s,d) = Support vector points
%       SY(s,1) = Support vector labels
%       SA(s,1) = Support vector alphas (Lagrange multipliers)
%       t(1,1)  = Computation time in seconds

% Start timer
t_s = tic;

% Get input size
[n,d] = size(X);

% Prepare algorithm
c_list = unique(Y);
c_size = size(c_list,1);
f_list = cell(c_size,1);

SX = []; % Support vector points
SY = []; % Support vector labels
SA = []; % Support vector alphas

% Perform algorithm
for i = 1 : c_size,
    % Prepare data
    [X1,X2] = dsep(c_list(i));

    % Learn classifier
    [f,SX_new,SY_new,SA_new,~] = svm_trn(X1,X2,N,C1,C2,k);

    % Store classifier
    f_list{i} = f;

    % Add support vectors
    SX = [SX;SX_new];
    SY = [SY;SY_new];
    SA = [SA;SA_new];
end


    %%%%%%%%%%%%%%%%%%%%%
    % Define classifier %
    %%%%%%%%%%%%%%%%%%%%%

    function y = f_fun(x)
        % Reset maximum
        max_j = 1; max_v = f_list{1}(x);

        % Fetch maximum
        for j = 2 : c_size,
            % Fetch value
            v = f_list{j}(x);

            % Check value
            if v > max_v,
                max_j = j ;
                max_v = v ;
            end
        end

        % Assign class
        y = c_list(max_j);
    end


% Assign classifier
f = @f_fun;

% Stop timer
t = toc(t_s);


%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary function %
%%%%%%%%%%%%%%%%%%%%%%

    function [X1,X2] = dsep(y)
        % Prepare separation
        c0 = sum(Y == y);
        X1 = zeros(c0,d); c1 = 0;
        X2 = zeros(n-c0,d); c2 = 0;

        % Perform separation
        for q = 1 : n,
            if Y(q) == y,
                c1 = c1 + 1; X1(c1,:) = X(q,:);
            else
                c2 = c2 + 1; X2(c2,:) = X(q,:);
            end
        end
    end

end