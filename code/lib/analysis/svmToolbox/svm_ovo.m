function [f,SX,SY,SA,t] = svm_ovo(X,Y,N,C1,C2,k)
% SVM_OVO Solves the multiclass problem with SVMs (one vs one).
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
f_size = c_size * (c_size - 1) / 2 ;
f_list = cell(f_size,3);

SX = []; % Support vector points
SY = []; % Support vector labels
SA = []; % Support vector alphas

m = 0;

% Perform algorithm
for i = 1 : c_size - 1,
    for j = i + 1 : c_size,
        % Prepare data
        [X1,X2] = dext(c_list(i),c_list(j));

        % Learn classifier
        [f,SX_new,SY_new,SA_new,~] = svm_trn(X1,X2,N,C1,C2,k);

        % Store classifier
        m = m + 1;
        f_list{m,1} = i ;
        f_list{m,2} = j ;
        f_list{m,3} = f ;

        % Add support vectors
        SX = [SX;SX_new];
        SY = [SY;SY_new];
        SA = [SA;SA_new];
    end
end


    %%%%%%%%%%%%%%%%%%%%%
    % Define classifier %
    %%%%%%%%%%%%%%%%%%%%%

    votes = zeros(1,c_size);

    function y = f_fun(x)
        % Reset votes
        votes = votes * 0;

        % Count votes
        for q = 1 : f_size,
            % Fetch value
            v = f_list{q,3}(x);

            % Check value
            if v > 0,
                v_index = f_list{q,1};
            else
                v_index = f_list{q,2};
            end

            % Add voting
            votes(v_index) = votes(v_index) + 1;
        end

        % Check votes
        max_q = 1; max_v = votes(1);

        for q = 2 : c_size,
            if votes(q) > max_v,
                max_v = votes(q); max_q = q;
            end
        end

        % Assign class
        y = c_list(max_q);
    end


% Assign classifier
f = @f_fun;

% Stop timer
t = toc(t_s);


%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary function %
%%%%%%%%%%%%%%%%%%%%%%

    function [X1,X2] = dext(y1,y2)
        % Prepare extraction
        X1 = zeros(sum(Y == y1),d); c1 = 0;
        X2 = zeros(sum(Y == y2),d); c2 = 0;

        % Perform extraction
        for q = 1 : n,
            if Y(q) == y1,
                c1 = c1 + 1; X1(c1,:) = X(q,:);
            elseif Y(q) == y2,
                c2 = c2 + 1; X2(c2,:) = X(q,:);
            end
        end
    end

end