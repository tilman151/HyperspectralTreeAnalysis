function [f,SX,SY,SA,t] = svm_trn(X1,X2,X3,C1,C2,k)
% SVM_TRN Computes a separating hyperplane (transductive SVM).
%
% Input:
%       X1(~,d) = Input data points (class 1, d-dimensional)
%       X2(~,d) = Input data points (class 2, d-dimensional)
%       X3(~,d) = Input data points (class 3, d-dimensional, neutral)
%       C1(1,1) = Misclassification penalty factor (for labeled data)
%       C2(1,1) = Misclassification penalty factor (for neutral data)
%       k = Kernel function handle
%
% Output:
%       f = Classification function handle (f(x) = k(w,x)+b)
%       SX(s,d) = Support vector points
%       SY(s,1) = Support vector labels
%       SA(s,1) = Support vector alphas (Lagrange multipliers)
%       t(1,1)  = Computation time in seconds

% Get input size
n1 = size(X1,1);
n2 = size(X2,1);
n3 = size(X3,1);

% Get input data
X = [X1;X2];
Y = [ones(n1,1);-ones(n2,1)];
C = C1*ones(n1+n2,1);

% Learn initial model
[f,SX,SY,SA,t] = svm_ind(X,Y,C,k);

% Verify neutral data
if n3 > 0,
    % Start timer
    t_s = tic;

    % Sign neutral data
    Y3 = zeros(n3,1);

    for i = 1 : n3,
        Y3(i) = f(X3(i,:));
    end

    % Sort neutral data
%    for i = 1 : n3,
%        sorted = 1;
%
%        for j = 1 : n3 - 1,
%            if Y3(j) > Y3(j+1),
%                tmp_x = X3(j,:); X3(j,:) = X3(j+1,:); X3(j+1,:) = tmp_x;
%                tmp_y = Y3(j,:); Y3(j,:) = Y3(j+1,:); Y3(j+1,:) = tmp_y;
%                sorted = 0;
%            end
%        end
%
%        if sorted,
%            break;
%        end
%    end

    % Sign neutral data
    Y3 = sign(Y3);

    % Save neutral data
    X = [X;X3]; Y = [Y;Y3];


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Learn transductive model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    min_i = n1 + n2 + 1 ;
    max_i = n1 + n2 + n3 ;

    switched_ij = zeros(20,2);

    for c = -5 : 0,
        % Compute penalty factor
        C3 = (C2*10^c)*ones(n3,1);

        % Perform learning algorithm
        switched = 1; cnt = 0;

        while switched,
            % Get new classifier
            [f,SX,SY,SA,~] = svm_ind(X,Y,[C;C3],k);

            % Switch data labels
            switched = 0;

            for i = min_i : max_i - 1,
                for j = i + 1 : max_i,
                    % Fetch labels
                    y1 = Y(i);
                    y2 = Y(j);

                    % Check labels
                    if y1 ~= y2 && ~is_switched(i,j),
                        % Fetch values
                        f1 = f(X(i,:));
                        f2 = f(X(j,:));

                        % Check values
                        if max([0,1-y1*f1]) + max([0,1-y2*f2]) > max([0,1-y2*f1]) + max([0,1-y1*f2]),
                            % Switch labels
                            Y(i) = y2;
                            Y(j) = y1;

                            % Store indices
                            cnt = cnt + 1; switched_ij(cnt,:) = [i,j];

                            % Leave loop
                            switched = 1; break;
                        end
                    end
                end

                if switched,
                    break;
                end
            end
        end
    end

    % Stop timer
    t = t + toc(t_s);
end


%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary function %
%%%%%%%%%%%%%%%%%%%%%%

    function s = is_switched(i,j)
        % Prepare search
        s = 0;

        % Perform search
        if cnt > 0,
            for q = 1 : cnt,
                if switched_ij(q,1) == i && switched_ij(q,2) == j,
                    s = 1; break;
                end
            end
        end
    end

end