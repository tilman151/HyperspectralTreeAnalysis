function [f,SX,SY,SA,t] = svm_ind(X,Y,C,k)
% SVM_IND Computes a separating hyperplane (inductive SVM).
%
% Input:
%       X(n,d) = Input data points (d-dimensional)
%       Y(n,1) = Input data labels (data labeling)
%       C(n,1) = Misclassification penalty factors
%       k = Kernel function handle
%
% Output:
%       f = Classification function handle (f(x) = k(w,x)+b)
%       SX(s,d) = Support vector points
%       SY(s,1) = Support vector labels
%       SA(s,1) = Support vector alphas (Lagrange multipliers)
%       t(1,1)  = Computation time in seconds

% Start timer
t_s = tic;

%%%%%%%%%%%%%%%%%%%%%%%%
% Perform optimization %
%%%%%%%%%%%%%%%%%%%%%%%%

[n,d] = size(X);

Z = zeros(n,1);
H = zeros(n,n);

for i = 1 : n,
    for j = i : n,
        H(i,j) = Y(i)*Y(j)*k(X(i,:),X(j,:));
    end
end

A = quadprog(H+triu(H,1)',-ones(n,1),[],[],[Y';zeros(n-1,n)],Z,Z,C,[],optimset('Display','off','LargeScale','off'));


%%%%%%%%%%%%%%%%%%%%%%%
% Get support vectors %
%%%%%%%%%%%%%%%%%%%%%%%

SX = zeros(n,d);
SY = zeros(n,1);
SA = zeros(n,1);

min_a = 10^-10;

s = 0;

for i = 1 : n,
    if A(i) > min_a && (A(i) + min_a) < C(i),
        % Increment index
        s = s + 1;

        % Store support vector
        SX(s,:) = X(i,:);
        SY(s,1) = Y(i,1);
        SA(s,1) = A(i,1);
    end
end

s_range = 1 : s ;

SX = SX(s_range,:);
SY = SY(s_range,1);
SA = SA(s_range,1);


%%%%%%%%%%%%%
% Compute b %
%%%%%%%%%%%%%

b = 0;

for i = s_range,
    b = b + SY(i);

    for j = s_range,
        b = b - SA(j)*SY(j)*k(SX(j,:),SX(i,:));
    end
end

b = b/s;


%%%%%%%%%%%%%
% Compute f %
%%%%%%%%%%%%%

    function y = f_fun(x)
        % Compute class
        y = b;

        for q = s_range,
            y = y + SA(q)*SY(q)*k(SX(q,:),x);
        end
    end


% Assign classifier
f = @f_fun;

% Stop timer
t = toc(t_s);

end