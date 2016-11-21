function [acc,prm,t] = svm_hps(X1,Y1,N,X2,Y2,mul,gen,arg,gsize,depth)
% SVM_HPS Learnes a TSVM model based on the given training data,
% multi-mode, kernel, and kernel parameters and determines its
% accuracy by classifying the given test data.
%
% If size(arg,2) == 1, then arg specifies the parameters
% C1 = arg(1) (misclassification penalty for labeled data)
% C2 = arg(2) (misclassification penalty for neutral data)
% and the kernel function parameters in arg(3:end,1).
%
% If size(arg,2) == 2, then arg specifies the parameter ranges with
% min_i = arg(i,1) and max_i = arg(i,2) for each parameter p_i where
% p_1 = C1, p_2 = C2 and p_3... specify the kernel parameters. An
% optimal parameter setting will be searched with a grid-based
% heuristic search.
%
% Input:
%       X1(n1,2) = Training data points
%       Y1(n1,1) = Training data labels
%       N (n2,2) = Training data points (neutral)
%       X2(n3,2) = Test data points
%       Y2(n3,1) = Test data labels
%       mul = Multiclass SVM function handle
%       gen = Kernel generator function
%       arg(d1,d2) = Kernel generator parameters (or ranges)
%       gsize(1,1) = Grid size
%       depth(1,1) = Search depth
%
% Output:
%       acc(1,1) = Accuracy (0.0 to 1.0)
%       prm(d1,d2) = Parameter setting
%       t(1,1) = Computation time in seconds

% Prepare accuracy function
n3 = size(X2,1); t_range = 1 : n3;

% Check arguments
[d1,d2] = size(arg);

if d2 == 1,
    % Generate kernel
    k = gen(arg(3:d1,1));

    % Learn classifier
    [f,~,~,~,t] = mul(X1,Y1,N,arg(1),arg(2),k);

    % Determine accuracy
    acc = accuracy(f); prm = arg;
else
    % Start timer
    t_s = tic;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Prepare parameter search %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Verify unlabeled data
    if size(N,1) == 0,
        karg = 2; arg = [arg(1,:);arg(3:d1,:)]; d1 = d1 - 1;
    else
        karg = 3;
    end

    % Set search variables
    prmgrd = zeros(d1,gsize);
    optprm = zeros(d1,1);
    tmpprm = optprm;
    optacc = -1;

    prm = ones(d1,1);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Perform parameter search %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for d = 1 : depth,

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 1: Instantiate parameter grid %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for i = 1 : d1,
            % Prepare instantiation
            range = (arg(i,2) - arg(i,1)) / 2 ;
            start = arg(i,1) + range / 2 ;
            width = range / (gsize-1) ;

            % Perform instantiation
            for j = 1 : gsize,
                prmgrd(i,j) = start + (j-1) * width ;
            end
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 2: Search optimal parameters %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Perform search
        optimum(1);

        % Assign optimum
        acc = optacc ;
        prm = optprm ;

        % Check accuracy
        if acc == 1,
            break;
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 3: Update parameter ranges %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for i = 1 : d1,
            % Prepare centering
            bias = (arg(i,2) - arg(i,1)) / 4 ;

            % Perform centering
            arg(i,1) = optprm(i) - bias ;
            arg(i,2) = optprm(i) + bias ;
        end
    end

    % Include C2 value
    if karg == 2,
        prm = [prm(1);0;prm(2:end,1)];
    end

    % Stop timer
    t = toc(t_s);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimum function %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function optimum(pos)
        if pos > d1,
            if max(tmpprm ~= prm),
                % Learn classifier
                C1 = tmpprm(1);

                if karg == 3,
                    C2 = tmpprm(2);
                else
                    C2 = 0;
                end

                [f,~,~,~,~] = mul(X1,Y1,N,C1,C2,gen(tmpprm(karg:d1,1)));

                % Determine accuracy
                tmpacc = accuracy(f);

                if tmpacc > optacc,
                    optacc = tmpacc ;
                    optprm = tmpprm ;
                end
            end
        else
            % Recursive callback
            for q = 1 : gsize,
                % Set this parameter
                tmpprm(pos) = prmgrd(pos,q);

                % Set next parameter
                optimum(pos+1);

                % Check accuracy
                if optacc == 1,
                    break;
                end
            end
        end
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define accuracy function %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function a = accuracy(f)
        % Prepare counter
        c = 0;

        % Check test data
        for q = t_range,
            if f(X2(q,:)) == Y2(q),
                c = c + 1;
            end
        end

        % Determine accuracy
        a = c / n3 ;
    end

end