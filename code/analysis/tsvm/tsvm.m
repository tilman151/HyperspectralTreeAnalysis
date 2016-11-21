classdef tsvm
    %TSVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Parameters
        C1 = 10; % Misclassification penalty (labeled data)
        C2 = 10; % Misclassification penalty (neutral data)
        k = kernel_gen_rbf; % Kernel function handle
        
        % Output results
        f;  % Classification function handle
        SX; % Support vector points
        SY; % Support vector labels
        SA; % Support vector alphas (Lagrange multipliers)
        t;  % Computation time in seconds
    end
    
    methods
        function obj = tsvm(C1, C2, k)
            obj.C1 = C1;
            obj.C2 = C2;
            obj.k = k;
        end
        
        function obj = trainOn(obj, X, y)
            [obj.f, obj.SX, obj.SY, obj.SA, obj.t] = ...
                svm_ovo(X, y, N, obj.C1, obj.C2, obj.k);
        end
        
        function [obj, label] = predict(obj, X)
            label = obj.f(X);
        end
    end
    
end

