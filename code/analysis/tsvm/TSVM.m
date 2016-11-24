classdef TSVM < Classifier
    %TSVM transductive Support Vector Machine
    %
    %    Detailed explanation goes here
    %
    %% Properties:
    %    C1 ......... 
    %    C2 ......... 
    %    k .......... 
    %    f .......... 
    %    SX ......... 
    %    SY ......... 
    %    SA ......... 
    %    t .......... 
    %
    %% Methods:
    %    TSVM ....... 
    %    trainOn .... 
    %    classifyOn . 
    %
    % Version: 2016-11-24
    % Author: Cornelius Styp von Rekowski
    %
    
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
        function obj = TSVM(C1, C2, k)
            obj.C1 = C1;
            obj.C2 = C2;
            obj.k = k;
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            N = []; % Unlabeled samples
            
            [obj.f, obj.SX, obj.SY, obj.SA, obj.t] = ...
                svm_ovo(trainFeatures, trainLabels, N, ...
                        obj.C1, obj.C2, obj.k);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            labels = obj.f(evalFeatures);
        end
    end
    
end

