classdef svm
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Trained model
        model;
    end
    
    methods
        function obj = trainOn(obj, X, y)
            obj.model = fitcsvm(X, y);
        end
        
        function [obj, label] = predict(obj, X)
            label = predict(obj.model, X);
        end
    end
    
end

