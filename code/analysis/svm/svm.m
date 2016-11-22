classdef svm < Classifier
    %SVM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Trained model
        model;
    end
    
    methods
        function obj = trainOn(obj, trainFeatures, trainLabels)
            obj.model = fitcsvm(trainFeatures, trainLabels);
        end
        
        function labels = classifiyOn(obj, evalFeatures)
            labels = predict(obj.model, evalFeatures);
        end
    end
    
end

