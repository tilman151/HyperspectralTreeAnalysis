classdef SVM < Classifier
    %SVM Standard Support Vector Machine
    %   Detailed explanation goes here
    
    properties
        % Trained model
        model;
    end
    
    methods
        function obj = trainOn(obj, trainFeatures, trainLabels)
            [featureList, labelList] = ...
                extractLabeledPixels(trainFeatures, trainLabels);
            obj.model = fitcsvm(featureList, labelList);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            featureList = mapToVec(evalFeatures);
            labels = obj.model.predict(featureList);
        end
    end
    
end

