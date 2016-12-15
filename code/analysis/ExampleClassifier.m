classdef ExampleClassifier < Classifier
    %EXAMPLECLASSIFIER Example for classifier inheriting from Classifier
    %Uses standard classify function
    
    properties
        trainingFeatures;
        trainingLabels;
    end
    
    methods
        function obj = trainOn(obj,trainLabels, trainFeatures)
            obj.trainingFeatures= permute(reshape(trainFeatures, [], 1, 160), [1 3 2]);
            obj.trainingLabels = reshape(trainLabels, [], 1);
        end
        
        function labels = classifyOn(obj,evalFeatures, mask)
            labels = classify(permute(reshape(evalFeatures, [], 1, 160), [1 3 2]),...
                              obj.trainingFeatures,...
                              obj.trainingLabels);
        end
    end
    
end

