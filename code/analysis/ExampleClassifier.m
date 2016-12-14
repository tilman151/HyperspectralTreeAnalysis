classdef ExampleClassifier < Classifier
    %EXAMPLECLASSIFIER Example for classifier inheriting from Classifier
    %Uses standard classify function
    
    properties
        trainingFeatures;
        trainingLabels;
    end
    
    methods
        function obj = trainOn(obj,trainFeatures,trainLabels)
            obj.trainingFeatures= trainFeatures;
            obj.trainingLabels = trainLabels;
        end
        
        function labels = classifyOn(obj,evalFeatures)
            labels = classify(evalFeatures,...
                              obj.trainingFeatures,...
                              obj.trainingLabels);
        end
    end
    
end

