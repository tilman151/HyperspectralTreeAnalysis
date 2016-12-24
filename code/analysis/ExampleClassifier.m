classdef ExampleClassifier < Classifier
    %EXAMPLECLASSIFIER Example for classifier inheriting from Classifier
    %    
    %    Uses standard classify function.
    
    properties
        trainFeatureList;
        trainLabelList;
    end
    
    methods
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Extract valid pixels from the given data to lists
            obj.trainFeatureList = ...
                validListFromSpatial(trainFeatureCube, trainLabelMap, true);
            obj.trainLabelList = ...
                validListFromSpatial(trainLabelMap, trainLabelMap, true);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Extract valid pixels from the given feature cube to a list
            evalFeatureList = validListFromSpatial(evalFeatureCube, maskMap);
            
            % Classify the given data
            predictedLabelList = classify(...
                evalFeatureList, obj.trainFeatureList, obj.trainLabelList);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

