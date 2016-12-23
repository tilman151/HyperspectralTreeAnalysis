classdef VisualizingClassifier < Classifier
    %VISUALIZINGCLASSIFIER Simply displays the data given to it
    %
    %    Classifier implementation that simply displays the data that is
    %    given to it. Label maps are always displayed, feature maps are 
    %    only displayed in the case of one-dimensional features.
    %    It does not perform any classification, so it can only be used for
    %    debugging purposes!
    %
    %% Methods:
    %    trainOn ...... See documentation in superclass Classifier.
    %    classifyOn ... See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
    end
    
    methods
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Display feature map if it is 2-dimensional
            if length(size(trainFeatureCube)) == 2
                visualizeLabels(trainFeatureCube, 'Training Features');
            end
            
            % Display labels
            visualizeLabels(trainLabelMap, 'Training Labels');
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Display feature map if it is 2-dimensional
            if length(size(evalFeatureCube)) == 2
                visualizeLabels(evalFeatureCube, 'Evaluation Features');
            end
            
            % Return pseudo output
            predictedLabelMap = maskMap;
        end
    end
    
end

