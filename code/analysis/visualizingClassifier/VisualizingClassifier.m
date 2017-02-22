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
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-01-09
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
    end
    
    methods
        function str = toString(obj)
            str = 'VisualizingClassifier';
        end
        
        function str = toShortString(obj)
            str = 'VisualizingClassifier';
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Display labels
            visualizeLabels(trainLabelMap, 'Training Labels');
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Return pseudo output
            predictedLabelMap = maskMap;
        end
    end
    
end

