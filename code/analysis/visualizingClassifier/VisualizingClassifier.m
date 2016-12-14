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
    
    properties
    end
    
    methods
        function obj = trainOn(obj, trainFeatures, trainLabels)
            % Display feature map if it is 2-dimensional
            if length(size(trainFeatures)) == 2
                visualizeLabels(trainFeatures, 'Training Features');
            end
            
            % Display labels
            visualizeLabels(trainLabels, 'Training Labels');
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Display feature map if it is 2-dimensional
            if length(size(trainFeatures)) == 2
                visualizeLabels(trainFeatures, 'Evaluation Features');
            end
        end
    end
    
end

