classdef SVMliblinear < Classifier
    %SVMliblinear Linear Support Vector Machine for multiclass problems
    %
    %    This class uses LIBLINEAR as its underlying SVM implementation. 
    %    The default strategy for multiclass classification is one vs. all.
    %    See the LIBLINEAR README in the corresponding lib directory for 
    %    detailed installation information.
    %
    %% Methods:
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2017-02-23
    % Author: Cornelius Styp von Rekowski
    %
    
    properties(Hidden=true)
        % Trained model
        model;
    end
    
    methods
        function str = toString(~)
            % Create output string with class name
            str = 'SVM [LIBLINEAR]';
        end
        
        function str = toShortString(~)
            % Create output string with class name and kernel function
            str = 'SVMliblinear';
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Extract labeled pixels
            featureList = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, true);
            labelList = validListFromSpatial(...
                trainLabelMap, trainLabelMap, true);
            
            % Make feature list sparse
            featureList = sparse(featureList);
            
            % Train model
            obj.model = train(labelList, featureList, '-q');
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Extract list of unlabeled pixels
            featureList = validListFromSpatial(evalFeatureCube, maskMap);
            labelList = zeros(size(featureList, 1), 1);
            
            % Make feature list sparse
            featureList = sparse(featureList);
            
            % Predict labels
            predictedLabelList = ...
                predict(labelList, featureList, obj.model);
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
    end
    
end

