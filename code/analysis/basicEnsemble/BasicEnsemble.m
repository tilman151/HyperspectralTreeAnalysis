classdef BasicEnsemble < Classifier
    %BASICENSEMBLE Classifier that uses other classifiers to predict labels
    %
    %% Properties:
    %    baseClassifiers ............... Classifiers which are used to
    %                                    predict the label
    %    trainingInstanceProportions ... The proportion, which is
    %                                    selected from the given training
    %                                    instances for a specific 
    %                                    classifier
    %
    %% Methods:
    %    BasicEnsemble ...... Constructor.
    %       Input:
    %           baseClassifiers .............. a Nx1 cell array with
    %                                          instantiated classifiers
    %           numClassifiers ............... a Nx1 array with the
    %                                          number of copies for each
    %                                          base classifier
    %           trainingInstanceProportions .. a Nx1 array with a
    %                                          number between 0 and 1. For
    %                                          example a 0.8 says that all
    %                                          classifiers are trained with
    %                                          80% of the training 
    %                                          instances
    %           
    %           
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Tuan Pham Minh
    %
    
    properties
        baseClassifiers;
        trainingInstanceProportions;
    end
    
    methods
        function obj = BasicEnsemble(baseClassifiers, numClassifiers, ...
                trainingInstanceProportions)
            
            % generate indices to copy the baseclassifiers
            % calculate the points, where new classifier starts
            classifierIndexStartPoint = ...
                [1 (cumsum(numClassifiers(1:end-1)) + 1)];
            % calculate the indices, which is then used to copy the objects
            classifierIndices = zeros(1, sum(numClassifiers));
            classifierIndices(classifierIndexStartPoint) = 1;
            classifierIndices = cumsum(classifierIndices);
            % copy the classifier
            obj.baseClassifiers = cellfun(@copy, ...
                baseClassifiers(classifierIndices), ...
                'UniformOutput', 0)';
            obj.trainingInstanceProportions = ...
                num2cell(trainingInstanceProportions(classifierIndices))';
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % make features 2D
            reshapedTrainFeatures = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, false);
            reshapedTrainLabels = validListFromSpatial(...
                trainLabelMap, trainLabelMap, false);
            
            % embed the features and labels inside of the anonymous
            % function
            trainFunction = @(classifier, proportion) ...
                trainBaseClassifier( classifier, ...
                                     reshapedTrainFeatures , ...
                                     reshapedTrainLabels, ...
                                     proportion);
                                 
            % train all classifiers
            cellfun(trainFunction, ...
                obj.baseClassifiers, ...
                obj.trainingInstanceProportions, ...
                'UniformOutput', 0);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Create classify function call handle
            classifyHandle = @(classifier) classifier.classifyOn(...
                evalFeatureCube, maskMap);
            
            % Classify all instances
            accumulatedLabels = cellfun(...
                classifyHandle, obj.baseClassifiers, 'UniformOutput', 0)';
            % concatenate the labels along the third dimension
            concatenatedLabels = cat(3, accumulatedLabels{:});
            % Calculate the majority vote
            predictedLabelMap = mode(concatenatedLabels, 3);
            
        end
    end
    
end

