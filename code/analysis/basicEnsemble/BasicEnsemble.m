classdef BasicEnsemble < Classifier
    %BASICENSEMBLE Classifier that uses other classifiers to predict labels
    %
    %% Properties:
    %    baseClassifier ............... Classifiers which are used to
    %                                   predict the label
    %    trainingInstanceProportions .. the proportion, which is
    %                                   selected from the given training
    %                                   instances for a specific classifier
    %
    %% Methods:
    %    BasicEnsemble ...... Constructor.
    %       Input:
    %           baseClassifier .............. a Nx1 cell array with
    %                                         instantiated classifier 
    %           numClassifier ............... a Nx1 array with the
    %                                         number of copies for each
    %                                         base classifier
    %           trainingInstanceProportions . a Nx1 array with a
    %                                         number betwee 0 and 1. For
    %                                         example a 0.8 says that all
    %                                         classifier are trained with
    %                                         80% of the training instances
    %           
    %           
    %    trainOn .... See documentation in superclass Classifier.
    %    classifyOn . See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Tuan Pham Minh
    %
    
    properties
        baseClassifier;
        trainingInstanceProportions;
    end
    
    methods
        function obj = BasicEnsemble(baseClassifier, numClassifier, ...
                trainingInstanceProportions)
            
            % generate indices to copy the baseclassifier 
            % calculate the points, where new classifier starts
            classifierIndexStartPoint = ...
                [1 (cumsum(numClassifier(1:end-1)) + 1)];
            % calculate the indices, which is then used to copy the objects
            classifierIndices = zeros(1, sum(numClassifier));
            classifierIndices(classifierIndexStartPoint) = 1;
            classifierIndices = cumsum(classifierIndices);
            % copy the classifier
            obj.baseClassifier = cellfun(@copy, ...
                baseClassifier(classifierIndices), ...
                'UniformOutput', 0)';
            obj.trainingInstanceProportions = ...
                num2cell(trainingInstanceProportions(classifierIndices));
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            [x,y,spectralBands] = size(trainFeatureCube);
            
            % generate randomized indices
            randomizedIndices = cellfun(@createRandPerm, repmat({x*y}, ...
                1, length(obj.trainingInstanceProportions)), ...
                obj.trainingInstanceProportions, 'UniformOutput', 0)';
            
            % make features 2D
            reshapedTrainFeatures = ...
                repmat({reshape(trainFeatureCube, x*y, spectralBands)}, ...
                length(obj.trainingInstanceProportions), 1);
            reshapedTrainLabels = ...
                repmat({reshape(trainLabelMap, x*y, 1)}, ...
                length(obj.trainingInstanceProportions), ...
                1);
            
            % select the features from the random permutation
            randomizedTrainFeatures = ...
                cellfun(@(x,y)(permute(x(y,:), [1 3 2])), ...
                reshapedTrainFeatures, ...
                randomizedIndices, ...
                'UniformOutput', 0);
            randomizedTrainLabels = ...
                cellfun(@(x,y)(x(y)), ...
                reshapedTrainLabels, ...
                randomizedIndices, ...
                'UniformOutput', 0);
            
            % train all classifiers
            cellfun(@trainBaseClassifier, ...
                obj.baseClassifier, ...
                randomizedTrainFeatures, ...
                randomizedTrainLabels, ...
                'UniformOutput', 0);
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % Create classify function call handle
            classifyHandle = @(classifier) classifier.classifyOn(...
                evalFeatureCube, maskMap);
            
            % Classify all instances
            accumulatedLabels = cell2mat(cellfun(...
                classifyHandle, obj.baseClassifier, 'UniformOutput', 0)');
            
            % TODO: Does not yet work for the returned label maps
            % Should create a matrix of dimensions X x Y x C with C being
            % the number of base classifiers
            
            % Calculate the majority vote
            predictedLabelMap = mode(accumulatedLabels, 3);
        end
    end
    
end

