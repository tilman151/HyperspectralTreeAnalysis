classdef BasicEnsemble < Classifier
    %Classifier which uses other classifier to predict a label
    %
    %% Properties:
    %    baseClassifier ............... Classifier which are used to
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
    % Version: 2016-12-05
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
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            [x,y,spectralBands] = size(trainFeatures);
            % generate randomized indices
            repmat({x*y}, 1, length(obj.trainingInstanceProportions));
            randomizedIndices = cellfun(@createRandPerm, repmat({x*y}, ...
                1, length(obj.trainingInstanceProportions)), ...
                obj.trainingInstanceProportions, 'UniformOutput', 0)';
            % make features 2D
            reshapedTrainFeatures = ...
                repmat({reshape(trainFeatures, x*y, spectralBands)}, ...
                length(obj.trainingInstanceProportions), 1);
            reshapedTrainLabels = ...
                repmat({reshape(trainLabels, x*y, 1)}, ...
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
            % train all classifier
            cellfun(@trainBaseClassifier, ...
                obj.baseClassifier, ...
                randomizedTrainFeatures, ...
                randomizedTrainLabels, ...
                'UniformOutput', 0);
        end
        
        function labels = classifyOn(obj,evalFeatures)
            % make a copy for all classifier
            reshapedEvalFeatures = ...
                repmat({evalFeatures}, ...
                length(obj.trainingInstanceProportions), ...
                1);
            % classify all instances
            accumulatedLabels = ...
                cell2mat( ...
                    cellfun(@classifyOnAll, ...
                        obj.baseClassifier, ...
                        reshapedEvalFeatures, ...
                        'UniformOutput', 0)');
            % calculate the majority vote
            labels = mode(accumulatedLabels,2);
        end
    end
    
end

