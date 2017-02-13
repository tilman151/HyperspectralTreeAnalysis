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
    %           baseClassifiers ............... a Nx1 cell array with
    %                                           instantiated classifiers
    %           numClassifiers ................  a Nx1 array with the
    %                                           number of copies for each
    %                                           base classifier
    %           trainingInstanceProportions ...  a Nx1 array with a
    %                                           number between 0 and 1. For
    %                                           example a 0.8 says that all
    %                                           classifiers are trained with
    %                                           80% of the training 
    %                                           instances
    %           votingMode .................... VotingMode enum to specify
    %                                           the voting method to get to 
    %                                           a conclusion
    %           validationSubsampleProportion . VotingMode enum to specify
    %                                           the voting method to get to 
    %                                           a conclusion
    %           remainClassDistribution ....... set true if the class
    %                                           distribution for the 
    %                                           training subsamples should 
    %                                           be approximately the same
    %                                           as the original 
    %                                           distribution
    %           multithreadingTraining ........ set to true if training
    %                                           should use the
    %                                           multithreaded 
    %                                           implementation
    %           multithreadingTraining ........ set to true if
    %                                           classification should use 
    %                                           the multithreaded 
    %                                           implementation
    %           
    %           
    %    toString ...... See documentation in superclass Classifier.
    %    toShortString . See documentation in superclass Classifier.
    %    trainOn ....... See documentation in superclass Classifier.
    %    classifyOn .... See documentation in superclass Classifier.
    %
    % Version: 2016-12-22
    % Author: Tuan Pham Minh
    %
    
    properties
        multithreadingTraining;
        multithreadingClassification;
        baseClassifiers;
        trainingInstanceProportions;
        numClassifiersStrings;
        trainingInstanceProportionsStrings;
        baseClassifiersStrings;
        baseClassifiersShortStrings;
        remainClassDistribution;
        votingMode;
        validationSubsampleSize;
        weights;
    end
    
    methods
        
        function obj = BasicEnsemble(baseClassifiers, numClassifiers, ...
                trainingInstanceProportions, votingMode, ...
                validationSubsampleSize, ...
                remainClassDistribution, ...
                multithreadingTraining, multithreadingClassification)
            obj.votingMode = votingMode;
            obj.validationSubsampleSize = validationSubsampleSize;
            obj.remainClassDistribution = remainClassDistribution;
            obj.multithreadingTraining = multithreadingTraining;
            obj.multithreadingClassification = ...
                multithreadingClassification;
            
            obj.numClassifiersStrings = ...
                cellfun(@(n)num2str(n), ...
                        num2cell(numClassifiers), ...
                        'uniformoutput', 0);
            obj.trainingInstanceProportionsStrings = ...
                cellfun(@(n)num2str(n), ...
                        num2cell(trainingInstanceProportions), ...
                        'uniformoutput', 0);
            obj.baseClassifiersStrings = cellfun(@(o)o.toString(), ...
                                                 baseClassifiers, ...
                                                 'uniformoutput', 0);
            obj.baseClassifiersShortStrings = ...
                cellfun(@(o)o.toShortString(), ...
                        baseClassifiers, ...
                        'uniformoutput', 0);
            
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
        
        
        function str = toString(obj)
            paramStr = ['maintainClassDistribution: ' ...
                         num2str(obj.remainClassDistribution) ...
                         'voting: ' ...
                         char(obj.votingMode) ...
                         'validationSubsampleSize: ' ...
                         num2str(obj.validationSubsampleSize)];
            str = ['BasicEnsemble ', paramStr, '[' ];
            stringParts = ...
                cellfun(@(n,p,c) ['(num:' n ' proportion:' p ...
                                  ' classifier:[' c '], )'], ...
                        obj.numClassifiersStrings, ...
                        obj.trainingInstanceProportionsStrings,...
                        obj.baseClassifiersStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-2:end) = [];
            str = [str horzcat(stringParts{:}) ']'];
        end
        
        function str = toShortString(obj)
            
            paramStr = ['mcd_' ...
                         num2str(obj.remainClassDistribution) ...
                         '_' ...
                         char(obj.votingMode) ...
                         '_vsSize_' ...
                         num2str(obj.validationSubsampleSize) ...
                         '--'];
            
            str = ['BasicEnsemble--' paramStr];
            stringParts = ...
                cellfun(@(n,p,c) [n '_' p '_' c '--'], ...
                        obj.numClassifiersStrings, ...
                        obj.trainingInstanceProportionsStrings,...
                        obj.baseClassifiersShortStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-1:end) = [];
            str = [str horzcat(stringParts{:})];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)

            % make features 2D
            reshapedTrainFeatures = validListFromSpatial(...
                trainFeatureCube, trainLabelMap, false);
            reshapedTrainLabels = validListFromSpatial(...
                trainLabelMap, trainLabelMap, false);
            
            if obj.votingMode ~= VotingMode.Majority
               validationSubsetIndices = subSampleData( ...
                                           reshapedTrainLabels, ...
                                           obj.validationSubsampleSize, ...
                                           true); 
               validationSubset = reshapedTrainFeatures(...
                                               validationSubsetIndices,...
                                               :);
               validationSubset = permute(validationSubset, [1,3,2]);
               validationLabel = reshapedTrainLabels( ...
                                               validationSubsetIndices);
               reshapedTrainFeatures(validationSubsetIndices,:) = [];
               reshapedTrainLabels(validationSubsetIndices) = [];
            end
            if ~obj.multithreadingTraining

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

            else
                
                classifier = obj.baseClassifiers;
                proportions = obj.trainingInstanceProportions;
                
                parfor i= 1:length(obj.baseClassifiers)
                    classifier{i} = trainBaseClassifier( classifier{i}, ...
                                     reshapedTrainFeatures , ...
                                     reshapedTrainLabels, ...
                                     proportions{i}, ...
                                     obj.remainClassDistribution);
                end
                
                obj.baseClassifiers = classifier;
            end
            if obj.votingMode ~= VotingMode.Majority
                
                obj.weights = ones(size(obj.baseClassifiers));
                
                backupClassifier = cellfun(@copy, ...
                                                obj.baseClassifiers, ...
                                                'UniformOutput', 0)';
                concatenatedLabels = obj.classifyOnEachClassifier(...
                                                 validationSubset, ...
                                                 validationLabel);
                obj.baseClassifiers = backupClassifier;
                newWeights = zeros(size(obj.baseClassifiers));
                
                
                parfor i = 1:numel(obj.baseClassifiers)
                    predictedLabels = concatenatedLabels(:,i);
                    availableLabels = unique([predictedLabels;validationLabel]);
                    cMat = confusionmat(...
                                     validationLabel, predictedLabels, ...
                                     'order',availableLabels);
                    newWeights(i) = ...
                        Evaluator.getAccuracy(cMat(2:end, 2:end));
                end
                
                if obj.votingMode == VotingMode.Majority
                    [~, bestClassifier] = max(newWeights);
                    obj.weights = ones(size(obj.weights));
                    obj.weights(bestClassifier) = numel(newWeights) - 0.5;
                else
                    obj.weights = normalizeRating( newWeights, [0.1,0.9] );
                end
            elseif obj.votingMode == VotingMode.Majority
                obj.weights = ones(size(obj.baseClassifiers));
            end
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap)
            
            % let each classifier predict
            concatenatedLabels = ...
                obj.classifyOnEachClassifier(evalFeatureCube, maskMap);
            [x,y,nClassifier] = size(concatenatedLabels);
            concatenatedLabels = ...
                reshape(concatenatedLabels, x*y, nClassifier);
            
            predictedLabelMap = zeros(x*y, 1);
            
            parfor i = 1:numel(maskMap)
                w = obj.weights;
                predictions = concatenatedLabels(i,:);
                labels = unique(predictions);
                labelWeights = zeros(size(labels));
                for labelIdx = 1:numel(labels)
                    l = labels(labelIdx);
                    labelWeights(labelIdx) = sum(w(predictions == l));
                end
                [~,labelIndex] = max(labelWeights);
                predictedLabelMap(i) = labels(labelIndex);
            end
            predictedLabelMap = reshape(predictedLabelMap, x, y);
        end
        
        function concatenatedLabels = ...
                    classifyOnEachClassifier(obj, evalFeatureCube, maskMap)
            
            
            if obj.multithreadingClassification
                classifier = obj.baseClassifiers;
                accumulatedLabels = cell(length(classifier),1);
                parfor i = 1:length(classifier)
                    accumulatedLabels{i} = ...
                        classifier{i}.classifyOn(...
                            evalFeatureCube, maskMap);
                end
            else
                % Create classify function call handle
                classifyHandle = @(classifier) classifier.classifyOn(...
                evalFeatureCube, maskMap);
                % Classify all instances
                accumulatedLabels = cellfun(...
                    classifyHandle, obj.baseClassifiers, ...
                    'UniformOutput', 0)';
            end
            
            % concatenate the labels along the third dimension
            concatenatedLabels = cat(3, accumulatedLabels{:});
            
        end
    end
    
end

