classdef BestClassifierEnsemble < Classifier
    %BestClassifierEnsemble Classifier that uses other classifiers to predict labels
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
    %           votingMode .................... VotingMode enum to specify
    %                                           the voting method to get to 
    %                                           a conclusion
    %           loadMatchingClassifier ........ set true if the ensemble
    %                                           should use the i-th 
    %                                           classifier for the i-th
    %                                           fold
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
    %           multithreadingClassification .. set to true if
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
    % Version: 2017-14-02
    % Author: Tuan Pham Minh
    %
    
    properties
        paths;
        loadMatchingClassifier;
        samplesetPath;
        multithreadingClassification;
        featureExtractors;
        baseClassifiers;
        numClassifiersStrings;
        baseClassifiersStrings;
        baseClassifiersShortStrings;
        votingMode;
        weights;
    end
    
    methods
        
        function obj = BestClassifierEnsemble(baseClassifierPaths, ...
                votingMode, loadMatchingClassifier, ...
                multithreadingClassification, samplesetPath)
            obj.paths = baseClassifierPaths;
            
            obj.loadMatchingClassifier = loadMatchingClassifier;
            
            obj.samplesetPath = samplesetPath;
            
            if ~obj.loadMatchingClassifier
                [baseClassifiers, featureExtractors, ...
                    accuracies, precisions] = ...
                    obj.loadClassifier(baseClassifierPaths); 
            else
                [baseClassifiers, featureExtractors, ...
                    accuracies, precisions] = ...
                    obj.loadMatchingFoldClassifier(baseClassifierPaths, 1); 
            end
            
            
            obj.votingMode = votingMode;
            obj.multithreadingClassification = ...
                multithreadingClassification;
            
            obj.baseClassifiersStrings = cellfun(@(o)o.toString(), ...
                                                 baseClassifiers, ...
                                                 'uniformoutput', 0);
            obj.baseClassifiersShortStrings = ...
                cellfun(@(o)o.toShortString(), ...
                        baseClassifiers, ...
                        'uniformoutput', 0);
            
            % copy the classifier
            obj.featureExtractors = featureExtractors;
            obj.baseClassifiers = baseClassifiers;
            
            if ~obj.loadMatchingClassifier
                if obj.votingMode ~= VotingMode.PrecisionWeighting
                    obj.weights = precisions;
                elseif obj.votingMode == VotingMode.Majority
                    obj.weights = ones(numel(accuracies), 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                elseif obj.votingMode == VotingMode.Presidential
                    [~, bestClassifier] = max(accuracies);
                    obj.weights = ones(1, numel(accuracies));
                    obj.weights(bestClassifier) = numel(accuracies) - 0.5;
                    obj.weights = reshape(obj.weights, [], 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                else
                    obj.weights = normalizeRating( accuracies, [0.1,0.9] );
                    obj.weights = reshape(obj.weights, [], 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                end
            end
        end
        
        
        function str = toString(obj)
            paramStr = ['voting: ' ...
                         char(obj.votingMode)];
            str = ['BestEnsemble ', paramStr, '[' ];
            stringParts = ...
                cellfun(@(c) ['( classifier:[' c '] )'], ...
                        obj.baseClassifiersStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-2:end) = [];
            str = [str horzcat(stringParts{:}) ']'];
        end
        
        function str = toShortString(obj)
            
            paramStr = char(obj.votingMode);
            
            str = ['BestEnsemble --' paramStr];
            stringParts = ...
                cellfun(@(c) [c '--'], ...
                        obj.baseClassifiersShortStrings, ...
                        'uniformoutput', 0);
            stringParts{end}(end-1:end) = [];
            str = [str horzcat(stringParts{:})];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
        end
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, foldNr)
            
            
            if obj.loadMatchingClassifier
                [baseClassifiers, featureExtractors, ...
                    accuracies, precisions] = ...
                    obj.loadMatchingFoldClassifier(obj.paths, foldNr); 
            
                obj.featureExtractors = featureExtractors;
                obj.baseClassifiers = baseClassifiers;
                
                if obj.votingMode ~= VotingMode.PrecisionWeighting
                    obj.weights = precisions;
                elseif obj.votingMode == VotingMode.Majority
                    obj.weights = ones(numel(accuracies), 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                elseif obj.votingMode == VotingMode.Presidential
                    [~, bestClassifier] = max(accuracies);
                    obj.weights = ones(1, numel(accuracies));
                    obj.weights(bestClassifier) = numel(accuracies) - 0.5;
                    obj.weights = reshape(obj.weights, [], 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                else
                    obj.weights = normalizeRating( accuracies, [0.1,0.9] );
                    obj.weights = reshape(obj.weights, [], 1);
                    obj.weights = repmat(obj.weights, 1, numel(precisions));
                end
            end
            
            disp('start predicting');
            % let each classifier predict
            concatenatedLabels = ...
                obj.classifyOnEachClassifier(evalFeatureCube, maskMap);
            [x,y,nClassifier] = size(concatenatedLabels);
            concatenatedLabels = ...
                reshape(concatenatedLabels, x*y, nClassifier);
            disp('end predicting');
            
            predictedLabelMap = zeros(x*y, 1);
            
            weights = obj.weights;
            
            for i = 1:numel(maskMap)
                if maskMap(i) >= 0
                    w = weights;
                    predictions = concatenatedLabels(i,:);
                    labels = unique(predictions);
                    labelWeights = zeros(size(predictions));
                    for labelIdx = 1:numel(labels)
                        l = labels(labelIdx);
                        wLabel = w(:,l);
                        labelWeights(labelIdx) = ...
                            sum(wLabel(predictions == l));
                    end
                    [~,labelIndex] = max(labelWeights);
                    predictedLabelMap(i) = labels(labelIndex);
                end
            end
            predictedLabelMap = reshape(predictedLabelMap, x, y);
        end
        
        function concatenatedLabels = ...
                    classifyOnEachClassifier(obj, evalFeatureCube, maskMap)
                    
            if obj.multithreadingClassification
                classifier = obj.baseClassifiers;
                extractors = obj.featureExtractors;
                accumulatedLabels = cell(length(classifier),1);
                parfor i = 1:length(classifier)
                    newFeatureCube = applyFeatureExtraction(...
                        evalFeatureCube, ...
                        maskMap, ...
                        extractors, ...
                        obj.samplesetPath);
                    accumulatedLabels{i} = ...
                        classifier{i}.classifyOn(...
                            newFeatureCube, maskMap);
                end
            else
                classifier = obj.baseClassifiers;
                accumulatedLabels = cell(length(classifier),1);
                for i = 1:length(classifier)
                    i
                    'start feature extraction'
                    newFeatureCube = applyFeatureExtraction(...
                        evalFeatureCube, ...
                        maskMap, ...
                        obj.featureExtractors{i}, ...
                        obj.samplesetPath);
                    
                    'end feature extraction'
                    
                    accumulatedLabels{i} = ...
                        classifier{i}.classifyOn(...
                            newFeatureCube, maskMap);
                end
            end
            
            % concatenate the labels along the third dimension
            concatenatedLabels = cat(3, accumulatedLabels{:});
            
        end
        
        function [bC, fE, w, pr] = loadClassifier(obj, baseClassifierPaths)
           bC = cell(size(baseClassifierPaths));
           fE = cell(size(baseClassifierPaths));
           w = zeros(size(baseClassifierPaths));
           pr = [];
            for i = 1:numel(baseClassifierPaths)
               p = baseClassifierPaths{i};
               [d,f,e] = fileparts(p);
               
               if(isempty(f))
                   searchStr = [d '/model_*'];
                   modelFilePaths = dir(searchStr);
                   modelFilePaths = {modelFilePaths.name};
                   accuracies = zeros(size(modelFilePaths));
                   for j = 1:numel(modelFilePaths)
                      modelFile = ...
                          load([d '/' modelFilePaths{j}], 'confMat');
                      cMat = modelFile.confMat;
                      accuracies(j) =  ...
                          Evaluator.getAccuracy(cMat);
                   end
                   [~, bestClassifierIndex] = max(accuracies);
                   f = modelFilePaths{bestClassifierIndex};
               else
                   f = [f e];
                   e = '';
               end
               
               modelFile = load([d '/' f]);
               classifier = modelFile.model;
               cMat = modelFile.confMat;
               accuracy = Evaluator.getAccuracy(cMat);
               precision = Evaluator.getPrecisions(cMat);
               
               featureExtractorFile = load([d '/FeatureExtractors.mat']);
               featureExtractors = featureExtractorFile.EXTRACTORS;
               
               bC{i} = classifier;
               fE{i} = featureExtractors;
               w(i) = accuracy;
               if isempty(pr)
                   pr = zeros(numel(baseClassifierPaths), numel(precision));
               end
               pr(i,:) = precision;
            end
        end
        
        function [bC, fE, w, pr] = ...
                loadMatchingFoldClassifier(obj, baseClassifierPaths, fold)
            paths = baseClassifierPaths;
            for i = 1:numel(baseClassifierPaths)
                paths{i} = [paths{i} 'model_' num2str(fold)];
            end
            
            disp('load :');
            disp(paths)
            [bC,fE,w, pr] = loadClassifier(obj,paths);
            disp('finished loading');
        end
    end
    
end


        
function rawFeatures = applyFeatureExtraction(rawFeatures, maskMap, ...
                                extractors, samplesetPath)
    for i = 1:numel(extractors)
        rawFeatures = extractors{i}.extractFeatures(... 
                            rawFeatures, ...
                            maskMap, samplesetPath);
    end 
end
