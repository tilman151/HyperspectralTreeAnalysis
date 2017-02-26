function models = train1vsAllSvmlin(featureList, labelList, useUnlabeled)
%TRAIN1VSALLSVMLIN Summary of this function goes here
%   Detailed explanation goes here
    
    % Check if useUnlabeled is given
    if ~exist('useUnlabeled', 'var')
        useUnlabeled = false;
    end
    
    % Get logger
    logger = Logger.getLogger();
    
    if useUnlabeled
        % Get neutral data
        neutralFeatureList = featureList(labelList == 0, :);
        neutralLabelList = zeros(size(neutralFeatureList, 1), 1);
    end
    
    % Get list of classes
    classes = unique(labelList(labelList > 0));
    numClasses = length(classes);
    
    % Create cell array for binary classifiers
    models = cell(numClasses, 2);
    
    % Create parameters (-A 1 for SVM, -A 2 for TSVM)
    if useUnlabeled
        params = '-A 2';
    else
        params = '-A 1';
    end
    
    % Train binary classifiers
    for ii = 1:numClasses
        % Get class
        c = classes(ii);
        
        logger.trace('SVM 1vsAll', ...
            ['Training ' num2str(c) ' vs. All']);
        
        % Get class indices
        cInds = labelList == c;
        allInds = ~cInds & labelList > 0;
        
        % Concatenate features and labels for c and all others
        binaryFeatureList = ...
            [featureList(cInds, :); featureList(allInds, :)];
        binaryLabelList = [ones(sum(cInds), 1); -ones(sum(allInds), 1)];
        
        if useUnlabeled
            % Append neutral data
            binaryFeatureList = [binaryFeatureList; neutralFeatureList];
            binaryLabelList = [binaryLabelList; neutralLabelList];
        end
        
        % Make feature list sparse
        binaryFeatureList = sparse(binaryFeatureList);
        
        % Train and store model
        modelWeights = svmlin(params, binaryFeatureList, binaryLabelList);
        models(ii, :) = {c, modelWeights};
    end
    
end

