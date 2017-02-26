function models = train1vs1Svmlin(featureList, labelList, useUnlabeled)
%TRAIN1VS1SVMLIN Summary of this function goes here
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
    
    % Create class combinations and cell array for binary classifiers
    classpairs = nchoosek(1:numClasses, 2);
    numBinaryClassifiers = size(classpairs, 1);
    models = cell(numBinaryClassifiers, 3);
    
    % Create parameters (-A 1 for SVM, -A 2 for TSVM)
    if useUnlabeled
        params = '-A 2';
    else
        params = '-A 1';
    end
    
    % Train binary classifiers
    for ii = 1 : numBinaryClassifiers
        % Get classes for this classifier
        classpair = classpairs(ii, :);
        c1 = classes(classpair(1));
        c2 = classes(classpair(2));
        
        logger.trace('SVM 1vs1', ...
            ['Training ' num2str(c1) ' vs. ' num2str(c2)]);
        
        % Concatenate features and labels for the two classes
        binaryFeatureList = [...
            featureList(labelList == c1, :); ...
            featureList(labelList == c2, :)];
        binaryLabelList = [...
            ones(sum(labelList == c1), 1); ...
            -ones(sum(labelList == c2), 1)];
        
        if useUnlabeled
            % Append neutral data
            binaryFeatureList = [binaryFeatureList; neutralFeatureList];
            binaryLabelList = [binaryLabelList; neutralLabelList];
        end
        
        % Make feature list sparse
        binaryFeatureList = sparse(binaryFeatureList);
        
        % Train and store model
        modelWeights = svmlin(params, binaryFeatureList, binaryLabelList);
        models(ii, :) = {c1, c2, modelWeights};
    end
    
end

