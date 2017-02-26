function predictedLabelList = predict1vs1Svmlin(featureList, models)
%PREDICT1VS1SVMLIN Summary of this function goes here
%   Detailed explanation goes here
    
    % Obtain vote from each model
    votes = cellfun(@(c1, c2, m) applyModel(c1, c2, m, featureList), ...
        models(:, 1), models(:, 2), models(:, 3), 'UniformOutput', false);
    
    % Reshape votes to numSamples x numModels
    votes = cell2mat(votes);
    votes = reshape(votes, [size(featureList, 1), size(models, 1)]);
    
    % Decide for class with maximum number of votes
    maxClass = max(votes(:));
    voteCounts = histc(votes, 1:maxClass, 2);
    [~, predictedLabelList] = max(voteCounts, [], 2);
    
end

function predictedLabelList = applyModel(c1, c2, model, featureList)
    % Make feature list sparse
    featureList = sparse(featureList);
    
    % Predict labels
    [~, predictedLabelList] = svmlin([], featureList, [], model);
    
    % Assign classes based on predictions
    predictedLabelList(predictedLabelList > 0) = c1;
    predictedLabelList(predictedLabelList <= 0) = c2;
end
