function predictedLabelList = predict1vsAllSvmlin(featureList, models)
%PREDICT1VSALLSVMLIN Summary of this function goes here
%   Detailed explanation goes here
    
    % Obtain vote from each model
    votes = cellfun(@(m) applyModel(m, featureList), ...
        models(:, 2), 'UniformOutput', false);
    
    % Reshape votes to numSamples x numModels
    votes = cell2mat(votes);
    votes = reshape(votes, [size(featureList, 1), size(models, 1)]);
    
    % Decide for class with maximum confidence
    [~, predictionList] = max(votes, [], 2);
    classes = cell2mat(models(:, 1));
    predictedLabelList = classes(predictionList);
end

function predictionList = applyModel(model, featureList)
    % Make feature list sparse
    featureList = sparse(featureList);
    
    % Predict labels
    [~, predictionList] = svmlin([], featureList, [], model);
end
