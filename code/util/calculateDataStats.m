function [dimMeans, dimStds] = calculateDataStats(featureCube, labelMap)
%CALCULATEDATASTATS Calculate mean and variance of the data
    
    % Initialize mean and std
    numFeatureDims = size(featureCube, 3);
    dimMeans = zeros(1, numFeatureDims);
    dimStds = zeros(1, numFeatureDims);
    
    % Exclude fill pixels from calculation
    valid = find(labelMap >= 0);
    
    % Calculate mean and std for each feature dimension
    for dim = 1:numFeatureDims
        slice = featureCube(:, :, dim);
        
        % Save mean and std
        dimMeans(dim) = mean(slice(valid));
        dimStds(dim) = std(slice(valid));
    end
end

