function featureCube = normalizeData(featureCube, dimMeans, dimStds)
%NORMALIZEDATA Normalize the data to mean 0 and variance 1
%   Detailed explanation goes here
    
    [x, y, ~] = size(featureCube);
    
    % Extend mean and std to size of feature cube
    dimMeans = repmat(reshape(dimMeans, 1, 1, []), [x, y]);
    dimStds = repmat(reshape(dimStds, 1, 1, []), [x, y]);
    
    % Normalize feature cube
    featureCube = (featureCube - dimMeans) ./ dimStds;
end

