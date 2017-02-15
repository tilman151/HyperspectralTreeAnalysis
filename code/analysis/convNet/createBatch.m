function batch = createBatch(indices, batchIndex, batchSize, sampleSize,...
    trainFeatureCube, trainLabelMap)
    
    batch.features = zeros(...
        sampleSize, sampleSize, size(trainFeatureCube, 3), 0, 'single');
    batch.labels = [];

    batchIndices = indices(:, (batchIndex - 1) * batchSize + 1 : ...
        min(batchIndex * batchSize, length(indices)));
    
    % Get "radius" for samples. sampleSize should be an odd number
    r = (sampleSize - 1) / 2;
    
    for index = batchIndices
        try
            labels = trainLabelMap(index(1) - r : index(1) + r, ...
                index(2) - r : index(2) + r);
            assert(all(all(labels >= 0)));
        catch e
            % index out of bounds or filling pixels
            continue;
        end
        batch.features(:, :, :, end + 1) = ...
            trainFeatureCube(index(1) - r : index(1) + r, ...
                index(2) - r : index(2) + r, :);
        batch.labels(end + 1) = trainLabelMap(index(1), index(2));
    end        
end