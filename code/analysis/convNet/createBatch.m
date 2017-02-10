function batch = createBatch(indices, batchIndex, batchSize, ...
    trainFeatureCube, trainLabelMap)
    batch.features = zeros(21, 21, size(trainFeatureCube, 3), 0);
    batch.labels = [];

    batchIndices = indices(:, (batchIndex - 1) * batchSize + 1 : ...
        min(batchIndex * batchSize, length(indices)));

    for index = batchIndices
        try
            labels = trainLabelMap(index(1) - 10 : index(1) + 10, ...
                index(2) - 10 : index(2) + 10);
            assert(all(all(labels >= 0)));
        catch e
            % index out of bounds or filling pixels
            continue;
        end
        batch.features(:, :, :, end + 1) = ...
            trainFeatureCube(index(1) - 10 : index(1) + 10, ...
                index(2) - 10 : index(2) + 10, :);
        batch.labels(end + 1) = trainLabelMap(index(1), index(2));
    end        
end