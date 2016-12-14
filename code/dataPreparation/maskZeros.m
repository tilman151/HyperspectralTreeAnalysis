function maskZeros(path)
%MASKZEROS labels all feature vectors containing only zeros with -1

    % Get all files in path
    files = dir(path);
    files = {files.name};
    
    % Filter for label files
    labelFiles = regexp(files, '\w*_labels_new\w*', 'match');
    labelFiles = cellfun('isempty', labelFiles);
    labelFiles = files(~labelFiles)';
    
    % Filter for feature files
    featureFiles = regexp(files, '[0-9]_new\w*', 'match');
    featureFiles = cellfun('isempty', featureFiles);
    featureFiles = files(~featureFiles)';
    
    for i = 1:size(labelFiles, 1)
        labelFile = load([path, labelFiles{i}]);
        featureFile = load([path, featureFiles{i}]);
        labels = labelFile.labels;
        cube = featureFile.cube;
        
        whereZero = all(cube == 0, 3);
        labels(whereZero) = -1;
        
        save([path, labelFiles{i}], 'labels');
        save([path, featureFiles{i}], 'cube');
        
    end

end

