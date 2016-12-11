function mergeData(path)
%MERGEDATA Merges the label and feature files in a path to two big file
%
%   ONLY TO BE USED AFTER EXECUTION OF PROCESSMATS AND SPLITONEANDFOUR
%
%   The function takes all files in path and filters for the label and
%   feature files. 
%
%   The label data is stored to the labels.mat file, where each variable
%   represents the file it is named after.
%   
%   The feature data is stored to the features.mat file, where each
%   variable represents the file it is named after.
%
%%  Input:
%       path ... a path to the files
%
% Version: 2016-12-11
% Author: Tilman Krokotsch
%%

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

    % Create labels.mat
    date = datestr(now);
    save([path, 'labels.mat'], 'date');
    
    % Load label files and save as variables to labels.mat
    for i = 1:size(labelFiles, 1)
        load([path, labelFiles{i}]);
        varName = labelFiles{i}(1:7);
        command = [varName, ' = labels;'];
        eval(command);
        save([path, 'labels.mat'], varName, '-append');
        clear(varName);
    end
    
    % Create features.mat
    save([path, 'features.mat'], 'date');
    
    % Load feature files and save as variables to features.mat
    for i = 1:size(featureFiles, 1)
        load([path, featureFiles{i}]);
        varName = featureFiles{i}(1:7);
        command = [varName, ' = cube;'];
        eval(command);
        save([path, 'features.mat'], varName, '-append');
        clear(varName);
    end
    
end

