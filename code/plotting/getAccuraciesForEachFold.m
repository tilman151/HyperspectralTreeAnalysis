function [ reults ] = getAccuraciesForEachFold( paths )
%GETACCURACIESFOREACHFOLD Summary of this function goes here
%   Detailed explanation goes here

numFolds = 10;

reults = zeros(numFolds, numel(paths));

for pIdx = 1:numel(paths)
    directoryPath = paths{pIdx};
    logFiles = dir([directoryPath '/*_log*']);
    logPath = [directoryPath '/' logFiles.name];
    text = fileread(logPath);
    accuracyStrings = regexp(text, '(?<=(accuracy: ))\d\.\d+', 'match');
    accuracyValues = cellfun(@str2num, accuracyStrings);
    reults(:, pIdx) = accuracyValues;

end

end