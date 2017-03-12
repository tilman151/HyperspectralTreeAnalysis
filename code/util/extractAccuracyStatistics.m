function durations = extractAccuracyStatistics(resultsPath)
%EXTRACTACCURACYSTATISTICS Extracts statistics from experiment log files.
%
%   This function decends recursively into the resultsPath and extracts
%   statistics about the accuracy of an experiment from a log file.
%
%   The statistics are written to a table and statistics of incomplete 
%   experiments are deleted.
%
%   The table is saved to a CSV file in the resultsPath.
%
%% Input:
%   resultsPath . Root path of the results dir
%
%% Output:
%   durations ... Table containing statistics of the experiments
%
%%

    % Begin descent
    durations = descendToFile(resultsPath, numel(resultsPath));
    % Delete incomplete experiments
    toDelete = isnan(durations{:, 2});
    durations = durations(~toDelete, :);
    
    % Write to CSV
    writetable(durations, ...
               fullfile(resultsPath, 'experimentAccuracyStats.csv'));

end

function statTable = descendToFile(resultsPath, prefixLength)
%DECENDTOFILE Recursively descend to the log file and accumulate durations
%   in table.
%%

    % Look for log file in current dir
    files = dir(fullfile(resultsPath, 'experiment*.txt'));
    
    % If log file in current dir
    if ~isempty(files)
        % Extract stats
        stats = processLog(fullfile(resultsPath, files(1).name));
        statTable = [{resultsPath(prefixLength+1:end)}, ...
                     stats];
    else
        % List sub dirs of current dir
        subDirs = dir(resultsPath);
        subDirs = subDirs(3:end);
        subDirs = subDirs([subDirs.isdir]);
        
        % Init table
        statTable = table({}, [], [], ...
                          'VariableNames', {'Experiment', 'Mean', 'STD'});
        % For each sub dir...
        for i = 1:numel(subDirs)
            % Accumulate statistics
            statTable = [statTable; ...
                         descendToFile(fullfile(resultsPath, ...
                                               subDirs(i).name), ...
                                      prefixLength) ...
                        ];
        end
    end
end

function stats = processLog(logFilePath)
%PROCESSLOG Extracts the duration of the experiment from the log file.
%
%%

    % Open file
    fid = fopen(logFilePath, 'r');
    
    % Extract lines
    fileLines = textscan(fid, '%s', 'Delimiter', '\n');
    fileLines = fileLines{1};
    
    % Close file
    fclose(fid);
    
    % Get idx of stop line
    stopIdx = ~(cellfun('isempty', strfind(fileLines, 'Stopped: ')));
    
    % If experiment is incomplete...
    if ~any(stopIdx)
        % Return -1
        stats = {NaN, NaN};
        return;
    end
    
    % Look for accuracy logs
    accuraciesIdx = ~(cellfun('isempty', ...
                              strfind(fileLines, 'Current accuracy: ')));
    % Extract accuracies
    accuracies = cellfun(@(x)str2double(x(end-5:end)), ...
                         fileLines(accuraciesIdx));
    
    % Compute and return statistics
    stats = {mean(accuracies), std(accuracies)};

end