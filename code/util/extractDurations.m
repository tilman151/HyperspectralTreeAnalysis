function durations = extractDurations(resultsPath)
%EXTRACTDURATIONS Extracts the durations from experiment log files.
%
%   This function decends recursively into the resultsPath and extracts the
%   duration of an experiment from a log file.
%
%   The durations are written to a table and durations of incomplete 
%   experiments are deleted.
%
%   The table is saved to a CSV file in the resultsPath.
%
%% Input:
%   resultsPath . Root path of the results dir
%
%% Output:
%   durations ... Table containing durations of the experiments
%
%%

    % Begin descent
    durations = descendToFile(resultsPath, numel(resultsPath));
    % Delete incomplete experiments
    toDelete = durations{:, 2} == duration(-24, 0, 0);
    durations = durations(~toDelete, :);
    
    % Write to CSV
    writetable(durations, ...
               fullfile(resultsPath, 'experimentDurations.csv'));

end

function durations = descendToFile(resultsPath, prefixLength)
%DECENDTOFILE Recursively descend to the log file and accumulate durations
%   in table.
%%

    % Look for log file in current dir
    files = dir(fullfile(resultsPath, 'experiment*.txt'));
    
    % If log file in current dir
    if ~isempty(files)
        % Extract duration
        durations = {resultsPath(prefixLength+1:end), ...
                     processLog(fullfile(resultsPath, files(1).name))};
    else
        % List sub dirs of current dir
        subDirs = dir(resultsPath);
        subDirs = subDirs(3:end);
        subDirs = subDirs([subDirs.isdir]);
        
        % Init table
        durations = table({}, [], ...
                          'VariableNames', {'Experiment', 'Duration'});
        % For each sub dir...
        for i = 1:numel(subDirs)
            % Accumulate durations
            durations = [durations; ...
                         descendToFile(fullfile(resultsPath, ...
                                               subDirs(i).name), ...
                                      prefixLength) ...
                        ];
        end
    end
end

function duration = processLog(logFilePath)
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
    
    % Get idx of start and stop line
    startIdx = ~(cellfun('isempty', strfind(fileLines, 'Started: ')));
    stopIdx = ~(cellfun('isempty', strfind(fileLines, 'Stopped: ')));
    
    % If experiment is incomplete...
    if ~any(stopIdx)
        % Return -1
        duration = -1;
        return;
    end
    
    % Get start date
    start = fileLines{startIdx};
    start = datetime(start(10:end));
    
    % Get stop date
    stop = fileLines{stopIdx};
    stop = datetime(stop(10:end));
    
    % Calculate and return duration
    duration = stop - start;

end