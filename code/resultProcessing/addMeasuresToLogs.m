function addMeasuresToLogs(logPath)
%ADDMEASURESTOLOGS Adds the accuracy measures to logs in a path

    if (logPath(end) ~= '/' && isunix)
        logPath = [logPath, '/'];
    elseif (logPath(end) ~= '\' && ispc)
        logPath = [logPath, '\'];
    end

    content = dir(logPath);
    content = content(3:end);
    
    directories = content([content.isdir]);
    for i = 1:numel(directories)
        addMeasuresToLogs([logPath, directories(i).name, '/']);
    end
    
    csvFiles = dir([logPath, '*csv']);
    logFiles = dir([logPath, '*_log.txt']);
    
    for i = 1:numel(csvFiles)
        confMat = dlmread([logPath, csvFiles(i).name]);
        measures = Evaluator.getAllMeasures(confMat);
        
        fields = fieldnames(measures);
        fid = fopen([logPath, logFiles(i).name], 'a');

        fprintf(fid, ...
                '-----------------------------------------------------\n');
        
        for num = 1:numel(fields)
            fprintf(fid, ['\n',fields{num} , ':\n']);
            dlmwrite([logPath, logFiles(i).name], ...
                     (measures.(fields{num}))', ...
                     'Delimiter', '\t', ...
                     'precision', '%.3f', ...
                     '-append');
       end

       fclose(fid);
    end

end

