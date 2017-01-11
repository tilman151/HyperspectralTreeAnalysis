classdef Logger
    %Logger Receives the data from experiment and saves it to a file
    
    properties
       
        logger;
        filePath;
        logging;
        
    end
    
    properties (Access = private)
       
        errorMessage;
        
    end
    
    methods (Static)
       
        function obj = getLogger()
            persistent localLogger;
            if isempty(localLogger)
                localLogger = Logger();
            end
            obj = localLogger;
        end
        
    end
    
    methods (Access = private)
       
        function obj = Logger()
            obj.filePath = ['./experiment_', ...
                            datestr(now, 'yyyymmdd_HHMM'), ...
                            '_log.txt'];
            obj.logger = log4m(obj.filePath);
            obj.logging = false;
            obj.errorMessage = ...
                'Cannot log, when experiment is not running';
            
            fid = fopen(obj.filePath, 'w');
            fprintf(fid, 'HyperSpectralTreeExperiment\n');
            fprintf(fid, '--------------------------------------------\n');
            fclose(fid);
        end
        
    end
    
    methods
        
        function obj = startExperiment(obj)
            fid = fopen(obj.filePath, 'a');
            fprintf(fid, 'Started: %s\n', datestr(now));
            fclose(fid);
            obj.logging = true;
        end
        
        function obj = stopExperiment(obj)
            fid = fopen(obj.filePath, 'a');
            fprintf(fid, 'Stopped: %s\n', datestr(now));
            fprintf(fid, '--------------------------------------------\n');
            fclose(fid);
            obj.logging = false;
        end
        
        function obj = logConfig(obj, ...
                                 classifier, ...
                                 extractors, ...
                                 samplePath, ...
                                 dataPath, ...
                                 crossValParts)
            fid = fopen(obj.filePath, 'a');
            fprintf(fid, 'Classifier:\t%s\n', class(classifier));
            extractorList = cellfun(@class, extractors, ...
                                    'UniformOutput', false);
            fprintf(fid, 'Extractors:\t%s\n', strjoin(extractorList, ', '));
            fprintf(fid, 'Sample Set:\t%s\n', samplePath);
            fprintf(fid, 'Data Set:\t%s\n', dataPath);
            fprintf(fid, 'CrossValParts:\n');
            fclose(fid);
            
            dlmwrite(obj.filePath, ...
                         crossValParts, 'Delimiter', '\t', '-append');
                     
            fid = fopen(obj.filePath, 'a');
            fprintf(fid, '--------------------------------------------\n');
            fclose(fid);
        end
        
        function obj = logConfusionMatrix(obj, confMat)
            fid = fopen(obj.filePath, 'a');
            fprintf(fid, 'Confusion Matrix:\n');
            fclose(fid);
            dlmwrite(obj.filePath, ...
                         confMat, 'Delimiter', '\t', '-append');
        end
        
        function trace(obj, funcName, message)
            if obj.logging
                obj.logger.trace(funcName, message);
            else
                error(obj.errorStruct);
            end
        end
        
        function debug(obj, funcName, message)
            if obj.logging
                obj.logger.debug(funcName, message);
            else
                error(obj.errorStruct);
            end
        end
        
 
        function info(obj, funcName, message)
            if obj.logging
                obj.logger.info(funcName, message);
            else
                error(obj.errorMessage);
            end
        end
        

        function warn(obj, funcName, message)
            if obj.logging
                obj.logger.warn(funcName, message);
            else
                error(obj.errorMessage);
            end
        end
        

        function error(obj, funcName, message)
            if obj.logging
                obj.logger.error(funcName, message);
            else
                error(obj.errorMessage);
            end
        end
        

        function fatal(obj, funcName, message)
            if obj.logging
                obj.logger.fatal(funcName, message);
            else
                error(obj.errorMessage);
            end
        end
        
    end
    
end

