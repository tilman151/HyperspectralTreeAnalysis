function runUnknownDataExperiment(configFilePath)
    %RUNUNKNOWNDATAEXPERIMENT Run a machine learning experiment
    %              with the parameters defined in the configuration file
    %
    %   The function takes a path to a configuration file and reads it. The
    %   statements in the file are executed to initialize the needed
    %   variables for the experiment.
    %   
    %   This function is intended to apply a trained classifier to new
    %   unknown data.
    %
    %   The configuration, all actions of this function and the confusion
    %   matrix are logged.
    %
    %%  Input:
    %       configFilePath ..... a path to the config file file
    %
    %
    % Version: 2017-03-03
    % Author: Tilman Krokotsch
    %%
    
    % Clear all functions from memory to avoid side effects 
    % (like persistent logger object)
    clear functions;
    
    % Init random seed
    rng('shuffle');
    
    % Add the code directory and all subdirectories to path
    addpath(genpath('./'));

    % Read and execute config file
    if nargin < 1
        configFilePath = './configUnknownData.m';
    end
    run(configFilePath);
    
    % Create logger singleton
    logPath = Logger.createLogPath(RESULTS_PATH, CLASSIFIER, EXTRACTORS);
    logger = Logger.getLogger(logPath);
    % Log configuration
    logger.logConfig(CLASSIFIER, ...
                     EXTRACTORS, ...
                     SAMPLE_SET_PATH, ...
                     DATA_SET_PATH, ...
                     []);
    
    % Set log level
    logger.setLogLevel(LOG_LEVEL);
    
    % Start logging
    logger = logger.startExperiment();
    
    % Fetch file names
    featureFiles = dir(fullfile(DATA_SET_PATH, '*.mat'));
    
    % Create result data structure
    predictedLabels = cell(numel(featureFiles), 1);
     
    for i=1:numel(featureFiles)
        % Load test set
        logger.debug('runUnknownDataExperiment', 'Loading data set...');
        load(fullfile(DATA_SET_PATH, featureFiles(i).name));
        
        % Create mask map (everything is unknown)
        maskMap = zeros(size(cube, 1), size(cube, 2));

        % Apply feature extraction
        logger.debug('runUnknownDataExperiment', ...
                     'Applying feature extraction...');
        cube = applyFeatureExtraction(cube, EXTRACTORS, ...
                                      maskMap, SAMPLE_SET_PATH);

        % Apply trained classifier
        logger.debug('runUnknownDataExperiment', 'Applying classifier...');
        predictedFileLabels = CLASSIFIER.classifyOn(cube, maskMap);
        logger.info('runUnknownDataExperiment', 'Instances classified');

        % Visualize predicted labels
        if VISUALIZE_PREDICTED_LABELS
            visualizeLabels(predictedFileLabels, 'Predicted Labels');
        end
        
        % Store result
        predictedLabels{i} = predictedFileLabels;

        % Free RAM
        clear('predictedFileLabels', 'cube');
    end
    
    % Save results
    logger.info('runUnknownDataExperiment', 'Save results');
    save(fullfile(logger.getLogPath(), 'results.mat'), 'predictedLabels');
    
    % Stop logging
    logger.stopExperiment();
end

function featureCube = applyFeatureExtraction(featureCube, extractors, ...
                                              maskMap, sampleSetPath)
    for i = 1:numel(extractors)
        featureCube = extractors{i}.extractFeatures(featureCube, ...
            maskMap, sampleSetPath);
    end
end
