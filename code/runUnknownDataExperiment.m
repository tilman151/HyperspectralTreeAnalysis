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
    
    % Initialize confusion matrix
    confMat = zeros(NUMCLASSES+1, NUMCLASSES+1, crossValidator.k);
    
    % Create logger singleton
    logPath = Logger.createLogPath(RESULTS_PATH, CLASSIFIER, EXTRACTORS);
    logger = Logger.getLogger(logPath);
    % Log configuration
    logger.logConfig(CLASSIFIER, ...
                     EXTRACTORS, ...
                     SAMPLE_SET_PATH, ...
                     DATA_SET_PATH, ...
                     crossValidator.crossValParts);
    
    % Set log level
    logger.setLogLevel(LOG_LEVEL);
    
    % Start logging
    logger = logger.startExperiment();
    
    % Fetch file names
    featureFiles = dir(fullfile(DATA_SET_DIRECTORY, 'features*'));
    labelFiles = dir(fullfile(DATA_SET_DIRECTORY, 'labels*'));
     
    for i=1:numel(featureFiles)
        % Load test set
        logger.debug('runUnknownDataExperiment', 'Loading data set...');
        load(fullfile(DATA_SET_PATH, featureFiles{i}.name));
        load(fullfile(DATA_SET_PATH, labelFiles{i}.name));

        % Visualize test labels
        if VISUALIZE_TEST_LABELS
            visualizeLabels(labels, 'Test Labels');
        end

        % Create mask map (only showing -1 and 0)
        maskMap = labels;
        maskMap(labels > 0) = 0;

        if ENSEMBLE == true
            % Apply feature extraction
            logger.debug('runUnknownDataExperiment', ...
                         'Applying feature extraction...');
            cube = ...
                applyFeatureExtraction(cube, EXTRACTORS, ...
                                       maskMap, SAMPLE_SET_PATH);
        end

        % Apply trained classifier
        logger.debug('runUnknownDataExperiment', 'Applying classifier...');
        classifiedLabels = ...
            CLASSIFIER.classifyOn(cube, maskMap, i);
        logger.info('runUnknownDataExperiment', 'Instances classified');

        % Visualize predicted labels
        if VISUALIZE_PREDICTED_LABELS
            visMask = labels ~= 0;
            newLabelMap = cat(1, labels, ...
                              classifiedLabels.*visMask, ...
                              classifiedLabels);
            visualizeLabels(newLabelMap, ...
                            'Predicted Labels and Ground Truth');
        end

        % Calculate confusion matrix
        logger.debug('runUnknownDataExperiment', ...
                     'Calculating confusion matrix...');
        confMat(:, :, i) = confusionmat(...
            validListFromSpatial(labels, maskMap), ...
            validListFromSpatial(classifiedLabels, maskMap), ...
            'order', 0:24);

        % Calculate accuracy of current classifier
        accuracy = Evaluator.getAccuracy(confMat(2:end, 2:end, i));
        % Log accuracy
        logger.info('runUnknownDataExperiment', ...
                    sprintf('Current accuracy: %.3f', accuracy));

        % Free RAM
        clear('labels', 'cube');

    end

    % Stop logging
    logger = logger.stopExperiment();
    
    % Sum up all confusion matrices
    confMat = sum(confMat(2:end, 2:end, :), 3);
    % Log confusion matrix
    logger.logConfusionMatrix(confMat);
    % Compute accuracy measures
    measures = Evaluator.getAllMeasures(confMat);
    % Log accuracy measures
    logger.logMeasures(measures);
    
end

function featureCube = applyFeatureExtraction(featureCube, extractors, ...
                                              maskMap, sampleSetPath)
    for i = 1:numel(extractors)
        featureCube = extractors{i}.extractFeatures(featureCube, ...
            maskMap, sampleSetPath);
    end
end
