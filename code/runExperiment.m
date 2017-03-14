function runExperiment(configFilePath)
    %RUNEXPERIMENT Run a machine learning experiment
    %              with the parameters defined in the configuration file
    %
    %   The function takes a path to a configuration file and reads it. The
    %   statements in the file are executed to initialize the needed
    %   variables for the experiment.
    %   
    %   The variables crossValParts and filesPerClass are loaded and the
    %   CrossValidator object is initialized. For each column in
    %   crossValParts the classifier is trained on the calculated training
    %   set and evaluated on the test set. Afterwards the accumulated
    %   confusion matrix is calculated.
    %
    %   The configuration, all actions of this function and the confusion
    %   matrix are logged.
    %
    %%  Input:
    %       configFilePath ..... a path to the config file file
    %
    %
    % Version: 2017-01-11
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
        configFilePath = './config.m';
    end
    run(configFilePath);
    
    % Load crossValParts and filesPerClass
    load('crossValParts.mat');
    
    % Initialize CrossValidator object
    crossValidator = CrossValidator(crossValParts, filesPerClass,...
                                    DATA_SET_PATH);
    
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
    startTime = datetime('now');
    
    save([logger.getLogPath() '/FeatureExtractors'], 'EXTRACTORS');
    
    % Initialize accuracy variable
    accuracies = zeros(crossValidator.k, 1);
    
    % For each test and training set
    for i = 1:crossValidator.k
        logger.info('runExperiment', ['Iteration ', num2str(i)]);
        
        % Load training set
        logger.debug('runExperiment', 'Loading training set...');
        [trainLabelMap, trainFeatureCube] = ...
            crossValidator.getTrainingSet(i);
        
        % Visualize train labels
        if VISUALIZE_TRAIN_LABELS
            visualizeLabels(trainLabelMap, 'Training Labels');
        end
        
        % Create mask map (only showing -1 and 0)
        maskMap = trainLabelMap;
        maskMap(trainLabelMap > 0) = 0;
        
        % Apply feature extraction
        logger.debug('runExperiment', 'Applying feature extraction...');
        trainFeatureCube = ...
            applyFeatureExtraction(trainFeatureCube, EXTRACTORS, ...
                                   maskMap, SAMPLE_SET_PATH);
        
        % Train classifier
        logger.debug('runExperiment', 'Training classifier...');
        CLASSIFIER.trainOn(trainFeatureCube, trainLabelMap);
        logger.info('runExperiment', 'Classifier trained');
        
        % Free RAM
        clear('trainLabelMap', 'trainFeatureCube');
        
        
        % Load test set
        logger.debug('runExperiment', 'Loading test set...');
        [testLabelMap, testFeatureCube] = crossValidator.getTestSet(i);
        
        % Visualize test labels
        if VISUALIZE_TEST_LABELS
            visualizeLabels(testLabelMap, 'Test Labels');
        end
        
        % Create mask map (only showing -1 and 0)
        maskMap = testLabelMap;
        maskMap(testLabelMap > 0) = 0;
        
         % Apply feature extraction
        logger.debug('runExperiment', 'Applying feature extraction...');
        testFeatureCube = ...
            applyFeatureExtraction(testFeatureCube, EXTRACTORS, ...
                                   maskMap, SAMPLE_SET_PATH);
        
        % Apply trained classifier
        logger.debug('runExperiment', 'Applying trained classifier...');
        classifiedLabelMap = ...
            CLASSIFIER.classifyOn(testFeatureCube, maskMap, i);
        logger.info('runExperiment', 'Test instances classified');
        
        % Visualize predicted labels
        if VISUALIZE_PREDICTED_LABELS
            visMask = testLabelMap ~= 0;
            newLabelMap = cat(1, testLabelMap, ...
                              classifiedLabelMap.*visMask, ...
                              classifiedLabelMap);
            visualizeLabels(newLabelMap, ...
                            'Predicted Labels and Ground Truth');
        end
        
        % Calculate confusion matrix
        logger.debug('runExperiment', 'Calculating confusion matrix...');
        confMat(:, :, i) = confusionmat(...
            validListFromSpatial(testLabelMap, maskMap), ...
            validListFromSpatial(classifiedLabelMap, maskMap), ...
            'order', 0:24);
        
        % Calculate accuracy of current classifier
        accuracy = Evaluator.getAccuracy(confMat(2:end, 2:end, i));
        accuracies(i) = accuracy;
        % Log accuracy
        logger.info('runExperiment', ...
                    sprintf('Current accuracy: %.3f', accuracy));
        % Save classifier to log directory
        CLASSIFIER.saveTo(confMat(2:end, 2:end, i), ...
                          logger.getLogPath(), ...
                          num2str(i));
        logger.debug('runExperiment', 'Saved classifier');
        
        % Free RAM
        clear('testLabelMap', 'testFeatureCube');
        
    end

    % Log accuracy statistics
    logger.info('runExperiment', ...
                sprintf('Accuracy standard deviation: %.3f', ...
                        std(accuracies)));
    % Log duration
    logger.info('runExperiment_Duration', ...
                datetime('now') - startTime);
                             
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
