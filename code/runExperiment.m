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
    confMat = zeros(18, 18, crossValidator.k);
    
    % Create logger singleton
    logPath = Logger.createLogPath(RESULTS_PATH, CLASSIFIER, EXTRACTORS);
    logger = Logger.createLoggerSingleton(logPath);
    % Log configuration
    logger.logConfig(CLASSIFIER, ...
                     EXTRACTORS, ...
                     SAMPLE_SET_PATH, ...
                     DATA_SET_PATH, ...
                     crossValidator.crossValParts);
    % Start logging
    logger = logger.startExperiment();
                        
    % For each test and training set
    for i = 1:crossValidator.k
        logger.info('runExperiment', ['Iteration ', num2str(i)]);
        % Load training set
        [trainLabelMap, trainFeatureCube] = ...
            crossValidator.getTrainingSet(i);
        
        % Visualize train labels
        if VISUALIZE_TRAIN_LABELS
            visualizeLabels(trainLabelMap, 'Training Labels');
        end
        
        % Apply feature extraction
        trainFeatureCube = ...
            applyFeatureExtraction(trainFeatureCube, EXTRACTORS, ...
                                   SAMPLE_SET_PATH);
        
        % Train classifier
        CLASSIFIER.trainOn(trainFeatureCube, trainLabelMap);
        logger.info('runExperiment', 'classifier trained');
        % Free RAM
        clear('trainLabelMap', 'trainFeatureCube');
        
        
        % Load test set
        [testLabelMap, testFeatureCube] = crossValidator.getTestSet(i);
        
        % Visualize test labels
        if VISUALIZE_TEST_LABELS
            visualizeLabels(testLabelMap, 'Test Labels');
        end
        
         % Apply feature extraction
        testFeatureCube = ...
            applyFeatureExtraction(testFeatureCube, EXTRACTORS, ...
                                   SAMPLE_SET_PATH);
        
        % Create mask map (only showing -1 and 0)
        maskMap = testLabelMap;
        maskMap(testLabelMap > 0) = 0;
        
        % Apply trained classifier
        classifiedLabelMap = ...
            CLASSIFIER.classifyOn(testFeatureCube, maskMap);
        logger.info('runExperiment', 'test instances classified');
        
        % Visualize predicted labels
        if VISUALIZE_PREDICTED_LABELS
            visualizeLabels(classifiedLabelMap, 'Predicted Labels');
        end
        
        % Calculate confusion matrix
        confMat(:, :, i) = confusionmat(...
            validListFromSpatial(testLabelMap, maskMap), ...
            validListFromSpatial(classifiedLabelMap, maskMap), ...
            'order', 0:17);
        
        % Free RAM
        clear('testLabelMap', 'testFeatureCube');
        
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
                                              sampleSetPath)
    for i = 1:size(extractors, 1)
        featureCube = extractors{i}.extractFeatures(featureCube, ...
            sampleSetPath);
    end
end