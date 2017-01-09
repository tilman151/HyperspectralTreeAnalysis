function confMat = runExperiment(configFilePath)
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
    %%  Input:
    %       configFilePath ..... a path to the config file file
    %
    %%  Output:
    %       confMat ............ a confusion matrix yielded by cross
    %                            validation using the classifier and the
    %                            data set defined in the config file.
    %
    % Version: 2016-12-22
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

    % TODO: Check config validity
    
    % Load crossValParts and filesPerClass
    load('crossValParts.mat');
    
    % Initialize CrossValidator object
    crossValidator = CrossValidator(crossValParts, filesPerClass,...
                                    dataSetPath);
    
    % Initialize confusion matrix
    confMat = zeros(18, 18, crossValidator.k);
      
    % For each test and training set
    for i = 1:crossValidator.k
        
        % Load training set
        [trainLabelMap, trainFeatureCube] = ...
            crossValidator.getTrainingSet(i);
        
        % Apply feature extraction
        trainFeatureCube = ...
            applyFeatureExtraction(trainFeatureCube, extractors);
        
        % Train classifier
        classifier.trainOn(trainFeatureCube, trainLabelMap);
        
        % Free RAM
        clear('trainLabelMap', 'trainFeatureCube');
        
        
        % Load test set
        [testLabelMap, testFeatureCube] = crossValidator.getTestSet(i);
        
         % Apply feature extraction
        testFeatureCube = ...
            applyFeatureExtraction(testFeatureCube, extractors);
        
        % Create mask map (only showing -1 and 0)
        maskMap = testLabelMap;
        maskMap(testLabelMap > 0) = 0;
        
        % Apply trained classifier
        classifiedLabelMap = ...
            classifier.classifyOn(testFeatureCube, maskMap);
        
        % Calculate confusion matrix
        confMat(:, :, i) = confusionmat(...
            validListFromSpatial(testLabelMap, maskMap), ...
            validListFromSpatial(classifiedLabelMap, maskMap), ...
            'order', 0:17);
        
        % Free RAM
        clear('testLabelMap', 'testFeatureCube');
        
    end
    
    % Sum up all confusion matrices
    confMat = sum(confMat(2:end, 2:end, :), 3);
    
end

function featureCube = applyFeatureExtraction(featureCube, extractors)
    for i = 1:size(extractors, 1)
        featureCube = extractors{i}.extractFeatures(featureCube);
    end
end
