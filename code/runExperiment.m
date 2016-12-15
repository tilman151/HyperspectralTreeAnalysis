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
    % Version: 2016-11-29
    % Author: Tilman Krokotsch
    %%
    
    % Add the code directory and all subdirectories to path
    addpath(genpath('./'));

    % Read and execute config file
    run(configFilePath);

    % TODO: Check config validity
    
    % Load crossValParts and filesPerClass
    load('crossValParts.mat');
    
    % Initialize CrossValidator object
    crossValidator = CrossValidator(crossValParts, filesPerClass,...
                                    dataSetPath);
    
    % Initialize confusion matrix
    confMat = zeros(19, 19, crossValidator.k);
      
    % For each test and training set
    for i = 1:crossValidator.k
        
        % Load training set
        [trainLabels, trainFeatures] = crossValidator.getTrainingSet(i);
        % Apply feature extraction
        trainFeatures = applyFeatureExtraction(trainFeatures, extractors);
        % Train classifier
        classifier.trainOn(trainLabels, trainFeatures);
        % Free RAM
        clear('trainLabels', 'trainFeatures');
        
        % Load test set
        [testLabels, testFeatures] = crossValidator.getTestSet(i);
         % Apply feature extraction
        testFeatures = applyFeatureExtraction(testFeatures, extractors);
        % Calculate instance mask
        instanceMask = testLabels;
        instanceMask(testLabels > 0) = 1;
        % Apply trained classifier
        classifiedLabels = classifier.classifyOn...
                                              (testFeatures, instanceMask);
        
        % Calculate confusion matrix
        confMat(:, :, i) = confusionmat(testLabels, classifiedLabels,...
                               'order', -1:17);
    end
    
    % Sum up all confusion matrices
    confMat = sum(confMat(2:end, 2:end, :), 3);
    
end

function features = applyFeatureExtraction(features, extractors)
    for i = 1:size(extractors, 1)
        features = extractors(i).extractFeatures(features);
    end
end
