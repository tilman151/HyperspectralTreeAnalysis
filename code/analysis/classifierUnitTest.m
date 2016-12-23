function classifierUnitTest(classifier, dataType)
    %UNITTEST A basic test for the provided classifiers
    %
    %   Run the given classifier on one data set and show the resulting
    %   confusion matrix.
    %
    %% Input:
    %    classifier ... Instance of the classifier to be tested.
    %    dataType ..... Data type to be used, either 'mat', 'csv' or 
    %                   'test'. For the option 'test', a small artificial
    %                   data sample is created. Default option is 'test'.
    %
    %% Example:
    %    unitTest(SVM());
    %
    % Version: 2016-12-22
    % Author: Cornelius Styp von Rekowski
    %
    
    % Default arguments
    if nargin < 2
        dataType = 'test';
    end
    
    % Load data
    disp('Load data');
    switch dataType
        case 'mat'
            path = ['../data/ftp-iff2.iff.fraunhofer.de/Data/'...
                'Hyperspectral/1000-2500/F2G9_02.mat'];
            [labelMap, featureCube] = Importer.loadDataFrom(path);
        case 'csv'
            path = ['../data/ftp-iff2.iff.fraunhofer.de/Data/CSV/'...
                '400-1000/spatialspectral/samples-32.csv'];
            [labelMap, featureCube] = Importer.loadDataFrom(path);
        otherwise
            % Test data set looks like this:
            % TODO: Create test data set that also has spatial information
            % 
            % 8|    +
            % 7|      o
            % 6|        o
            % 5|  o       o
            % 4|    -       o
            % 3|      o
            % 2|        o
            % 1|F   *
            %   ----------------
            %   1 2 3 4 5 6 7 8
            % 
            % + = class 1
            % - = class 2
            % * = class 3
            % o = unlabeled sample
            % F = fill pixel
            %
            
            labelMap             = [1; 0; 0; 0; 0; 0; 2; 0; 0; 3; -1];
            
            % X-values
            featureCube          = [3; 4; 5; 6; 7; 2; 3; 4; 5; 3; 1];
            
            % Y-values
            featureCube(:, :, 2) = [8; 7; 6; 5; 4; 5; 4; 3; 2; 1; 1];
    end
    
    % Train model
    disp('Train model');
    classifier.trainOn(featureCube, labelMap);
    
    % Predict labels for all data samples
    disp('Predict labels');
    maskMap = labelMap;
    maskMap(labelMap > 0) = 0;
    predictedLabelMap = classifier.classifyOn(featureCube, maskMap);
    
    % Calculate and show the confusion matrix
    confMat = confusionmat(...
        validListFromSpatial(labelMap, maskMap), ...
        validListFromSpatial(predictedLabelMap, maskMap), ...
        'order', 0:17);
    disp('Confusion Matrix:');
    disp(confMat);
end

