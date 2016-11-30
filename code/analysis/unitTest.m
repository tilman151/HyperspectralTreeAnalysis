function unitTest(classifier, dataType)
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
    %    unitTest(SVM);
    %
    % Version: 2016-11-30
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
            [labels, cube] = Importer.loadDataFrom(path);
        case 'csv'
            path = ['../data/ftp-iff2.iff.fraunhofer.de/Data/CSV/'...
                '400-1000/spatialspectral/samples-32.csv'];
            [labels, cube] = Importer.loadDataFrom(path);
        otherwise
            % Test data set looks like this:
            % 
            % 8|    +
            % 7|      o
            % 6|        o
            % 5|  o       o
            % 4|    -       o
            % 3|      o
            % 2|        o
            % 1|    *
            %   ----------------
            %   1 2 3 4 5 6 7 8
            % 
            % + = class 1
            % - = class 2
            % * = class 3
            % o = unlabeled sample
            %
            labels        = [1; 0; 0; 0; 0; 0; 2; 0; 0; 3];
            cube          = [3; 4; 5; 6; 7; 2; 3; 4; 5; 3]; % X-values
            cube(:, :, 2) = [8; 7; 6; 5; 4; 5; 4; 3; 2; 1]; % Y-values
    end
    
    % Train model
    disp('Train model');
    classifier.trainOn(cube, labels);
    
    % Predict labels for all data samples
    disp('Predict labels');
    predictedLabels = classifier.classifyOn(cube);
    
    % Calculate and show the confusion matrix
    confMat = confusionmat(mapToVec(labels), predictedLabels);
    disp('Confusion Matrix:');
    disp(confMat);
end

