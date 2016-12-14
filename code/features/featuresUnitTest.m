function featuresUnitTest(featureExtractor, dataType)
    %FEATURESUNITTEST A basic test for the provided classifiers
    %
    %   Run the given feature extractor on one data set.
    %
    %% Input:
    %    featureExtractor . Instance of the feature extractor to be tested.
    %    dataType ......... Data type to be used, either 'mat', 'csv' or 
    %                       'test'. For the option 'test', a small 
    %                       artificial data sample is created. Default 
    %                       option is 'test'.
    %
    %% Example:
    %    FeaturesUnitTest(SELD);
    %
    % Version: 2016-12-12
    % Author: Marianne Stecklina
    %
    addpath('lib/features/');
    
    % test calculation of transformation matrix
    disp('Calculate transformation matrix:');
    
    sampleSet.features          = [1 5 3 5 2; 
                                   2 3 4 6 3; 
                                   1 2 4 6 3; 
                                   5 3 2 3 4];
    sampleSet.labels            = [1 1 2 2];
    sampleSet.unlabeledFeatures = [3 5 2 4 6;
                                   2 3 4 5 6;
                                   5 3 4 3 3;
                                   2 3 2 5 3;
                                   1 1 2 5 2;
                                   3 5 2 3 2;
                                   2 5 4 3 2];
    
    transformationMatrix = featureExtractor.calculateTransformation(sampleSet);
    disp(transformationMatrix);
    
    % test application of transformation matrix
    
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
            [~, cube] = Importer.loadDataFrom(path);
        case 'csv'
            path = ['../data/ftp-iff2.iff.fraunhofer.de/Data/CSV/'...
                '400-1000/spatialspectral/samples-32.csv'];
            [~, cube] = Importer.loadDataFrom(path);
        otherwise
            cube          = [1 2 3;
                             4 5 6;
                             7 8 9];
            cube(:, :, 2) = [1 5 5;
                             4 2 3;
                             6 1 9];
            cube(:, :, 3) = [1 4 3;
                             8 6 6;
                             2 6 9];
            cube(:, :, 4) = [1 2 5;
                             1 5 7;
                             7 2 4];
            cube(:, :, 5) = [3 2 3;
                             5 7 3;
                             7 8 8];
    end
    
    % Extract features
    disp('Apply transformation');
    features = featureExtractor.applyTransformation(cube, ...
        transformationMatrix);
    disp('New cube size:');
    disp(size(features));
end

