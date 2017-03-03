% config.m Configuration file for runExperiment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classifier configurations

% Example Classifier
exampleClassifierConfig = @ExampleClassifier;

% Random Forest - Parameters: numTrees
randomForest20Config = @() RandomForest(20);
randomForest100Config = @() RandomForest(100);

% Rotation Forest - Parameters: numTrees, splitParameter
rotationForest203Config = @() RotationForest(20,3);

% SVM - Parameters: Coding
svmConfig = @() SVMsvmlin(...
    'Coding', 'onevsall');

% TSVM - Parameters: Coding, UnlabeledRate
tsvmLinearConfig = @() TSVM(...
    'Coding', 'onevsall', ...
    'UnlabeledRate', 0.001);

% Convolutional Network - Parameters: cudnn, gpus, numEpochs
convNetConfig = @() ConvNet(...
    'cudnn', true, ...
    'gpus', [1], ...
    'numEpochs', 500, ...
    'plotErrorRates', true, ...
    'stoppingEpochWindow', 100, ...
    'stoppingErrorMargin', 0.005, ...
    'dropoutRate', 0.0, ...
    'doPooling', true);

% BasicEnsemble - Parameters: baseClassifiers, numClassifiers, 
%                             trainingInstanceProportions
basicEnsembleEC100_08Config = ...
    @() BasicEnsemble({ExampleClassifier}, ...
        100, ...
        0.8, ...
        VotingMode.WeightedMajority, ...
        0.1, ...
        true, ...
        true, ...
        true);
basicEnsembleLinearSVM100_001Config = ...
    @() BasicEnsemble({svmLinearConfig()}, ...
        100, ...
        0.002, ...
        VotingMode.WeightedMajority, ...
        0.5, ...
        true, ...
        true, ...
        true);
basicEnsembleRF20Config = ...
    @() BasicEnsemble({randomForest20Config()}, ...
        10, ...
        0.5, ...
        VotingMode.WeightedMajority, ...
        0.1, ...
        true, ...
        true, ...
        true);

% VisualizingClassifier
visualizingClassifierConfig = @() VisualizingClassifier();

% SpatialReg - Parameters: Classifier, R, [PropagationThreshold,]
%                          VisualizeSteps
randomForestLabelPropConfig = @() LabelPropagator(...
    randomForest20Config(), ...
    'R', 15, ...
    'PropagationThreshold', 0.0, ...
    'VisualizeSteps', false);
randomForestOutputRegConfig = @() OutputRegularizer(...
    '../results/RandomForest_20/MulticlassLda_5/20170217_1239/', ...
    'R', 5, ...
    'VisualizeSteps', false);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature extractor configurations

% SELD - Parameters: k, numDimensions
seld20_5Config = @() SELD(20, 5);
seld40_5Config = @() SELD(40, 5);
seld60_5Config = @() SELD(60, 5);
seld20_14Config = @() SELD(20, 14);
seld40_14Config = @() SELD(40, 14);
seld60_14Config = @() SELD(60, 14);

% PCA - Parameters: numDimensions
pca1Config = @() PCA(1);
pca5Config = @() PCA(5);
pca14Config = @() PCA(14);
pca20Config = @() PCA(20);
pca25Config = @() PCA(25);

% MulticlassLda
mclda5Config = @() MulticlassLda(5);
mclda14Config = @() MulticlassLda(14);

% ContinuumRemoval - Parameters: useMultithread
continuumRemoval= @() ContinuumRemoval(true);

% SpatialFeatureExtractor - Parameters: radius, implementationType
spatialFeatureExtractorConfig_20= @() SpatialFeatureExtractor(20, 2);
spatialFeatureExtractorConfig_15= @() SpatialFeatureExtractor(15, 2);
spatialFeatureExtractorConfig_10= @() SpatialFeatureExtractor(10, 2);
spatialFeatureExtractorConfig_5= @() SpatialFeatureExtractor(5, 1);

% Indices
indicesConfig = @() Indices();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment configurations

global NUMCLASSES;
NUMCLASSES = 24;

CLASSIFIER = convNetConfig();

EXTRACTORS = {mclda14Config()};



DATA_SET_PATH = ...
        '../data/ftp-iff2.iff.fraunhofer.de/ProcessedData/400-1000/';
SAMPLE_SET_PATH = ...
        ['../data/ftp-iff2.iff.fraunhofer.de/FeatureExtraction/' ...
            'Samplesets/sampleset_012.mat'];

RESULTS_PATH = '../results/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output configurations

% log levels:
% ALL   = 0
% TRACE = 1
% DEBUG = 2
% INFO  = 3
% WARN  = 4
% ERROR = 5
% FATAL = 6
% OFF   = 7
LOG_LEVEL = 3;

VISUALIZE_TRAIN_LABELS = false;
VISUALIZE_TEST_LABELS = false;
VISUALIZE_PREDICTED_LABELS = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
