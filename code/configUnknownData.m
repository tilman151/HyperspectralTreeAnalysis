% configUnknownData.m Configuration file for runUnknownDataExperiment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classifier configurations

bestEnsembleConfig = @(classifierPaths, sampleSetPath)...
                                BestClassifierEnsemble(...
                                        classifierPaths, ...
                                        VotingMode.Majority, ...
                                        true, ...
                                        sampleSetPath);
                                    
loadClassifierConfig = ...
    @(classifierPath) Classifier.loadFrom(classifierPath);

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

BASE_CLASSIFIER_PATHS = ...
    ['/home/cornelius/Projects/HyperspectralTreeAnalysis/results/'...
     'RandomForest_20__regularizedOutput_r5/MulticlassLda_14/'...
     '20170224_1752/model_1.mat'];

DATA_SET_PATH = ...
        '../data/ftp-iff2.iff.fraunhofer.de/Testdaten/';
SAMPLE_SET_PATH = ...
        ['../data/ftp-iff2.iff.fraunhofer.de/FeatureExtraction/' ...
            'Samplesets/sampleset_012.mat'];

RESULTS_PATH = '../results/unknown/';

CLASSIFIER = loadClassifierConfig(BASE_CLASSIFIER_PATHS);

EXTRACTORS = {mclda14Config()};

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
LOG_LEVEL = 2;

VISUALIZE_TRAIN_LABELS = false;
VISUALIZE_TEST_LABELS = false;
VISUALIZE_PREDICTED_LABELS = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
