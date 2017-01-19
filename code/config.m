% config.m Configuration file for runExperiment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classifier configurations

% Example Classifier
exampleClassifierConfig = @ExampleClassifier;

% Random Forest - Parameters: numTrees
randomForest2Config = @() RandomForest(2);

% Rotation Forest - Parameters: numTrees, splitParameter
rotationForest22Config = @() RotationForest(2,2);

% SVM - Parameters: KernelFunction, PolynomialOrder, Coding
svmLinearConfig = @() SVM(...
    'KernelFunction', 'linear', ...
    'Coding', 'onevsone');
svmPolynomial1vs1Config = @() SVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'Coding', 'onevsone');
svmPolynomial1vsAllConfig = @() SVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'Coding', 'onevsone');

% TSVM - Parameters: C1 (misclassification penalty labeled),
%                    C2 (misclassification penalty unlabeled),
%                    KernelFunction, PolynomialOrder, Coding
tsvmLinearConfig = @() TSVM(...
    'C1', 10, ...
    'C2', 10, ...
    'KernelFunction', 'linear', ...
    'Coding', 'onevsone');
tsvmPolynomialConfig = @() TSVM(...
    'C1', 10, ...
    'C2', 10, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'Coding', 'onevsone');

% VisualizingClassifier
visualizingClassifierConfig = @() VisualizingClassifier();

% SpatialReg - Parameters: Classifier, R, LabelPropagation,
%                          OutputRegularization, PropagationThreshold,
%                          VisualizeSteps
visualizingSpatialRegConfig = @() SpatialReg(...
    visualizingClassifierConfig(), ...
    'R', 50, ...
    'LabelPropagation', true, ...
    'OutputRegularization', true, ...
    'PropagationThreshold', 0.0, ...
    'VisualizeSteps', true);

% BasicEnsemble - Parameters: baseClassifiers, numClassifiers, 
%                             trainingInstanceProportions
basicEnsembleEC100_08Config = ...
    @() BasicEnsemble({ExampleClassifier}, ...
        100, ...
        0.8, ...
        true);
basicEnsembleSVM100_01Config = ...
    @() BasicEnsemble({svmLinearConfig()}, ...
        100, ...
        0.001, ...
        true, ...
        true);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature extractor configurations

% SELD - Parameters: k, numDimensions
seld85Config = @() SELD(8, 5);

% PCA - Parameters: numDimensions
pca1Config = @() PCA(1);
pca5Config = @() PCA(5);
pca20Config = @() PCA(20);

% MulticlassLda
mcldaConfig = @() MulticlassLda;

% ContinuumRemoval - Parameters: useMultithread
continuumRemoval= @() ContinuumRemoval(true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment configurations

CLASSIFIER = randomForest20Config();

EXTRACTORS = {mcldaConfig(), SpatialFeatureExtractor(10, true)};

DATA_SET_PATH = ...
        '../data/ftp-iff2.iff.fraunhofer.de/Data/Hyperspectral/400-1000/';
SAMPLE_SET_PATH = ...
 '../data/ftp-iff2.iff.fraunhofer.de/Data/FeatureExtraction/sampleSet.mat';

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