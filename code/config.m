% config.m Configuration file for runExperiment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classifier configurations

% Example Classifier
exampleClassifierConfig = @ExampleClassifier;

% Random Forest - Parameters: numTrees
randomForest100Config = @() RandomForest(100);

% Rotation Forest - Parameters: numTrees, splitParameter
rotationForest56Config = @() RotationForest(5,6);

% SVM - Parameters: KernelFunction, PolynomialOrder, Coding
svmLinearConfig = @() SVM(...
    'KernelFunction', 'linear', ...
    'Coding', 'onevsone');
svmPolynomialConfig = @() SVM(...
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

% SpatialReg - Parameters: classifier, r, 
%                          doLabelPropagation, doRegularization
svmLinearSpatiallyRegularizedConfig = @() SpatialReg(...
    svmLinearConfig(), ...
    5, ...
    true, ...
    true);
tsvmLinearSpatiallyRegularizedConfig = @() SpatialReg(...
    tsvmLinearConfig(), ...
    5, ...
    true, ...
    true);

% BasicEnsemble - Parameters: baseClassifiers, numClassifiers, 
%                             trainingInstanceProportions
basicEnsembleEC100_08Config = ...
    @() BasicEnsemble({ExampleClassifier}, ...
        100, ...
        0.8);
basicEnsembleSVM100_08Config = ...
    @() BasicEnsemble({svmLinearConfig()}, ...
        100, ...
        0.1);

% VisualizingClassifier
visualizingClassifierConfig = @() VisualizingClassifier();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature extractor configurations

% SELD - Parameters: k, numDimensions
seld85Config = @() SELD(8, 5);

% PCA - Parameters: numDimensions
pca1Config = @() PCA(1);
pca5Config = @() PCA(5);

% MulticlassLda
mcldaConfig = @() MulticlassLda;

% ContinuumRemoval
continuumRemoval= @() ContinuumRemoval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment configurations

CLASSIFIER = ExampleClassifierConfig()

EXTRACTORS = {mcldaConfig()}

DATA_SET_PATH = ...
        '../data/ftp-iff2.iff.fraunhofer.de/Data/Hyperspectral/400-1000/'
SAMPLE_SET_PATH = ...
  '../data/ftp-iff2.iff.fraunhofer.de/Data/FeatureExtraction/sampleSet.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output configurations

VISUALIZE_TRAIN_LABELS = false;
VISUALIZE_TEST_LABELS = false;
VISUALIZE_PREDICTED_LABELS = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%