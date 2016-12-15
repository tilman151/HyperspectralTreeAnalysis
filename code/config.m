% config.m Configuration file for runExperiment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classifier configurations

exampleClassifierStruct = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature extractor configurations

seldStruct = 0; % TODO

pcaStruct = 0; % TODO

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment configurations

classifier = ExampleClassifier

extractors = {SELD(seldStruct);
              PCA(pcaStruct)}

dataSetPath = ...
        '../data/ftp-iff2.iff.fraunhofer.de/Data/Hyperspectral/400-1000/'
sampleSetPath = ...
  '../data/ftp-iff2.iff.fraunhofer.de/Data/FeatureExtraction/sampleSet.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%