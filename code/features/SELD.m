function [ extractedFeatures ] = SELD(features, labels, parameters)
%SELD perform semisupervised local discriminant analysis
%
%    The function extracts features using semisupervised local discriminant
%    analysis proposed by Wenzhi et al. (2013).
%
%% Input:
%    features ... (n x D) matrix containing n D-dimensional instances
%    labels ..... (n x 1) vector containing the corresponding labels
%    parameters . struct containing the parameters
%                 k ...... number of neighbors considered in the  
%                          unsupervised local linear feature extraction
%                          method
%                 numDim . number of dimensions of the extracted features
%                       
%% Output:
%    extractedFeatures . (n x numDim) matrix with the extracted features of
%                        the n input instances
% 
% Version: 2016-11-29
% Author: Marianne Stecklina
%

 [~, extractedFeatures] = SELD.SELD(features, labels, ...
     parameters.numDim, parameters.k);

end

