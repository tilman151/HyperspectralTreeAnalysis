function featureVec = mapToVec(featureMap)
%MAPTOVEC Reshape a given feature map to a single vector
%
%    For a map of features that has feature values for each pixel of a
%    two-dimensional image, the first two dimensions are concatenated to
%    create one vector of features.
%    Input map is expected to have dimensions X x Y x F, output has 
%    dimensions (X * Y) x F. This also works for label maps having F=1.
%
%% Input:
%    featureMap . Map of features for each image pixel. Dimensions 
%                 X x Y x F with X and Y being the image dimensions and F 
%                 being the number of features.
%
%% Output:
%    featureVec . Reshaped feature map with dimensions (X * Y) x F.
%
% Version: 2016-11-30
% Author: Cornelius Styp von Rekowski
%

% TODO: Remove empty pixels (class = -1). This needs a given mask.
[x, y, f] = size(featureMap);
featureVec = reshape(featureMap, x*y, f);

end

