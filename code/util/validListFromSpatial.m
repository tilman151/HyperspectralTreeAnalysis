function outputList = validListFromSpatial(...
    spatialInput, maskMap, excludeUnlabeled)
%VALIDLISTFROMSPATIAL Extract list of valid pixels from the spatial data
%
%    For given spatially represented input data (e.g. a cube of features or
%    a map of labels), the first two dimensions are concatenated to create 
%    a list. In a second step, fill pixels that are indicated by a -1 in 
%    the mask map are removed. If excludeUnlabeled is true, pixels
%    indicated by 0 are also removed.
%    The input is expected to have dimensions X x Y x F with X and Y being
%    the image dimensions and F being the number of features or 1 for a
%    label map. The mask map has dimensions X x Y and Output has dimensions
%    (X * Y) x F.
%
%% Input:
%    spatialInput ..... Spatially represented input data (features or 
%                       labels). Dimensions X x Y x F with X and Y being 
%                       the image dimensions and F being the number of 
%                       features or 1 for a label map.
%    maskMap .......... Mask for the image pixels with dimensions X x Y.
%                       Indicates -1 for fill pixels and bigger values for 
%                       all other kinds of pixels.
%    excludeUnlabeled . Boolean that indicates, if unlabeled pixels should
%                       also be removed from the output. Default: false
%
%% Output:
%    outputList ....... List of features or labels with dimensions 
%                       (X * Y) x F with X and Y being the image dimensions 
%                       and F being the number of features or 1 for labels.
%
% Version: 2016-12-22
% Author: Cornelius Styp von Rekowski
%
    
    % Reshape feature cube to list
    [x, y, f] = size(spatialInput);
    outputList = reshape(spatialInput, x*y, f);

    % Remove fill pixels (-1) and optionally unlabeled pixels (0)
    if nargin < 3 || ~excludeUnlabeled
        labelThreshold = -1;
    else
        labelThreshold = 0;
    end
    
    outputList = outputList(maskMap > labelThreshold, :);
    
end

