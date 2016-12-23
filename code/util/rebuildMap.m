function labelMap = rebuildMap(labelList, maskMap)
%REBUILDMAP Rebuild the map from the list of labels and the given mask
%
%    The given list of labels is reshaped to the spatial map
%    representation. This requires reinserting fill pixels that have been
%    removed in a previous step according to the mask map.
%    The label list has dimensions (X * Y) x 1, mask map has dimensions 
%    X x Y and the output label map has dimensions X x Y with X and Y
%    being the image dimensions.
%
%% Input:
%    labelList .. List of labels. Dimensions (X * Y) x 1 with X and Y being
%                 the image dimensions.
%    maskMap .... Mask for the image pixels with dimensions X x Y.
%                 Indicates -1 for fill pixels and bigger values for all
%                 other kinds of pixels.
%
%% Output:
%    labelMap ... Label map with dimensions X x Y and reinserted fill
%                 pixels.
%
% Version: 2016-12-22
% Author: Cornelius Styp von Rekowski
%
    
    labelMap = maskMap;
    labelMap(maskMap >= 0) = labelList;

end

