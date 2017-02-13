function err = errorMulticlass(~, labels, res)
% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;

    % be resilient to badly formatted labels
    if numel(labels) == size(predictions, 4)
      labels = reshape(labels,1,1,1,[]) ;
    end

    % skip null labels
    mass = single(labels(:,:,1,:) > 0) ;
    if size(labels,3) == 2
      % if there is a second channel in labels, used it as weights
      mass = mass .* labels(:,:,2,:) ;
      labels(:,:,2,:) = [] ;
    end

    m = min(5, size(predictions,3)) ;

    error = ~bsxfun(@eq, predictions, labels) ;
    err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
    err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;
end

