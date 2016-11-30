function [W, mappedX] = lda(X, labels, no_dims)
%LDA Perform linear discriminant analysis algorithm
%
%   [W, Fe] = lda(X, labels, no_dims)
%
%         Input:
%                 X                       - Input Data matrix. Each row vector of data is a data point
%                 labels                  - The labeled information of data points
%                 no_dims                 - Sets the number of dimensions of the feature points in the embedded feature space
%                
%
%         Output:
%                 W                       - Each column is an embedding function, for a new
%                                           data point (row vector) x,  y = x*eigvector
%                                           will be the embedding result of x.
%                 Fe                      - The coordinates of the low-dimensional data
%
%
%       Copyright notes
%       Author: Wenzhi Liao IPI, Telin, Ghent University, Belgium
%       Date: 25/2/2010



    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
	
	% Make sure data is zero mean
	Xmean = mean(X, 1);
	X = X - repmat(Xmean, [size(X, 1) 1]);
    
	% Make sure labels are nice
	[classes, bar, labels] = unique(labels);
    nc = length(classes);
	
	% Intialize Sw
	Sw = zeros(size(X, 2), size(X, 2));
    
    % Compute total covariance matrix
    St = cov(X);

	% Sum over classes
	for i=1:nc
        
        % Get all instances with class i
        cur_X = X(labels == i,:);

		% Update within-class scatter
		C = cov(cur_X);
		p = size(cur_X, 1) / (length(labels) - 1);
		Sw = Sw + (p * C);
    end
    
    % Compute between class scatter
    Sb       = St - Sw;
    Sb(isnan(Sb)) = 0; Sw(isnan(Sw)) = 0;
	Sb(isinf(Sb)) = 0; Sw(isinf(Sw)) = 0;
    
    % Make sure not to embed in too high dimension
    if nc <= no_dims
        no_dims = nc - 1;
        warning(['Target dimensionality reduced to ' num2str(no_dims) '.']);
    end
	
	% Perform eigendecomposition of inv(Sw)*Sb
    [W, lambda] = eig(Sb, Sw);
    
    % Sort eigenvalues and eigenvectors in descending order
    lambda(isnan(lambda)) = 0;
	[lambda, ind] = sort(diag(lambda), 'descend');
	W = W(:,ind(1:min([no_dims size(W, 2)])));    
    
	% Compute mapped data
	mappedX = X * W;
    
 
    