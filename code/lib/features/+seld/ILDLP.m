function [eigvector, mappedX] = ILDLP(X, labels, no_dims, aa, k)
%ILDLP linearly combines LDA and LPP
%
%   [W, Fe] = ILDLP(X, labels, no_dims)
%
%         Input:
%                 X                       - Input Data matrix. Each row vector of data is a data point
%                 labels                  - The labeled information of data points
%                 no_dims                 - Sets the number of dimensions of the feature points in the embedded feature space
%                 aa                      - parameter combining LDA and LPP
%                 k                       - The number of neighbors
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
    if size(X, 2) > size(X, 1)
        error('Number of samples should be higher than number of dimensions.');
    end
    if ~exist('k', 'var')
        k = 12;
    end
	% Make sure data is zero mean
    mapping.mean = mean(X, 1);
	X = bsxfun(@minus, X, mapping.mean);
	
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
	
	% Perform eigendecomposition of inv(Sw)*Sb
%     [W, lambda] = eig(Sb, Sw);
%     
%     % Sort eigenvalues and eigenvectors in descending order
%     lambda(isnan(lambda)) = 0;
% 	[lambda, ind] = sort(diag(lambda), 'descend');
% 	W = W(:,ind(1:min([no_dims size(W, 2)])));    
%     
% 	% Compute mapped data
% 	mappedX = X * W;
%     
 
    
    % Construct neighborhood graph
    disp('Constructing neighborhood graph...');
    if size(X, 1) < 4000
        G = L2_distance(X', X');
        % Compute neighbourhood graph
        [tmp, ind] = sort(G); 
        for i=1:size(G, 1)
            G(i, ind((2 + k):end, i)) = 0; 
        end
        G = sparse(double(G));
        G = max(G, G');             % Make sure distance matrix is symmetric
    else
        G = find_nn(X, k);
    end
    G = G .^ 2;
	G = G ./ max(max(G));
    
    % Compute weights (W = G)
    disp('Computing weight matrices...');
    
    % Compute Gaussian kernel (heat kernel-based weights)
    G(G ~= 0) = exp(-G(G ~= 0) / (2 * 1 ^ 2));
        
    % Construct diagonal weight matrix
    D = diag(sum(G, 2));
    
    % Compute Laplacian
    L = D - G;
    L(isnan(L)) = 0; D(isnan(D)) = 0;
	L(isinf(L)) = 0; D(isinf(D)) = 0;

    % Compute XDX and XLX and make sure these are symmetric
    disp('Computing low-dimensional embedding...');
    DP = X' * D * X;
    LP = X' * L * X;
    DP = (DP + DP') / 2;
    LP = (LP + LP') / 2;
    Slb=aa*Sb+(1-aa)*DP;
    Slt=aa*St+(1-aa)*LP;
    % Perform eigenanalysis of generalized eigenproblem (as in LEM)
%     [W, lambda] = eig(Sb, Sw);

    [eigvector, eigvalue] = eig(Slb, Slt);
    % Sort eigenvalues in descending order and get largest eigenvectors
    [eigvalue, ind] = sort(diag(eigvalue), 'descend');
    eigvector = eigvector(:,ind(1:no_dims));
    
    % Compute final linear basis and map data
    mappedX = X * eigvector;
   
