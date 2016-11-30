function [eigvector,Fe] = SELD(X, groundtruth, no_dims, k)
%SELD_MAIN combining LDA and Linearity Preserving Projection to discover 
%the class discrimination and spatial information of the data
%
%   [eigvector,Fe] = NLDLP(X, labels, no_dims, k)
%
%         Input:
%                 X                       - Input Data matrix. Each row vector of data is a data point
%                 groundtruth             - The labeled information of data points
%                 no_dims                 - Sets the number of dimensions of the feature points in the embedded feature space
%                 k                       - The number of neighbors
%
%         Output:
%                 eigvector               - Each column is an embedding function, for a new
%                                           data point (row vector) x,  y = x*eigvector
%                                           will be the embedding result of x.
%                 Fe                      - The coordinates of the low-dimensional data
%
%
%       Copyright notes
%       Author: Wenzhi Liao IPI, Telin, Ghent University, Belgium
%       Date: 24/6/2010
%
% W. Liao, A. Pizurica, P. Scheunders, W. Philips, Y. Pi, "Semi-Supervised Local Discriminant Analysis for Feature Extraction in Hyperspectral 
% Images", In IEEE Transactions on Geoscience and Remote Sensing, vol. 51, no. 1, pp. 184- 198, Jan. 2013.
    import SELD.*

    if size(X, 2) > size(X, 1)
        error('Number of samples should be higher than number of dimensions.');
    end
	% Make sure data is zero mean
    X=double(X);
    Xmean = mean(X, 1);
	X = X - repmat(Xmean, [size(X, 1) 1]);
    
    %%%%% Find the labeled samples and unlabeled samples
    Xlabeled=[];
    numlabeled=groundtruth(groundtruth>0);
    num=unique(numlabeled);

    
    for i=1:length(num)
        indlabeled=find(groundtruth==num(i));
        Xlabeled=[Xlabeled;X(indlabeled,:)];
    end
    indunlabeled=find(groundtruth<=0);
    Xunlabeled=X(indunlabeled,:);
    XT=[Xlabeled;Xunlabeled];
    
  %%%%%%%%%%perform LDA method  %%%%%%%%%%

    Qt=[];

    for i=1:length(num)
        indlabeled=find(groundtruth==num(i));
        matrix=1/length(indlabeled)*ones(length(indlabeled));
        Qt=[Qt,zeros(size(Qt,1),size(matrix,1))
            zeros(size(matrix,1),size(Qt,1)),matrix];       
    end
    
    II=sparse(1:size(Xlabeled,1),1:size(Xlabeled,1),ones(size(Xlabeled,1),1),size(Xlabeled,1)+size(Xunlabeled,1),size(Xlabeled,1)+size(Xunlabeled,1));


%%%%%%%%perform LPP method %%%%%%

    % Construct neighborhood graph
    disp('Constructing neighborhood graph...');
    if size(Xunlabeled, 1) < 4000
        G = L2_distance(Xunlabeled', Xunlabeled');
        % Compute neighbourhood graph
        [tmp, ind] = sort(G); 
        for i=1:size(G, 1)
            G(i, ind((2 + k):end, i)) = 0; 
        end
        G = sparse(double(G));
        G = max(G, G');             % Make sure distance matrix is symmetric
    else
        G = find_nn(Xunlabeled, k);
    end
    % Compute weights (W = G)
    G = G .^ 2;
	G = G ./ max(max(G));
    
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
    DP = XT' * ([zeros(size(Xlabeled,1)),zeros(size(Xlabeled,1),size(D,1));zeros(size(D,1),size(Xlabeled,1)),D]+...
        size(Xunlabeled,1)/size(Xlabeled,1)*[Qt,zeros(size(Qt,1),size(Xunlabeled,1));zeros(size(Xunlabeled,1),size(Qt,1)),zeros(size(Xunlabeled,1))]) * XT;
    
    LP = XT' * ([zeros(size(Xlabeled,1)),zeros(size(Xlabeled,1),size(D,1));zeros(size(D,1),size(Xlabeled,1)),L]+...
        size(Xunlabeled,1)/size(Xlabeled,1)*(II-[Qt,zeros(size(Qt,1),size(Xunlabeled,1));zeros(size(Xunlabeled,1),size(Qt,1)),zeros(size(Xunlabeled,1))])) * XT;
    
    DP = (DP + DP') / 2;
    LP = (LP + LP') / 2;
    LP(isnan(LP)) = 0; DP(isnan(DP)) = 0;
	LP(isinf(LP)) = 0; DP(isinf(DP)) = 0;
    
    % Solve generalize eigenproblem
    [eigvector, eigvalue] = eig(DP, LP);

    % Sort eigenvalues in descending order and get largest eigenvectors
    [eigvalue, ind] = sort(diag(eigvalue), 'descend');
    eigvector = eigvector(:,ind(1:no_dims));
    
    % Compute extracted features
    Fe = X * eigvector;
   
