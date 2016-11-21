function k = kernel_gen_rbf(sigma)
% KERNEL_GEN_RBF Generates a radial basis function kernel.
%
% Input:
%       sigma(1,1) = Kernel argument (see code)
%
% Output:
%       k = Kernel function handle

%%%%%%%%%%%%%%%%%
% Define kernel %
%%%%%%%%%%%%%%%%%

function d = kernel_rbf(x1,x2)
    d = exp(-(norm(x1-x2)^2)/(2*(sigma^2)));
end

%%%%%%%%%%%%%%%%%
% Return kernel %
%%%%%%%%%%%%%%%%%

k = @kernel_rbf;

end