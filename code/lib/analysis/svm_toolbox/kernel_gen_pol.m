function k = kernel_gen_pol(args)
% KERNEL_GEN_POL Generates a polynomial kernel.
%
% Input:
%       args(2,1) or args(1,2) = Kernel arguments (see code)
%
% Output:
%       k = Kernel function handle

%%%%%%%%%%%%%%%%%
% Define kernel %
%%%%%%%%%%%%%%%%%

a = args(1); b = args(2);

function d = kernel_pol(x1,x2)
    d = (dot(x1,x2)+a)^b;
end

%%%%%%%%%%%%%%%%%
% Return kernel %
%%%%%%%%%%%%%%%%%

k = @kernel_pol;

end