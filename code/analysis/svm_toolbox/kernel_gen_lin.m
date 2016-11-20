function k = kernel_gen_lin(~)
% KERNEL_GEN_LIN Generates a linear kernel.
%
% Output:
%       k = Kernel function handle

%%%%%%%%%%%%%%%%%
% Define kernel %
%%%%%%%%%%%%%%%%%

function d = kernel_lin(x1,x2)
    d = dot(x1,x2);
end

%%%%%%%%%%%%%%%%%
% Return kernel %
%%%%%%%%%%%%%%%%%

k = @kernel_lin;

end