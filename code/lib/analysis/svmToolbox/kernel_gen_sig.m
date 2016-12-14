function k = kernel_gen_sig(args)
% KERNEL_GEN_SIG Generates a sigmoidal kernel.
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

function d = kernel_sig(x1,x2)
    d = tanh(a*dot(x1,x2)-b);
end

%%%%%%%%%%%%%%%%%
% Return kernel %
%%%%%%%%%%%%%%%%%

k = @kernel_sig;

end