function TSVMModel = tsvmTrain(X, y, N)
%TSVMTRAIN Transductive SVM.
%   Detailed explanation goes here

addpath('svm_toolbox');

C1 = 10; % Misclassification penalty (labeled data)
C2 = 10; % Misclassification penalty (neutral data)
k = kernel_gen_rbf; % Kernel function handle
[f,SX,SY,SA,t] = svm_ovo(X, y, N, C1, C2, k);

TSVMModel = struct('f', f, 'SX', SX, 'SY', SY, 'SA', SA, 't', t);

