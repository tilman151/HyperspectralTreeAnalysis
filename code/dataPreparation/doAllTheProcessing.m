function doAllTheProcessing(path)
%DOALLTHEPROCESSING Executes all data processing functions on the path
    
    if nargin < 1
        path = ['../data/ftp-iff2.iff.fraunhofer.de/Data/Hyperspectral/'...
                '400-1000/'];
    end
    
    processMats(path);
    
    splitOneAndFour(path);
    
    maskZeros(path);
    
    mergeData(path);

end

