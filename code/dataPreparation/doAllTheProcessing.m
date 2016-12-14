function doAllTheProcessing(path)
%DOALLTHEPROCESSING Executes all data processing functions on the path

    processMats(path);
    
    splitOneAndFour(path);
    
    maskZeros(path);
    
    mergeData(path);

end

