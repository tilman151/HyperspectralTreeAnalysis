function processMatsInSubdir( directoryPath )
%PROCESSMATSINSUBDIR Applying processMats to all subdirectories recursively
%   
%%  Input:
%       path ... a path to the Hyperspectral data directory.
directoryContent = dir(directoryPath);
currentAndParentFolder = ismember({directoryContent.name},  {'.', '..'});
directoryContent = directoryContent(~currentAndParentFolder);
isDirectory = cellfun(@isdir, {directoryContent.name});

subDirectoryPathes = {directoryContent.name};
subDirectoryPathes = subDirectoryPathes(isDirectory);
cellfun(@(subpath)(processMatsInSubdir([directoryPath '/' subpath])), subDirectoryPathes);
if sum(isDirectory) == 0
    processMats(directoryPath);        
end
end

