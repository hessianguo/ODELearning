function subds = partition(obj, partitionStrategy, partitionIndex)
%PARTITION Create a signalDatastore that is a partitioned portion of the
%original datastore
%   SUBSDS = PARTITION(SDS,NUMPARTITIONS,INDEX) partitions SDS into
%   NUMPARTITIONS parts and returns the partitioned datastore, SUBSDS,
%   corresponding to INDEX. Use the NUMPARTITIONS function to estimate a
%   reasonable value for NUMPARTITIONS.
%
%   SUBSDS = PARTITION(SDS,'Observations',INDEX) partitions SDS by files
%   (if the datastore contains file data) or members (if the datastore
%   contains in-memory data) and returns the partition corresponding to
%   scalar integer index, INDEX. An index equal to k points to the k-th
%   file or member of SDS.
%
%   SUBSDS = PARTITION(SDS,'Observations',OBSNAME) partitions SDS by files
%   (if the datastore contains file data) or members (if the datastore
%   contains in-memory data) and returns the partition corresponding to the
%   observation name, OBSNAME. OBSNAME is a file name for the file data
%   case, and a member name for the in-memory data case.
%
%   % EXAMPLE:
%      % Create a signal datastore and two partitions
%      data = {randn(100,1); randn(120,3); randn(135,2); randn(100,1);...
%              randn(150,2); randn(155,2); randn(85,10); randn(170,2);...
%              randn(145,3); randn(112,2)};
%      sds = signalDatastore(data,'SampleRate',1000);
%      sdsP1 = partition(sds,2,1);
%      sdsP2 = partition(sds,2,2);
%
%   See also numpartitions.

%   Copyright 2019 The MathWorks, Inc.

narginchk(3,3)

partitionStrategy = convertStringsToChars(partitionStrategy);
partitionIndex = convertStringsToChars(partitionIndex);

if isnumeric(partitionStrategy) && isnumeric(partitionIndex)
    % No op
elseif (ischar(partitionStrategy) && string(partitionStrategy) == "Observations") && (isnumeric(partitionIndex) || ischar(partitionIndex))        
    if obj.pInMemoryFlag
        partitionStrategy = 'Members';
    else
        partitionStrategy = 'Files';
    end
else
    error(message('signal:signalDatastore:signalDatastore:InvalidPartitionInputs'));    
end
subds = copy(obj);
subds.pDatastoreInternal = partition(subds.pDatastoreInternal,partitionStrategy,partitionIndex);

% Get correct time values if sample rate, sample time, or time values were
% specified and were set to vectors or matrices
if ~isempty(obj.pTimeInformationPropertyName) && obj.pTimeInformationPropertyName == "SampleRate"
    if ~isscalar(obj.pSampleRate)
        pIndices = getPartitionIndices(obj,subds);
        subds.pSampleRate = obj.pSampleRate(pIndices);
    end
elseif ~isempty(obj.pTimeInformationPropertyName) && obj.pTimeInformationPropertyName == "SampleTime"
    if ~isscalar(obj.pSampleTime)
        pIndices = getPartitionIndices(obj,subds);
        subds.pSampleTime = obj.pSampleTime(pIndices);
    end
elseif ~isempty(obj.pTimeInformationPropertyName) && obj.pTimeInformationPropertyName == "TimeValues"
    if ~isvector(obj.pTimeValues)
        pIndices = getPartitionIndices(obj,subds);
        subds.pTimeValues = obj.pTimeValues(:,pIndices);
    end
end
end