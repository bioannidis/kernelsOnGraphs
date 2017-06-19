function [m_adjacency] = readFacebookDataset
%%
%This function returns the brain data in a tensor so that every slice of
%the tensor correspond to different measured signal
%The adjacency is constructed using partial correlation

%% initialize
pth = which('readFacebookDataset');
folder=[pth(1:end-(length('readFacebookDataset')+2))  'FacebookDataset\'];

%%
% Eight sets of measurements were taken on this patient, corresponding to 
% eight different epileptic seizures.  An epileptologist, examining the 
% data for each seizure, identified two key periods relating to the seizure
% i.e., the so-called pre-ictal and ictal periods, and time series data 
% for those periods were extracted.

m_adjacency=zeros(347);
% ictal period
file=strcat(folder,'0.edges');
m_rawData = importdata(file);
for s_it=1:size(m_rawData,1)
m_adjacency(m_rawData(s_it,1),m_rawData(s_it,2))=1;
end

end