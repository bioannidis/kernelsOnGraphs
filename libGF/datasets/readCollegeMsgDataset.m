function [t_adjacency] = readCollegeMsgDataset
%%
%This function returns the brain data in a tensor so that every slice of
%the tensor correspond to different measured signal
%The adjacency is constructed using partial correlation

%% initialize
pth = which('readCollegeMsgDataset');
folder=[pth(1:end-(length('readCollegeMsgDataset')+2))  'CollegeMsgDataset\'];

%%
% Eight sets of measurements were taken on this patient, corresponding to 
% eight different epileptic seizures.  An epileptologist, examining the 
% data for each seizure, identified two key periods relating to the seizure
% i.e., the so-called pre-ictal and ictal periods, and time series data 
% for those periods were extracted.

t_adjacency=zeros(1899,1899,4);
s_chuncks=floor(59835/4);
% ictal period
file=strcat(folder,'CollegeMsg.txt');
s_tensorNum=1;
m_rawData = importdata(file);
for s_it=1:size(m_rawData,1)
    if (s_chuncks==0)
        s_chuncks=floor(59835/4);
        s_tensorNum=s_tensorNum+1;
    end
    t_adjacency(m_rawData(s_it,1),m_rawData(s_it,2),s_tensorNum)=t_adjacency(m_rawData(s_it,1),m_rawData(s_it,2),s_tensorNum)+1;
    s_chuncks=s_chuncks-1;
end

end
