function [m_adjacency,m_messagesTimeSeries] = readCollegeMsgDatasetAdjAndSignal
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

m_adjacency=zeros(1899,1899);
s_chuncks=floor(59835/2);
% ictal period
file=strcat(folder,'CollegeMsg.txt');
%s_tensorNum=1;
m_rawData = importdata(file);
for s_it=1:s_chuncks
    m_adjacency(m_rawData(s_it,1),m_rawData(s_it,2))=m_adjacency(m_rawData(s_it,1),m_rawData(s_it,2))+1;
    %s_chuncks=s_chuncks-1;
end
m_adjacency=m_adjacency+m_adjacency';
g=Graph('m_adjacency',m_adjacency);
v_indComponent=g.getComponents{1};
s_timeInd=6;
s_chuncksForMessages=floor(59835/(s_timeInd));
m_messagesTimeSeries=zeros(1899,s_timeInd);
v_messagesForThisChunk=zeros(1899,1);
v_indboolSubstantial=ones(1899,1);
s_substantialMessages=10;
    s_ind=1;
for s_it=s_chuncks+1 :size(m_rawData,1)
    if (s_chuncksForMessages==0)
        s_chuncksForMessages=floor(59835/(s_timeInd));
        m_messagesTimeSeries(:,s_ind)=v_messagesForThisChunk;
        s_ind=s_ind+1;
        v_indSubSubstatial=v_messagesForThisChunk>=s_substantialMessages;
        v_indboolSubstantial=v_indboolSubstantial&v_indSubSubstatial;
        v_messagesForThisChunk=zeros(1899,1);
    end
    v_messagesForThisChunk(m_rawData(s_it,1))=v_messagesForThisChunk(m_rawData(s_it,1))+1;
    v_messagesForThisChunk(m_rawData(s_it,2))=v_messagesForThisChunk(m_rawData(s_it,2))+1;
    %m_adjacency(m_rawData(s_it,1),m_rawData(s_it,2))=m_adjacency(m_rawData(s_it,1),m_rawData(s_it,2))+1;
    s_chuncksForMessages=s_chuncksForMessages-1;
end
v_indSubstantial=find(v_indboolSubstantial);
v_indToKeep=intersect(v_indSubstantial,v_indComponent);
m_adjacency=m_adjacency(v_indToKeep,v_indToKeep);
m_messagesTimeSeries=m_messagesTimeSeries(v_indToKeep,:);
g=Graph('m_adjacency',m_adjacency);
v_indComponent=g.getComponents{1};
m_adjacency=m_adjacency(v_indComponent,v_indComponent);
m_messagesTimeSeries=m_messagesTimeSeries(v_indComponent,:);

g=Graph('m_adjacency',m_adjacency);

end
