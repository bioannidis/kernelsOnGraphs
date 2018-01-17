function [m_adjacency,m_delayTimeSeries] = readNetworkDelayDataset
%%
%This function returns the temperature data per hour
%The subscript n and o denotes the old and new measurments

%% initialize
pth = which('readNetworkDelayDataset');
folder=[pth(1:end-(length('readNetworkDelayDataset')+2))  'NetworkDelayDataset\'];
file=strcat(folder,'latmat.mat');

load(file);
numpath = size(G,1);
numnds = size(G,2);
lats = L(:,ap)';
m_adjacency = zeros(numpath);
deg = sum(G')';
for i = 1:numpath-1
    for j = i+1:numpath
        m_adjacency(i,j) = (G(i,:)*G(j,:)')/(deg(i) + deg(j) - G(i,:)*G(j,:)');
    end
end
m_adjacency=m_adjacency+m_adjacency';
m_adjacency(:,53)=[];
m_adjacency(53,:)=[];
m_adjacency(:,38)=[];
m_adjacency(38,:)=[];
g=Graph('m_adjacency',m_adjacency);
lats(53,:)=[];
lats(38,:)=[];
m_delayTimeSeries=lats;
end
