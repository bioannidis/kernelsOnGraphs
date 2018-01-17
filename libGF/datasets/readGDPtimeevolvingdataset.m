function [m_adj,m_testgdp] = readGDPtimeevolvingdataset
%This function returns the GDP signals

%% initialize
pth = which('readGDPtimeevolvingdataset');
folder=[pth(1:end-(length('readGDPtimeevolvingdataset')+2))  'GDPtimeevolvingdataset\'];

file=strcat(folder,'GDPedited.xls');

	m_gdp = xlsread(file);
    %v_meangdp=mean(m_gdp,2);
    %m_gdp=m_gdp-repmat(v_meangdp,1,size(m_gdp,2));
    m_traingdp=m_gdp(:,1:25);
    m_adj=m_traingdp*m_traingdp';
    m_adj=m_adj-diag(diag(m_adj));
    m_adj=m_adj/max(max(m_adj));
    m_adj(m_adj<10^(-2))=0;
    graph=Graph('m_adjacency',m_adj);
    m_testgdp=m_gdp(:,26:end);
end
