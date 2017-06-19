function [m_test,m_adjacencyItem,m_adjacencyUser,m_ratings,m_userFeatures,m_moviesFeatures] = readRSMovielensDataset
%%
%This function
%%
pth = which('readRSMovielensDataset');
folder=[pth(1:end-(length('readRSMovielensDataset')+2))  'RSMovielensDataset\'];
load(strcat(folder,'ML1M_TopN.mat'));
m_moviesFeatures=full(C_NCD);
m_userFeatures=full(D_NCD);
m_ratings=TrainSet;
m_reducedRatings=m_ratings;
s_iterItems=1;
while (s_iterItems<=size(m_reducedRatings,1))
    if(m_reducedRatings(s_iterItems,:)==zeros(1,size(m_reducedRatings,2)))
        m_reducedRatings(s_iterItems,:)=[];
        m_moviesFeatures(s_iterItems,:)=[];
    end
    s_iterItems=s_iterItems+1;
end

s_iterUsers=1;
while (s_iterUsers<=size(m_reducedRatings,2))
    if(m_reducedRatings(:,s_iterUsers)==zeros(size(m_reducedRatings,1),1))
        m_reducedRatings(:,s_iterUsers)=[];
    end
    s_iterUsers=s_iterUsers+1;
end

m_adjacencyItem=m_reducedRatings*m_reducedRatings';

% Scale entries by inverse of diagonal?
m_adjacencyItem=m_adjacencyItem-diag(diag(m_adjacencyItem));    
m_adjacencyUser=m_reducedRatings'*m_reducedRatings;

% Scale entries by inverse of diagonal?
m_adjacencyUser=m_adjacencyUser-diag(diag(m_adjacencyUser)); 
m_test=[];
m_ratings=m_reducedRatings;


end
