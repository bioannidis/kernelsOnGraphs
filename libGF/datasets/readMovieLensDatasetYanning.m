function [R_train,R_test,c_Um,c_Up,c_rUm,c_S,c_y]=readMovieLensDatasetYanning

N=5:20;
load movielensyanning/A
% A=importdata('ratings.dat');
[u_list,idu1,idu2]= unique(A(:,1));
Nu=length(u_list);
[m_list,idm1,idm2]=unique(A(:,3));
Nm=length(m_list);
R=zeros(Nu,Nm);
ratings =A(:,5);
Nr=length(ratings);
M=R;

for i=1:Nr
    R(idu2(i),idm2(i))=ratings(i);
end
M(find(R))=1;
M1=binornd(M,0.03);
R_test=R.*M1;
R_train=R-R_test;
n_prob=0;
for s_indexNu=1:Nu
   c_Um{s_indexNu}=find(R_train(s_indexNu,:)==0); 
   c_Up{s_indexNu}=find(R_test(s_indexNu,:)==5);
   n_prob=n_prob+length(c_Up{s_indexNu});
   v_ratedIndex=find(R_train(s_indexNu,:)~=0); 
   c_rUm{s_indexNu}=v_ratedIndex;
   m_S=zeros(size(v_ratedIndex,2),Nm);
   for s_ratIndexNm=1:size(v_ratedIndex,2)
        m_S(s_ratIndexNm,v_ratedIndex(s_ratIndexNm))=1;
   end
   c_S{s_indexNu}=m_S;
   c_y{s_indexNu}=R_train(s_indexNu,v_ratedIndex);
end

end
