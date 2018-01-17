A = zeros(numpath);
deg = sum(G')';
for i = 1:numpath-1
    for j = i+1:numpath
        A(i,j) = (G(i,:)*G(j,:)')/(deg(i) + deg(j) - G(i,:)*G(j,:)');
    end
end
A=A+A';
A(:,53)=[];
A(53,:)=[];
A(:,38)=[];
A(38,:)=[];
g=Graph('m_adjacency',A);
lats(53,:)=[];
lats(38,:)=[];
