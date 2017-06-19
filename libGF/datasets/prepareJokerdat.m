
function m_reducedRatings=prepareJokerdat
pth=which('prepareJokerdat');
path_left=pth(1:end-(length('prepareJokerdat')+2));
file=strcat(path_left,'JesterDataset\jester-data-1.xls');
m_ratings = xlsread(file);
m_reducedRatings=m_ratings(randperm(size(m_ratings,1),500),(2:end));
end