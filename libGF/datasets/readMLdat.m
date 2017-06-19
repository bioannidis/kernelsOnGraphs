
function [m_train,m_test,c_userInfo,c_movieInfo] = readMLdat

% m_train,m_test are 943X1682 matrices contain as training and test data
%               user ratings, (943 users and 1682 movies)
% c_userInfo cell array containing
%   -- Demographic information about the users; this is a tab
%               separated list of
%               user id | age | gender | occupation | zip code
%
% c_movieInfo cell array containing
% -- Information about the items (movies); this is a tab separated
%               list of
%               movie id | movie title | release date | video release date |
%               IMDb URL | unknown | Action | Adventure | Animation |
%               Children's | Comedy | Crime | Documentary | Drama | Fantasy |
%               Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
%               Thriller | War | Western |
%               The last 19 fields are the genres, a 1 indicates the movie
%               is of that genre, a 0 indicates it is not; movies can be in
%               several genres at once.
%               The movie ids are the ones used in the u.data data set.
%for more information go to the readme file in the MovieLens folder



pth=which('readMLdat');
path_left=pth(1:end-(length('readMLdat')+2));
file=strcat(path_left,'MovieLensDataset\ml-100k\u1');
% u.data     -- The full u data set, 100 000 ratings by 943 users on 1682 items.
%               Each user has rated at least 20 movies.  Users and items are
%               numbered consecutively from 1.  The data is randomly
%               ordered. This is a tab separated list of
% 	         user id | item id | rating | timestamp.
%               The time stamps are unix seconds since 1/1/1970 UTC
D = readtable(file,'Delimiter','\t','ReadVariableNames',false);
TrainData=table2array(D);

file=strcat(path_left,'MovieLensDataset\ml-100k\u1_test');
D = readtable(file,'Delimiter','\t','ReadVariableNames',false);
TestData=table2array(D);

%  -- Demographic information about the users; this is a tab
%               separated list of
%               user id | age | gender | occupation | zip code
%               The user ids are the ones used in the u.data data set.
file=strcat(path_left,'MovieLensDataset\ml-100k\user');
D = readtable(file,'Delimiter','|','ReadVariableNames',false);
c_userInfo=table2cell(D);

file=strcat(path_left,'MovieLensDataset\ml-100k\u.item.txt');
D = readtable(file,'Delimiter','|','ReadVariableNames',false);
c_movieInfo=table2cell(D);


%I rearrange the train dataset and the the test so that it is in a
%user- items table
m_train=zeros(size(c_userInfo,1),size(c_movieInfo,1));
for(i=1:size(TrainData,1))
	m_train(TrainData(i,1),TrainData(i,2))=TrainData(i,3);
end

m_test=zeros(size(c_userInfo,1),size(c_movieInfo,1));
for(i=1:size(TestData,1))
	m_test(TestData(i,1),TestData(i,2))=TestData(i,3);
end

end

