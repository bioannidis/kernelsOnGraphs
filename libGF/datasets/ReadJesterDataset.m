
classdef ReadJesterDataset < ReadDataset

	methods(Static)
		
		
		function [ m_table ] = getTestTables			
			% Obtain 5 disjoint sets  with 20.000 ratings  to reproduce the
			% simulations in [narang2013structured]
			%
			% m_test        : 1412 x 100  x  5 array. m_test(:,:,i) contains
			%               the  i-th test set. It has one row per user and
			%               one column per joke. 
		    
			%function m_reducedRatings=prepareJokerdat
			ch_folderName = './libGF/datasets/JesterDataset/';
			
			s_SetNum = 5;
			s_UsersNum = 1412; 
			s_JokesNum = 100;			
			m_table=zeros(s_UsersNum,s_JokesNum,s_SetNum);
			
			
			ch_fileName=[ch_folderName 'jester-data-1.xls'];
			m_ratings = xlsread(ch_fileName); % value 99 means no rating
			m_ratings = m_ratings(:,2:end); % remove the first column, which contains total number of ratings for each user			
			
			% NaN and range [0,1]
			
			% Select 1412 users (those with more ratings?) --> subtable
			
			% Select randomly a subset of 100.000 ratings 
			
			
			% Create 5 disjoint sets (slabs of a table)
			
			
			
		end
		
	end

end