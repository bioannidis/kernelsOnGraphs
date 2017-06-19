
classdef ReadMovieLensDataset < ReadDataset
	
	
	properties(Constant)
		ch_folderName = './libGF/datasets/MovieLensDataset/ml-100k/';
		ch_cacheFileName = [ReadMovieLensDataset.ch_folderName 'CVCache.mat'];
	end
	
	
	
	methods(Static)
		
		function [ m_test, v_range ] = getGivenPartition			
			% Obtain 5 disjoint sets  with 20.000 ratings  to reproduce the
			% simulations in [narang2013structured]
			%
			%
			% m_test        : 1682 x 943 x 5 array. m_test(:,:,i) contains
			%               the  i-th test set. It has one row per movie and
			%               one column per user. 
			%               The i-th slab is obtained from ui.data
			%               Missing entries are marked with NaN
			%
			% v_range       : 2x1 vector contains the min and max rating 
			%               available in the dataset
			%
				
			s_SetNum = 5;
			s_UsersNum = 943; 
			s_MoviesNum = 1682;			
			m_test=zeros(s_MoviesNum,s_UsersNum,s_SetNum);
			v_range=zeros(2,1);
			
			for s_setInd = 1:s_SetNum
				
				ch_file = sprintf('%s%d%s',[ReadMovieLensDataset.ch_folderName 'u'],s_setInd,'_test.txt');
				
				D = readtable(ch_file,'Delimiter','\t','ReadVariableNames',false);
				TestData=table2array(D); % columns contain user id | item id | rating | timestamp.
				%                          The time stamps are unix seconds since 1/1/1970 UTC
				
				for icount=1:size(TestData,1)
					m_test(TestData(icount,2),TestData(icount,1),s_setInd)=TestData(icount,3);
				end
			
			end		
			%m_test=(m_test-1)/4;
			m_test(m_test==0)=NaN;
			v_range(1)=min(min(min(m_test)));
			v_range(2)=max(max(max(m_test)));
			% NaN and range [0,1]
			
			%save([ReadMovieLensDataset.ch_folderName 'm_test.mat'],'m_test','v_range');
		end
		
		
		
		
		function [v_CVSets,v_range] = getCVSets()
			% v_CVSets is a 5 x 1 vector of structs  with fields
			%     -v_CVSets(i).m_training
			%     -v_CVSets(i).m_validation
			%    Both these two fields are userNum x P matrices, where the (i,j)
			%    element contains the rating of user j to movie i. Misses
			%    (unknown entries) are marked with NaN.
			%    This struct is suited for input to
			%    RecommenderSystemsSimulator.simulateDataset
			% v_range:  2-dimensional vector with the minimum and the
			%    maximum  rating
					
			s_cacheFile = 1; % 0 to disable caching
			
			if s_cacheFile				
				if (exist(ReadMovieLensDataset.ch_cacheFileName,'file')) % file exists
					disp(['Loading cache data from ' ReadMovieLensDataset.ch_cacheFileName]);
					load(ReadMovieLensDataset.ch_cacheFileName);
					return
				end
			end
			
			
			% obtain data set
			[t_T,v_range] = ReadMovieLensDataset.getGivenPartition();  % t_T is a s_userNum x s_itemNum x s_foldCrossvalidationNum			
			s_foldCrossvalidationNum = size(t_T,3);
								
			%for s_CVInd = 1:s_foldCrossvalidationNum
			for s_CVInd = s_foldCrossvalidationNum:-1:1
				v_CVSets(s_CVInd,1).m_training = ReadDataset.mergeDataMatrices(t_T(:,:,[1:s_CVInd-1 s_CVInd+1:s_foldCrossvalidationNum] ));
				v_CVSets(s_CVInd,1).m_validation = t_T(:,:,s_CVInd);
			end
			
			if s_cacheFile			
				disp(['Saving cache data to ' ReadMovieLensDataset.ch_cacheFileName]);
				save(ReadMovieLensDataset.ch_cacheFileName,'v_CVSets','v_range');
			end
			
		end
		
		
		
		
		
		
	end
end
