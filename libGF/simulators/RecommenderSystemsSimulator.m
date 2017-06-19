classdef RecommenderSystemsSimulator < Simulator
	
	properties(Constant)
		s_smallGraphs = 1;   % If 1, then a graph containing only the 
                             % training and test vertices is constructed
	end
	
	methods(Static)
		
		function mse = simulateDataset( v_CVSets , graphConstructor, v_estimator )
			% v_CVSets:    L x 1 vector of structs with fields
			%     -v_CVSets(i).m_training
			%     -v_CVSets(i).m_validation
			%    Both these two fields are moviesNum x usersNum matrices,
			%    where the (i,j) element contains the rating of user j to
			%    movie i. Misses (unknown entries) are marked with NaN.
			%
			% graphConstructor: function that, given a matrix like
			%    v_CVSets(i).m_training, it returns an object of class graph.
			%
			% v_estimator:   E x 1 vector of objects of class GraphFunctionEstimator
			%
			% mse is computed per estimator v_estimator(k) as follows:
			%    1. v_mse(i) is computed for the i-th set, i.e., v_CVSets(i),
			%    for all i. To do this
			%          1- a graph is constructed using graphConstructor
			%          applied to v_CVSets(i).m_training
			%          2- estimator is invoked per column to estimate the
			%          non-empty (different from NaN) entries in
			%          v_CVSets(i).m_validation 
			%          3- v_mse(i) is computed over the non-empty entries of
			%          v_CVSets(i).m_validation 
			%    2. mse is the result of averaging v_mse
			%             
			%         

			

			s_movieNum = size(v_CVSets(1).m_training,1);			
			s_userNum = size(v_CVSets(1).m_training,2);			
			s_foldCVNum = length(v_CVSets);
			s_estimatorNum = length(v_estimator);
			
			m_mse = NaN(s_estimatorNum,s_foldCVNum);
			v_rowIndices = (1:s_movieNum)';
			
			load_graph = 1;
			if load_graph
				load('graph_movielens.mat','v_graph'); disp('graph loaded')
			else
				for s_foldCVInd = s_foldCVNum:-1:1
					v_graph(s_foldCVInd) = graphConstructor(v_CVSets(s_foldCVInd).m_training);
				end
				save('graph_movielens.mat','v_graph'); disp('graph saved')
			end
debug =1;			
if debug
	s_userNum = 50;
end
			

			for s_foldCVInd = 1:s_foldCVNum
				s_foldCVInd

				graph_now = v_graph(s_foldCVInd);
				
				
				% update estimators
				if ~RecommenderSystemsSimulator.s_smallGraphs
					for s_estimatorInd = s_estimatorNum:-1:1
						v_estimator(s_estimatorInd) = v_estimator(s_estimatorInd).prepareForGraph(graph_now);
					end
				end
%s_userNum
		        v_squaredError = NaN(s_estimatorNum,s_userNum) ;
				s_estimatedEntriesNum = NaN(s_estimatorNum,s_userNum);
                for s_userInd = 1:s_userNum

					%s_userInd
					
					% estimation of missing entries column by column					
					v_training = v_CVSets(s_foldCVInd).m_training(:,s_userInd);
					v_trainingEntries = v_rowIndices(~isnan(v_training));
					v_trainingSamples = v_training(v_trainingEntries);
					
					v_validation = v_CVSets(s_foldCVInd).m_validation(:,s_userInd);
					v_validationEntries = v_rowIndices(~isnan(v_validation));
					v_validationSamples = v_validation(v_validationEntries);
					
					% estimation
					sideInfo.v_sampledEntries = v_trainingEntries;
					sideInfo.v_wantedEntries = v_validationEntries;					

					for s_estimatorInd = 1:s_estimatorNum

						if RecommenderSystemsSimulator.s_smallGraphs
							m_graphAdjacency = graph_now.m_adjacency;
							v_graphIndices = [sideInfo.v_sampledEntries;sideInfo.v_wantedEntries];
							m_subgraphAdjacency = m_graphAdjacency( v_graphIndices , v_graphIndices );
							subgraph = Graph('m_adjacency',m_subgraphAdjacency);
							
							sideInfo.v_sampledEntries = (1:length(sideInfo.v_sampledEntries))';
							sideInfo.v_wantedEntries = (length(sideInfo.v_sampledEntries)+1:length(v_graphIndices))';
							v_estimator(s_estimatorInd) = v_estimator(s_estimatorInd).prepareForGraph(subgraph);							
						end

if debug
	m = mean([v_trainingSamples;v_validationSamples]);						
	v_signalEstimate = v_estimator(s_estimatorInd).estimate(v_trainingSamples-m,sideInfo);
	v_signalEstimate.v_wantedSamples  = v_signalEstimate.v_wantedSamples +m;												
else
	v_signalEstimate = v_estimator(s_estimatorInd).estimate(v_trainingSamples,sideInfo);
end
	
	

						% error computation
						v_squaredError(s_estimatorInd,s_userInd) = norm( v_validationSamples - v_signalEstimate.v_wantedSamples )^2;
						s_estimatedEntriesNum(s_estimatorInd,s_userInd) = length( v_validationEntries );
					
if 0&&debug
	s = [v_trainingSamples;v_validationSamples];
	subgraph.plotFourierTransform(s-m);
end
					
					end
					
				end
				m_mse(:,s_foldCVInd) = sum( v_squaredError , 2 )./sum( s_estimatedEntriesNum , 2 )
				%end
			end
			m_mse
			mse = mean(m_mse,2);
			
			
		end
		
		
	end
	
end

