classdef GraphFunctionEstimator < Parameter
	% This class is cool
	properties(Constant)
	end
	
	properties
		s_regularizationParameter
		s_numFoldValidation  = 10;
	end
		
	methods
		
		function obj = GraphFunctionEstimator(varargin)
			obj@Parameter(varargin{:});
		end
		
		
		function [s_optMu,s_ind] = crossValidation(obj,v_samples,v_positions,v_mu)
			% Input:
			% V_SAMPLES                 S x S_NUMBEROFREALIZATIONS  matrix with
			%                           samples of the graph function in
			%                           M_GRAPHFUNCTION
			% V_POSITIONS               an S x S_NUMBEROFREALIZATIONS
			%                           matrix containing the indices of
			%                           the vertices where the samples were
			%                           taken  
			
			assert(size(v_samples,2)==1,'not implemented');
			
            %N = obj.getNumOfVertices();
            S = length(v_samples);
			m_mse = NaN(length(v_mu), obj.s_numFoldValidation);
            cvIndex = crossvalind('Kfold', S, obj.s_numFoldValidation);    % 5-fold cross validation
			for muInd = 1:length(v_mu)
				
				% partition v_positions in obj.s_numFoldValidation subsets
				% m_cvPositions =   % N0 x obj.s_numFoldValidation matrix
				
                for valInd = 1:obj.s_numFoldValidation
					% Create test and validation set
					v_validationIndex = (cvIndex == valInd);
                    v_trainingIndex = ~v_validationIndex;
                    
                    m_samples_training = v_samples(v_trainingIndex);
                    v_positions_training = v_positions(v_trainingIndex);
                    
                    v_samples_validation = v_samples(v_validationIndex);
                    v_positions_validation = v_positions(v_validationIndex);
					
					% Estimate
					obj.s_regularizationParameter = v_mu(muInd);
					v_signal_est = obj.estimate(m_samples_training, v_positions_training);
					
					% Measure MSE
% 					m_mse(muInd,valInd) = norm( v_samples - ...
%                         v_signal_est(v_positions) )^2 / ...
%                         norm( v_samples )^2;
					m_mse(muInd,valInd) = norm( v_samples_validation - ...
                        v_signal_est(v_positions_validation) )^2 / ...
                        norm( v_samples_validation )^2;
					
                end
			end
			v_mse = mean(m_mse,2);
			[~,s_ind] = min(v_mse);
			s_optMu = v_mu(s_ind);
        end	
	end
	
	methods(Abstract)
				
		estimate = estimate(obj,m_samples,sideInfo);			
		%
		% Input:
		% M_SAMPLES                 
		%                           S x S_NUMBEROFREALIZATIONS  matrix with
		%                           samples of the graph function in
		%                           M_GRAPHFUNCTION 
		%                           
		% sideInfo                  It can be either:
		%      a) an S x S_NUMBEROFREALIZATIONS matrix containing the
		%      indices of the vertices where the samples were taken
		%      b) a 1 x S_NUMBEROFREALIZATIONS vector of structs with fields
		%         sideInfo(i).v_sampledEntries:  S x 1
		%                           vector where each column contains the
		%                           indices of the sampled vertices
		%         sideInfo(i).v_wantedEntries:  W x 1
		%                           vector where each column contains the
		%                           indices of the desired vertices. If not
		%                           defined, it is assumed that this field
		%                           is 1:N.
		%
		% Output:                   
		% estimate                  If sideInfo is not a struct, then it is
		%      an N x S_NUMBEROFREALIZATIONS matrix. N is the number of
		%      nodes and each column contains the estimate of the graph
		%      function 
		%                           If sideInfo IS a struct, then it is            
		%      a 1 x S_NUMBEROFREALIZATIONS vector of structs with
		%      fields
		%         estimate(i).v_wantedSamples: W x 1 vector containing the
		%                            estimated signal at the entries
		%                            indicated by
		%                            sideInfo(i).v_wantedEntries 
		%          
		% 
		
		N = getNumOfVertices(obj);
        % return the number of vertices of the graph that this estimator is 
        %        trying to operate on. Since this base class has no idea what 
        %        the graphs are likely to be estimated by descendants class
        %        it is implemented as an abstract class
		
	end
	
end

