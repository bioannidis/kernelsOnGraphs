classdef GraphLearningInverseCovariance < GraphGenerator
	properties % required by parent classes
		c_parsToPrint  = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	
	properties(Constant)
		ch_name = 'Graph-Learning-For-Inverse-Covariance';
	end
	
	properties
		m_adjacency; % contains the assumed adjacency
		m_training_data; %contais the training data for which the covariance will be estimated
		
		
	end
	methods
		function obj = GraphLearningInverseCovariance(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
		end
		function [graph] = realization(obj)
			%initialization
			m_adj=obj.m_adjacency;
			m_train=obj.m_training_data;

			%sparsity = sum(m_adj(:))/(numel(m_adj)-size(m_adj,1))

			
			% data normalization
			v_mean = mean(m_train,2);
			v_std = std(m_train')';
			m_normalized_training_data = diag(1./v_std)*(m_train - v_mean*ones(1,size(m_train,2)));
			%m_normalized_test_data = diag(1./v_std)*(m_test_data - v_mean*ones(1,size(m_test_data,2)));
			
			% covariance of normalized data
			m_covInv = GraphLearningInverseCovariance.learnInverseCov( cov(m_normalized_training_data') , m_adj );
		    
			% estimate constrained Laplacian with zeros in the initial
			% positions dictated by the m_adj
			m_constrainedLaplacian = GraphLearningInverseCovariance.approximateWithLaplacian(m_covInv,m_adj);

			
			m_adj=Graph.createAdjacencyFromLaplacian(m_constrainedLaplacian);
			graph = Graph('m_adjacency',m_adj);
		end
	end
	methods (Static)
		function m_covInv = learnInverseCov( m_sampleCov , m_adjacency )
			% Learns the inverse covariance of a normal distribution
			% m_adjacency is optional. m_covInv is such
			% that m_covInv(i,j) = 0 if m_covInv(i,j) = 0  (i~=j)If given, then
			%
			
			d = size(m_sampleCov,1);
			m_adjacency = m_adjacency + triu(ones(d));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			variable S(d,d) symmetric
			minimize( -log_det(S) +trace(S*m_sampleCov) )
			subject to
			S(m_mask) == 0;
			cvx_end
			
			m_covInv = S;
			
		end
		
		function m_laplacian = approximateWithLaplacian(m_input,m_adjacency)
			% m_laplacian is the best Laplacian matrix approximating matrix
			% m_input in the Frobenius norm
			% m_adjacency is an optional parameter. m_laplacian is such
			% that m_laplacian(i,j) = 0 if m_adjacency(i,j) = 0  (i~=j)
			%
			s_nodeNum = size(m_input,1);
			if nargin<2
				m_adjacency = ones(s_nodeNum);
			end
			m_adjacency = m_adjacency + triu(ones(s_nodeNum));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			variable L(s_nodeNum,s_nodeNum) symmetric
			minimize( norm(L - m_input,'fro') )
			subject to
			L*ones(s_nodeNum,1) == zeros(s_nodeNum,1);
			triu(L,1) <= 0;
			L(m_mask) == 0;
			cvx_end
			
			m_laplacian = L;
		end
	end
end
