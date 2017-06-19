classdef NonParametricBatchGraphFunctionEstimator< GraphFunctionEstimator
	% This was written by Vassilis
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	%test commi
	properties
		ch_name = 'NONPARAMETRIC';
		m_kernels;   % Nt x Nt matrix containing the kernel function evalueated at each pair of nodes
		s_mu;
		s_maximumTime;
	end
	
	methods
		
		function obj = NonParametricBatchGraphFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		function N = getNumOfVertices(obj)
            N = size(obj.m_kernels,1)/obj.s_maximumTime;
        end
		
	end
	
	methods
		function m_estimate = estimate(obj,m_samples,m_positions,v_numberOfSamples)
			%
			% Input:
			% M_SAMPLES                 T*Stx T*St_NUMBEROFREALIZATIONS  matrix with
			%                           samples of the graph function in
			%                           M_GRAPHFUNCTION
			% M_POSITIONS               S x S_NUMBEROFREALIZATIONS matrix
			%                           containing the indices of the vertices
			%                           where the samples were taken
			%
			% Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
			%                           the number of nodes and each column
			%                           contains the estimate of the graph
			%                           function
			%
			s_mu=obj.s_mu;
			s_numberOfVertices = size(obj.m_kernels,1)/obj.s_maximumTime;
			s_numberOfRealizations = size(m_samples,2);
			if (size(v_numberOfSamples,1)==1) %not supported different sampling sizes yet
				m_estimate = zeros(s_numberOfVertices*obj.s_maximumTime,s_numberOfRealizations);
				for realizationCounter = 1:s_numberOfRealizations
					m_superPhi=zeros(size(m_positions,1),s_numberOfVertices*obj.s_maximumTime);
					for s_time=1:obj.s_maximumTime
						for s_ind=1:v_numberOfSamples
							m_superPhi((s_time-1)*v_numberOfSamples+s_ind...
								,(s_time-1)*s_numberOfVertices+m_positions(...
								(s_time-1)*v_numberOfSamples+s_ind,realizationCounter))=1;
						end
					end % WRITE MORE EFFICIENT 
					m_estimate(:,realizationCounter) = (m_superPhi'*m_superPhi+...
						v_numberOfSamples*obj.s_maximumTime... this must change to a sum in a general v_numberOfSamples
						*s_mu*obj.m_kernels^-1)\m_superPhi'*m_samples(:,realizationCounter);
				end
				
				
			end
		end
		
	end
	
end
