classdef IPRGraphFunctionEstimator < GraphFunctionEstimator
	% Implementation of the IPR algorithm from [wang2015local] -->
	% INCOMPLETE
	
	% This is a comment
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {'ch_name','s_bandwidth'};
		c_stringToPrint  = {'','ASS. BW'};
		c_patternToPrint = {'%s%s','%s = %d'};
	end
	
	properties
		ch_name = 'IPR';
		m_laplacian;          % N x N Laplacian matrix
		s_bandwidth = [];     %    number of the first S_BANDWIDTH columns 
		                      %    of m_laplacian that span the signal
		                      %    subspace
	end
	
	methods
		
		function obj = IPRGraphFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		
	end
	
	methods
		
		function m_estimate = estimate(obj,m_samples,m_positions)
			%
			% Input:
			% M_SAMPLES                 S x S_NUMBEROFREALIZATIONS  matrix with
			%                           samples of the graph function in
			%                           M_GRAPHFUNCTION
			% M_POSITIONS               S x S_NUMBEROFREALIZATIONS matrix
			%                           containing the indices of the vertices
			%                           where the samples were taken
			%
			% Output:                   
			% M_ESTIMATE                N x S_NUMBEROFREALIZATIONS matrix. N is
			%                           the number of nodes and each column
			%                           contains the estimate of the graph
			%                           function
			%
			
			if isempty(obj.s_bandwidth)
				error('undefined bandwidth s_bandwidth');
			end	
			if( obj.s_bandwidth > size(obj.m_laplacian,2) )
				error('s_bandwidth cannot be greater than the number of vertices');
			end
			[m_V,~] = eig(obj.m_laplacian);
			m_V = m_V(:,1:obj.s_bandwidth);
						
			s_numberOfVertices = size(m_V,1);
			s_numberOfRealizations = size(m_samples,2);
			Proj = m_V*m_V';
						
			m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
			for iRealization = 1:s_numberOfRealizations
				%m_PhiB = m_V( m_positions(:,iRealization) , : );
				%v_alphas = m_PhiB\m_samples(:,iRealization);
				%m_estimate(:,iRealization) = m_V*v_alphas;
				m_estimate(:,iRealization) = IPRGraphFunctionEstimator.iprEstimate(m_samples(:,iRealization),m_positions(:,iRealization),Proj,obj.m_laplacian);
			end
				
        end
        
        function N = getNumOfVertices(obj)
            N = size(obj.m_laplacian,1);
        end
		
	end
	
	methods(Static,Access = private)
		
		
		function v_estimate = iprEstimate(v_samples,v_positions,Proj,m_laplacian)
			
			deltas = (m_laplacian(:,v_positions)~=0);
			
			error('not implemented --> local set construction is required');
			v_estimate = randn(size(Proj,1),1);
		
		end
		
	end

end
