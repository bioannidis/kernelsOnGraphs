classdef BandlimitedGraphFunctionEstimator < GraphFunctionEstimator
	% This is a comment
	properties(Constant)
	end
	
	properties % Required by superclass Parameter
		c_parsToPrint    = {'ch_name','s_bandwidth'};
		c_stringToPrint  = {'',''};
		c_patternToPrint = {'%s%s','%s%s'};
	end
	
	properties
		ch_name = 'Bandlimited';
		m_laplacian;               % Laplacian matrix
		
		s_bandwidth; %    Number of the first eigenvectors of m_laplacian 
		% that span the signal subspace. If s_bandwidth ==-1, then
		% s_bandwidth is set to the "cut-off" bandwidth from
		% [narang2013structured] using a combinatorial Laplacian
		% [anis2016proxies], which is the max bandwidth that allows 
		% identifiability of the bandlimited signal given the sampling set
		% and Laplacian matrix.
		%
		
		proxy_order = 5; % parameter k in [anis2016proxies] used to compute 
		% the cuttoff frequency. 
		
		
		% backwards compatibility
		m_laplacianEigenvectors;   % N x P matrix with the P first 
		% eigenvectors of the Laplacian Matrix (smallest eigenvalues).
		% Although this property could be obtained from m_laplacian, it is
		% required for efficiency.
				
	end
	
	properties(Access = private) % precomputed properties for efficiency
		v_laplacianEigenvalues_precomp % sorted in increasing order
		m_laplacianEigenvectors_precomp
	end
	
	methods
		
		function obj = BandlimitedGraphFunctionEstimator(varargin)
			obj@GraphFunctionEstimator(varargin{:});
		end
		
		function obj = set.m_laplacian(obj,L)
			obj.m_laplacian = L;
			
			[V,D] = eig(L);%,size(L,1));
			obj.v_laplacianEigenvalues_precomp = diag(D); 
			if ( norm(imag(diag(D)))/norm(real(diag(D)) ) > .01)
				warning('Eigenvalues may be complex --> check your Laplacian matrix');
			end
			obj.m_laplacianEigenvectors_precomp = V; 				
		end
		
		function obj = set.m_laplacianEigenvectors(obj,L)
			error('behavior of BandlimitedGraphFunctionEstimator changed: please assign properties m_laplacian and s_bandwidth instead of m_laplacianEigenvectors');
			
		end
		
		function str = s_bandwidth_print(obj)
			assert(~isempty(obj.s_bandwidth));
			if obj.s_bandwidth == -1
				str = 'Ass. B = cut-off freq.';
			else
				str = sprintf('Ass. B = %d',obj.s_bandwidth);
			end
			
        end
        
        function N = getNumOfVertices(obj)
            N = size(obj.m_laplacian,1);
        end
		
	end
	
	methods
		
		function m_estimate = estimate(obj,m_samples,m_positions)
			%
			% Input:
			%
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
			
			if isempty(obj.m_laplacian)~=isempty(obj.s_bandwidth)
				error('Unassigned properties m_laplacian and s_bandwidth');
			end
			if isempty(obj.s_bandwidth)
				error('unassigned value of s_bandwidth. Set to -1 to use the cut-off freq. from [narang2013structured]');
			end
			if obj.s_bandwidth ~= -1
				assert(obj.s_bandwidth<=size(obj.m_laplacian,1),'Too large bandwidth')
				m_eigenvecs = obj.m_laplacianEigenvectors_precomp(:,1:obj.s_bandwidth);
			end
						
			s_numberOfVertices = size(obj.m_laplacian,1);
			s_numberOfRealizations = size(m_samples,2);						
			m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
			
			
			for iRealization = 1:s_numberOfRealizations
				
				if obj.s_bandwidth == -1				
					s_bw = obj.computeCutoffFrequency(m_positions(:,iRealization));
					m_eigenvecs = obj.m_laplacianEigenvectors_precomp(:,1:s_bw);
				end
				v_alphas=zeros(obj.s_bandwidth,1);
				m_PhiB = m_eigenvecs( m_positions(:,iRealization) , : );
                if cond(m_PhiB)>1e6 % for cases where LS returns a very bad 
					%                 estimate, the mean is returned
					%                 instead. This improves considerably
					%                 the performance of LS.                     
					v_alphas = mean(m_samples(:,iRealization))*[1; zeros(size(m_PhiB,2),1)]/mean(m_eigenvecs(:,1));
                    v_alphas=v_alphas(1:obj.s_bandwidth);
                else
                    v_alphas = m_PhiB\m_samples(:,iRealization);
                end
				m_estimate(:,iRealization) = m_eigenvecs*v_alphas;
			end
			
			
		end
		
		
		function s_bw = computeCutoffFrequency(obj, m_positions)
			
			
			L = obj.m_laplacian;
			L_power = L^(2*obj.proxy_order);
			
			% svds(L_power(m_positions,m_positions),1,0) does not always work, even
			% after setting a high tolerance -> we use
			v_svals = svd(L_power(m_positions,m_positions));
			min_sval = min(v_svals);
			omega_S = (  min_sval  )^(1/(2*obj.proxy_order));
						
			s_bw = sum(obj.v_laplacianEigenvalues_precomp <= omega_S);
			
		end
		
	end

end
