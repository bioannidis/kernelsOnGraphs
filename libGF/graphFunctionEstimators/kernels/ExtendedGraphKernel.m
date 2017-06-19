classdef ExtendedGraphKernel < GraphKernel
	
	
	properties % required by parent classes
		c_parsToPrint  = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	
	properties(Constant)
		ch_name = 'ExtendedGraphKernel';
	end
	
	properties
		t_invSpatialKernel;  % NxNxt tensor containing the spatial kernel at each
		% time t
		t_invTemporalKernel % NxNx(t-1) tensor containing the time kernel at each
		% time t
	end
	
	methods
		
		function obj = ExtendedGraphKernel(varargin)
			% Constructor
			obj@GraphKernel(varargin{:});
			
		end
		
		function m_extendedKernel=generateKernelMatrix(obj)
			% Output:
			% NtxNt matrix containing the kernels evaluation
			
			
			assert(~isempty(obj.t_invSpatialKernel));
			assert(~isempty(obj.t_invTemporalKernel));
			t_invTemporalKernel=obj.t_invTemporalKernel;
			t_invSpatialKernel=obj.t_invSpatialKernel;
			s_maximumTime=size(t_invSpatialKernel,3);
			s_numberOfVertices=size(t_invTemporalKernel,1);
			m_invExtendedKernel=zeros(s_numberOfVertices*s_maximumTime);
			for s_ind=1:s_maximumTime
				m_invExtendedKernel((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
					(s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
					t_invSpatialKernel(:,:,s_ind);
				if s_ind<s_maximumTime
					m_invExtendedKernel((s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices),...
						(s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices))=...
					t_invTemporalKernel(:,:,s_ind);
					m_invExtendedKernel((s_ind-1)*(s_numberOfVertices)+1:(s_ind)*(s_numberOfVertices),...
						(s_ind)*(s_numberOfVertices)+1:(s_ind+1)*(s_numberOfVertices))=...
					t_invTemporalKernel(:,:,s_ind)';
								
				end
			end
			m_extendedKernel =inv(m_invExtendedKernel);
			
		end
	end
end
