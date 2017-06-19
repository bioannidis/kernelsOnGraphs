classdef GraphKernel < Parameter
	% 
	properties(Constant)
	end
	properties(Abstract)
    end
	properties
            m_kernels; % N x N matrix containing the kernel function evalueated at each pair of nodes
    end
    
		
	methods
		
		function obj = GraphKernel(varargin)
			obj@Parameter(varargin{:});
		end
		
	end
	
% 	methods(Abstract)
% 		
% 	end
	
end

