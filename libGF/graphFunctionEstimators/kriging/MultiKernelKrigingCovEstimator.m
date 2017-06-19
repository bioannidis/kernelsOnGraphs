classdef MultiKernelKrigingCovEstimator < Parameter
	% 
	properties(Constant)
	end
	properties(Abstract)
    end
	properties
            t_kernelDictionary; % M x N x N tensor containing M kernel 
                                %functions evalueated at each pair of nodes
    end
    
		
	methods
		
		function obj = MultiKernelKrigingCovEstimator(varargin)
			obj@Parameter(varargin{:});
		end
		
	end
	
% 	methods(Abstract)
% 		
% 	end
	
end

