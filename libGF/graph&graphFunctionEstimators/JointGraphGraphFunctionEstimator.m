classdef JointGraphGraphFunctionEstimator < Parameter
	% This class is cool
	properties(Constant)
	end
	
	properties
		s_regularizationParameter
		s_numFoldValidation  = 10;
	end
		
	methods
		
		function obj = JointGraphGraphFunctionEstimator(varargin)
			obj@Parameter(varargin{:});
        end
		
   end
	
		
	methods(Abstract)
				
		estimate = estimate(obj);		
	end
	
end

