classdef KernelGraphFunctionEstimator < GraphFunctionEstimator
	% This class is cool
	properties(Constant)
	end
	
	properties
		m_kernel   % N x N kernel matrix
		           %          N: number of vertices	
		h_kernelMat = [];   % handle to a function that is given a 
		    % graph and returns a kernel matrix. Used to update the
			% kernel matrix when the graph changes.
		%s_useNormalizedLaplacian = 0;
		
	end
		
	methods
		
        function obj = KernelGraphFunctionEstimator(varargin)  % constructor
            obj@GraphFunctionEstimator(varargin{:});
        end
		
		
		function obj = prepareForGraph(obj,graph)
			% updates the kernel matrix 
			assert(~isempty(obj.h_kernelMat));
			obj.m_kernel = obj.h_kernelMat(graph);
		end
		
		
		
	end
	
	
end

