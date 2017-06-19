classdef LaplacianKernel < Parameter
    % KernelGenerator
    %       Class to generator graph kernel matrices
    %   
    
    properties
        c_parsToPrint = {'ch_type','s_sigma'};
		c_stringToPrint = {'','\sigma'};
		c_patternToPrint = {'%s kernel','%s = %f'};
    end
    
    properties
        h_r_inv         % cell array of r^{-1}(\lambda) in graph kernel
        m_laplacian     % laplacian matrix
                       
    end
    
    methods
        function obj = LaplacianKernel(varargin)
            obj@Parameter(varargin{:});
        end
        
        function t_kernelMatrix = getKernelMatrix(obj)
            % t_kernelMatrix is an N x N x P tensor where
            %        N = size(obj.m_laplacian,1)
            %        P = length(obj.h_r_inv)
            %
            
            if ~iscell(obj.h_r_inv)
                error('h_r_inv must be a cell array');
            end
            
            if isempty(obj.m_laplacian)
                error('Property m_laplacian cannot be empty');
			end

		    [V,D] = eig(obj.m_laplacian);
            d = diag(D);  
            d(1) = 0;                  % fix a bug since the first
                                            % eigenvalue of L is always 0
            N = size(obj.m_laplacian,1);
            P = length(obj.h_r_inv);
            t_kernelMatrix = NaN(N,N,P);
            for p = 1 : P
                r = obj.h_r_inv{p};
                Kp = V * diag(r(d)) * V.';
                t_kernelMatrix(:,:,p) = ( Kp + Kp' )/2;
            end
		end
		
		function m_kernel = getNewKernelMatrix(obj,graph)
obj.m_laplacian = graph.getLaplacian();
%obj.m_laplacian = graph.getNormalizedLaplacian();			
			m_kernel = obj.getKernelMatrix;
		end
		
		function plotEigenvalueFunctions(obj)
			% represents r vs \lambda
			
			if ~iscell(obj.h_r_inv)
				error('h_r_inv must be a cell array');
			end
			
			if isempty(obj.m_laplacian)
				error('Property m_laplacian cannot be empty');
			end
			
			[V,D] = eig(obj.m_laplacian);
			d = diag(D)';  d(1) = 0;                  % fix a bug since the first
			% eigenvalue of L is always 0
			N = size(obj.m_laplacian,1);
			P = length(obj.h_r_inv);
			m_rinvcurves = NaN(P,N);
			for p = 1 : P
				r = obj.h_r_inv{p};
				m_rinvcurves(p,:) = r(d);
			end
			plot(d',m_rinvcurves','.-');
			
		end
		
		
    end
    
    % type specific methods
    methods(Static)
        % diffusion kernel: r(lambda) = exp(sigma^2*lambda/2)
        function h_r_inv = diffusionKernelFunctionHandle(sigmaArray)
                       
            for iSigma = length(sigmaArray):-1:1
                sigma = sigmaArray(iSigma);
                h_r_inv{iSigma} = @(lambda) exp( - sigma^2 * lambda / 2 );
            end
            
        end
        
         % regularized kernel: r(lambda) = 1 + sigma^2*lambda
        function h_r_inv = regularizedKernelFunctionHandle( sigmaArray)
            
            for iSigma = length(sigmaArray):-1:1
                sigma = sigmaArray(iSigma);
                h_r_inv{iSigma} = @(lambda) 1./(1 + sigma^2 * lambda);
            end
            
		end
		
		% bandlimited kernel
		function h_r_inv = bandlimitedKernelFunctionHandle( m_laplacian , v_bandwidth , s_beta)
			%   h_r_inv:  cell array of function handlers to create a
			%   bandlimited kernel:
			%                 |  1/s_beta      if     lambda <=  lambda_B
			%   r_B(lambda) = |
			%                 |  s_beta        otherwise
			% 
			%  where lambda_B is the B-th smallest eigenvalue of the Laplacian
			% v_bandwidth:     nKernels-long vector with the values of B
			
			assert(~isempty(m_laplacian));
			[~,D] = eig(m_laplacian);
			v_evals = diag(D); % eigenvalues sorted in ascending order
            v_lambda_B = v_evals(v_bandwidth);
			
            for iKernel = length(v_lambda_B):-1:1                
                h_r_inv{iKernel} = @(lambda) 1./(...
				   1/s_beta*(lambda<=v_lambda_B(iKernel)) + s_beta*(lambda>v_lambda_B(iKernel))...
				);
            end
            
        end
		
		
        
        
    end
    methods
        
       
    end
    
end
