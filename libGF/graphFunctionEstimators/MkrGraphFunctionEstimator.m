classdef MkrGraphFunctionEstimator < KernelGraphFunctionEstimator
    % Function estimator using multi-kernel regression method
    
    properties
% <<<<<<< HEAD
%         c_parsToPrint  = {'ch_name', 'ch_type','s_regularizationParameter','m_kernel'};
% 		c_stringToPrint  = {'', '','\mu',''};
% 		c_patternToPrint = {'%s%s','%s%s','%s = %g','%s%s'};
% =======
        c_parsToPrint  = {'ch_name', 'legendString','ch_type','s_regularizationParameter','m_kernel'};
		c_stringToPrint  = {'', '','','\mu',''};
		c_patternToPrint = {'%s%s', '%s%s','%s%s','%s = %g','%s%s'};
%>>>>>>> 1ed5d407b0079214c6e868b987daeb2d96beb25a
    end
    
    properties
		ch_name = 'Multi-kernel';
        %m_kernel   %  N x N x P tensor, where 
		           %       N: number of vertices
				   %       P: number of kernels
        %s_regularizationParameter
        s_sigma    % only valid for single kernel
		ch_type = 'RKHS superposition'; 
		           % 'RKHS superposition': ADMM algorithm following
		           % [bazerque2013basispursuit] 
				   %
				   % 'kernel superposition': IIA algorithm from
				   % [cortes2009regularization]
        singleKernelPostEstimator = [];  % if non-empty, the final estimate 
		           % is computed using the GraphFunctionEstimator
		           % singleKernelPostEstimator. 
		
		
        %b_finishSingleKernel = 0; % if 1, then a last regression step
		           % performed with the kernel that has strongest weight.
				   % Current version implements only ridge regression. 
        %s_finishRegularizationParameter =[]; % reg parameter for finishing 
		           % with ridge regression. If empty, it is set equal to
		           % s_regularizationParameter.
        b_estimateFreq = 0;   % If bandlimited kernels are used to estimate 
                   % frequencies of graph signals, then set this bit to 1. 
                   % The effect is that the "best" kernel index will be returned
                   % to determine which bandlimited kernel is best, thus getting
                   % the estimated frequency of graph signals
    end
    
    properties	% IIA related properties  
        % IIA parameters
		s_IIARadius = 1; 
		s_IIAStepSize = 0.5;
		s_IIATolerance = 1e-5;
		v_IIACenter = [];   % if empty => all zero vector
		s_IIAMaxIterations = 200;
				
				   
    end
    
    properties (Dependent)
        legendString
    end
    
    methods
        function str = get.legendString(obj)
            % return string that can be used for legend generating
            % because when replicate along m_kernel, this property cannot 
            % be printed, thus define this dependent property to accomplish
            % this task
            nKernels = size(obj.m_kernel,3);
            if nKernels == 1
                str = sprintf('1 kernel, \\sigma^2 = %3.2f', obj.s_sigma^2);
            else
                str = sprintf('%d kernels', nKernels);
            end
		end
		
%<<<<<<< HEAD
		function str = m_kernel_print(obj)
			
			str = obj.legendString;
		end
		
% 		function str = s_regularizationParameter_print(obj)
% 			if ~isempty(obj.s_regularizationParameter)
% 				str = 'Finish with single kernel';
% 			else
% 				str = '';
% 			end
	end
    
    methods
		
        function obj = MkrGraphFunctionEstimator(varargin)  % constructor
            obj@KernelGraphFunctionEstimator(varargin{:});
        end
        

        function [v_estimate, m_alpha , v_theta, main_kernel_ind] = estimate(obj, m_samples, sideInfo)
			% See header of GraphFunctionEstimator@estimate 		
			%
% =======
%         function [v_estimate, m_alpha_norm , v_theta, main_kernel_ind] = estimate(obj, m_samples, m_positions)
% >>>>>>> 9e4b127124bd4b15843c52f384da625c5ab101d9
			% v_estimate:     N x 1 vector with the signal estimate
			%
			% if obj.ch_type == 'RKHS superposition', then 
			%    m_alpha:        S x nKernels vector with a vector of
			%                    alphas per kernel, where nKernels =
			%                    size(obj.m_kernel,3) and S =
			%                    size(m_samples,1).
			%    v_theta:        empty
			%
			% if obj.ch_type == 'kernel superposition', then 
			%    m_alpha:        S x 1 vector with the vector of alphas
			%    v_theta:        nKernels x 1 vector with the weight of
			%                    each kernel
			
			if isstruct(sideInfo)
				m_positions = sideInfo.v_sampledEntries;
			else
				m_positions = sideInfo;
			end
			
			% Initial checks
            if isempty(obj.m_kernel) || isempty(obj.s_regularizationParameter)
                error('MkrGraphFunctionEstimator:notEnoughInfo',...
                    'Kernel and mu not set');
            elseif ~isequaln(size(m_samples),size(m_positions))				
                error('MkrGraphFunctionEstimator:inconsistentParameter',...
                    'size of m_positions and m_samples not the same');
            elseif max(m_positions(:)) > size(obj.m_kernel,1)
                error('MkrGraphFunctionEstimator:outOfBound', ...
                    'position out of bound');
            end						
            [N,Np,nKernel] = size(obj.m_kernel);  % N is # of vertices
            assert(N==Np, 'Kernel matrix should be square');
			assert(size(m_samples,2)==1,'not implemented');
			
			% Kernel preparation and scaling
			K_observed = obj.m_kernel(m_positions, m_positions, :);
			kernelScale = NaN(nKernel,1); 
			for iKernel = 1 : size(K_observed,3) 		
				kernelScale(iKernel) = trace(K_observed(:,:,iKernel))/N;
				%kernelScale(iKernel) = sum(sum(abs(K_observed(:,:,iKernel))))/N;
				K_observed(:,:,iKernel) = K_observed(:,:,iKernel)/kernelScale(iKernel);
            end
            
            % do cross validation to select best regularization parameter
            if length(obj.s_regularizationParameter) > 1
                obj.s_regularizationParameter = obj.crossValidation(m_samples, m_positions, obj.s_regularizationParameter);
            end
			
			% Estimation of alpha
			switch obj.ch_type
				case 'RKHS superposition' %juan's
					m_alpha_norm =  obj.estimateAlphaRKHSSuperposition( m_samples, K_observed );
					% undo scaling
					m_alpha = m_alpha_norm*diag(1./kernelScale);
					%m_alpha = m_alpha*diag(kernelScale);
					% recover signal on whole graph
					
					
					v_theta = [];
					if obj.b_estimateFreq
						% select kernel with more weight
						[norm_alpha, indices] = sort(sum(m_alpha.^2,1),'descend');
						factor = 1;
						if norm_alpha(1) > factor * norm_alpha(2)
							main_kernel_ind = indices(1);
						else
							main_kernel_ind = NaN;
						end
					end
					
					if ~isempty(obj.singleKernelPostEstimator)    % second stage single kernel regression
						[~,main_kernel_ind] = max(sum(m_alpha.^2,1));
						m_main_kernel = obj.m_kernel(:,:,main_kernel_ind);
						% modify to do crossvalidation + ...
						obj.singleKernelPostEstimator.m_kernel = m_main_kernel;
						v_estimate_vec = obj.singleKernelPostEstimator.estimate(m_samples, m_positions);						
					else
						v_estimate_vec = zeros(N,1);  % y = \sum K_i * alpha_i
						for iKernel = 1 : nKernel
							Ki = obj.m_kernel(:,m_positions,iKernel); % kernel of the whole graph
							v_estimate_vec = v_estimate_vec + Ki*m_alpha(:,iKernel);
						end						
					end
					
				case 'kernel superposition'
					[m_alpha,v_theta]=  obj.estimateAlphaKernelSuperposition( m_samples, K_observed ) ;
					% undo scaling
					v_theta = diag(1./kernelScale)*v_theta;
					% recover signal on whole graph
					m_learnedKernel = obj.thetaToKernel(obj.m_kernel(:, m_positions, :),v_theta);
					v_estimate_vec = m_learnedKernel*m_alpha;
					if ~isempty(obj.singleKernelPostEstimator)
						error('not implemented');
					end
					
				otherwise
					error('unrecognized property ch_type')
			
			end
			
			% selecting only the requested samples
			if isstruct(sideInfo)
				v_estimate.v_wantedSamples = v_estimate_vec(sideInfo.v_wantedEntries);
			else
				v_estimate = v_estimate_vec;
			end
			
			
		end
        
		
        function m_alpha = estimateAlphaRKHSSuperposition(obj,m_samples, m_sampledKernelMatrix )
            % use group lasso to solve alpha
            % Input:
            %       m_samples       observed signal
            %       K               kernel matrix for observed part			
            % Output:
            %       a               alpha in group lasso format
            %
            
            S = length(m_samples);
            mu = obj.s_regularizationParameter;
            y = m_samples;            % observed signal
            nKernel = size(m_sampledKernelMatrix,3);      % # of kernels

            % change variable to group lasso format
            A = NaN(S,S*nKernel);
            for iKernel = 1 : nKernel
                Ki = m_sampledKernelMatrix(:,:,iKernel);
                A(:, ((iKernel-1)*S + 1) : iKernel*S ) = real(mpower((Ki+Ki')/2,1/2));
            end
% 			A = reshape(K, [size(K,1) size(K,2)*size(K,3)]);
            
            % set the parameter for group lasso solver
            lambda = S/2 * mu;
            p = ones(nKernel,1) * S;
            rho = 1;
            alpha = 1;
            
            % solve the problem
            [r, history] = group_lasso(A, y, lambda, p, rho, alpha);
            r = real(r);
            
            % interpret the result
            m_alpha = NaN(S,nKernel);
            for iKernel = 1 : nKernel				
                sqrtKi = A(:, ((iKernel-1)*S+1) : iKernel*S);
                ri = r( ((iKernel-1)*S+1) : iKernel*S );
% 
% if rcond(sqrtKi)< 1e-17
% 	keyboard
% end
	
                m_alpha(:,iKernel) = real( sqrtKi \ ri );
			end
			
% 			
% 			m_alpha = zeros(length(m_positions), nKernel);
%             for iKernel = 1 : nKernel
%                 alpha_i = alpha( ((iKernel-1)*S + 1) : iKernel*S ); % extract ai
% 				m_alpha(:,iKernel) = alpha_i;  
% 			end
			
			
		end
        
		% functions for 'kernel superposition'
		function [v_alpha,v_theta] = estimateAlphaKernelSuperposition(obj,m_samples, m_sampledKernelMatrix )
			
			if size(m_samples,2)>1
				error('not implemented');
			end
			
			nKernels = size(m_sampledKernelMatrix,3);
			S = size(m_samples,1); % number of sampled vertices
			eta = obj.s_IIAStepSize;
			
			if isempty(obj.v_IIACenter)
				obj.v_IIACenter = zeros( nKernels,1);
			end
								
			% initialization
			v_theta = obj.v_IIACenter + [obj.s_IIARadius;zeros(nKernels-1,1)];		
			reg_K =  obj.thetaToKernel(m_sampledKernelMatrix,v_theta) + obj.s_regularizationParameter*S*eye(S);
			v_alpha = reg_K\m_samples;
			
			for iiter = 1:obj.s_IIAMaxIterations
		
				v_kappa = obj.alphaToKappa(v_alpha,m_sampledKernelMatrix);
				v_theta = obj.v_IIACenter + (obj.s_IIARadius/norm(v_kappa))*v_kappa;
				
				reg_K =  obj.thetaToKernel(m_sampledKernelMatrix,v_theta) + obj.s_regularizationParameter*S*eye(S);
				v_alpha_new = eta*v_alpha + (1-eta)*( reg_K\m_samples );
				
				if norm(v_alpha_new - v_alpha)<obj.s_IIATolerance					
					return
				end
				v_alpha = v_alpha_new;
				
			end
			warning('maximum number of iterations reached in IIA');			
			
		end
		
		function N = getNumOfVertices(obj)
            N = size(obj.m_kernel,1);
        end
		
	end
	
	methods(Static)
		function K_of_theta = thetaToKernel(m_sampledKernelMatrix,v_theta)
			K_of_theta = zeros(size(m_sampledKernelMatrix(:,:,1)));
			for iKernel = 1:size(m_sampledKernelMatrix,3)
				K_of_theta = K_of_theta + v_theta(iKernel)*m_sampledKernelMatrix(:,:,iKernel);
			end					
		end
		
		function v_kappa = alphaToKappa(v_alpha,m_sampledKernelMatrix)
			nKernels = size(m_sampledKernelMatrix,3);
			v_kappa = zeros(nKernels,1);
			for iKernel = 1:nKernels
				v_kappa(iKernel) = v_alpha'*m_sampledKernelMatrix(:,:,iKernel)*v_alpha;
			end
		end
		
		
	end
	
	
end