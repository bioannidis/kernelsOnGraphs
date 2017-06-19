classdef ExponentiallyDecayingGraphFunctionGenerator < GraphFunctionGenerator
    % Generate exponentially decaying function on graphs [anis2016proxies]
    % need to specify the graph and bandwidth
    
    properties % Required by superclass Parameter
		c_parsToPrint    = {'ch_name','s_bandwidth'};
		c_stringToPrint  = {'','B'};
		c_patternToPrint = {'%s%s signal','%s = %d'};
	end 
    
    properties
		ch_name = 'Exp. decaying';
        s_bandwidth;
		s_decayingRate = 4; 
    end
    
    methods
        function obj = ExponentiallyDecayingGraphFunctionGenerator(varargin)
            obj@GraphFunctionGenerator(varargin{:});
        end
        
        function M_graphFunction = realization(obj,s_numberOfRealizations)
            % First generate lambda according to normal distribution
            %                coefficient ~ N(1, 0.5^2)
            % Then rescale by h(lambda) according to the following rule
            %      h(lambda) = 1;    if lambda <= lambda_r
            %      h(lambda) = exp( -obj.s_decayingRate*(lambda-lambda_r) );  if lambda > lambda_r
            % Therefore, the signal can be obtained by
            %      x = V * (coefficient * h(r))
            
			if nargin<2
				s_numberOfRealizations = 1;
			end
			
            if isempty(obj.graph) || isempty(obj.s_bandwidth)
                error('ExponentiallyDecayingFunctionGenerator: Paramter not set');
            end
            assert(isa(obj.graph,'Graph'));
            
            L = obj.graph.getLaplacian();
            [V,D] = eig(L);
            lambda = diag(D);
            
            lambda_r = lambda(obj.s_bandwidth);
            h_lambda = exp( -obj.s_decayingRate*(lambda - lambda_r ) );   % h(lambda)
            h_lambda( lambda < lambda_r ) = 1;
            
            M_graphFunction = NaN(size(L,1), s_numberOfRealizations);
            for iRealization = 1 : s_numberOfRealizations
                coef = 1 + randn( size(L,1) ,1 )/2;     % coef ~ N(1, 0.25)
                M_graphFunction(:,iRealization) = V * (coef .* h_lambda);
            end
            
        end
    end
    
end

