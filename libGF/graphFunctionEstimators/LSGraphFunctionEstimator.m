classdef LSGraphFunctionEstimator < GraphFunctionEstimator
    % LS method (Graph Signal Processing method) to estimate function on
    % graph.
    properties
        c_parsToPrint  = {'ch_name','s_edgeProbability','s_numberOfVertices'};
		c_stringToPrint  = {'','Edge prob.',''};
		c_patternToPrint = {'%s%s','%s = %g','%s%d vertices'};
    end
    
    properties
        s_bandwidth = [];
        graph;
    end
    
    methods
        function obj = LSGraphFunctionEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        
        function m_estimate = estimate(obj,m_samples,m_positions)
            % LS estimate of functions on graph
            % Input:
            %       obj -- object of LSGraphFunctionEstimator
            %              this object should have paramter of bandwidth
            %              and propery of the underlying graph on which
            %              graph signal resides
            %       m_samples -- matrix of function samples, where each
            %                    column is a valid sample
            %       m_positions -- matrix of sample positions; same size
            %                      with m_samples; each column
            %                      corresponding to positions of sample
            % Output:
            %       m_estimate -- matrix estimated signal; each column is a
            %                     valid estimated signal
            if isempty(obj.graph) || isempty(obj.s_bandwidth)
                error('LSEstimator:ParameterNotSet','Set underlying graph and bandwidth first')
            end
            
            if ~isequaln(size(m_samples), size(m_positions))
                error('LSEstimator:IllegalParameter','Paramter size not equal!')
            end
            
            % check out of bound positions
            if max(m_positions(:)) > obj.graph.getNumberOfVertices() || ...
                    min(m_positions(:)) < 1
                error('LSEstimator:SampleGraphInconsistent','m_positions not valide');
            end
            
            V = obj.graph.getLaplacianEigenvectors();
            B = obj.s_bandwidth;
            VB = V(:,1:B);
            I = eye(obj.graph.getNumberOfVertices());
            for iSample = 1 : size(m_samples,2)
                observedSignal = m_samples(:,iSample);
                samplingSet = m_positions(:,iSample);
                
                PhiS = I(samplingSet,:);
                m_estimate(:,iSample) = VB*( (PhiS*VB) \ observedSignal );
            end
        end
    end
    
end

