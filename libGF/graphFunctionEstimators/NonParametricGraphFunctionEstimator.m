classdef NonParametricGraphFunctionEstimator< GraphFunctionEstimator
    % This was written by Vassilis
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    %test commi
    properties
        ch_name = 'NONPARAMETRIC';
        m_kernels;   % N x N matrix containing the kernel function evalueated at each pair of nodes
        s_lambda;
    end
    
    methods
        
        function obj = NonParametricGraphFunctionEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        
        function N = getNumOfVertices(obj)
            N = size(obj.m_kernel,1);
        end
    end
    
    methods
        function m_estimate = estimate(obj,m_samples,m_positions,s_lambda)
            %
            % Input:
            % M_SAMPLES                 S x S_NUMBEROFREALIZATIONS  matrix with
            %                           samples of the graph function in
            %                           M_GRAPHFUNCTION
            % M_POSITIONS               S x S_NUMBEROFREALIZATIONS matrix
            %                           containing the indices of the vertices
            %                           where the samples were taken
            %
            % Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
            %                           the number of nodes and each column
            %                           contains the estimate of the graph
            %                           function
            %
           if isempty(s_lambda)
            s_lambda=obj.s_lambda;
           end
            s_numberOfVertices = size(obj.m_kernels,1);
            s_numberOfRealizations = size(m_samples,2);
            
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            for realizationCounter = 1:s_numberOfRealizations
                
                m_subK=obj.m_kernels(m_positions(:,realizationCounter),m_positions(:,realizationCounter));
                v_alphas=(m_subK+s_lambda*size(m_subK,1)*eye(size(m_subK)))\m_samples(:,realizationCounter);
                m_estimate(:,realizationCounter) = obj.m_kernels(:,m_positions(:,realizationCounter))*v_alphas;
                
                
            end
            
            
        end
        
    end
    
end
