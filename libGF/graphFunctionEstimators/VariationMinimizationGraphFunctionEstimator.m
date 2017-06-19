classdef VariationMinimizationGraphFunctionEstimator< GraphFunctionEstimator
    % This was written by Vassilis
	%
	% Signal Recovery on Graphs: Variation Minimization
    % Siheng Chen, Aliaksei Sandryhaila, Jos ́e M. F. Moura, Jelena Kovaˇcevi ́c
	%
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'VARIATIONMINIMIZATION';
        graph;
        
    end
    
    methods
        
        function obj = VariationMinimizationGraphFunctionEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        
    end
    
    methods
        
        function m_estimate = estimate(obj,m_samples,m_positions,s_alpha,s_beta,s_gama,s_eta,s_tol)
            %
            % Input:
            % M_SAMPLES                 S x S_NUMBEROFREALIZATIONS  matrix with
            %                           samples of the graph function in
            %                           M_GRAPHFUNCTION
            % M_POSITIONS               S x S_NUMBEROFREALIZATIONS matrix
            %                           containing the indices of the vertices
            %                           where the samples were taken
            %
            %S_ALPHA,S_BETA,S_GAMA,S_ETA
            %                           Scalars controling the
            %                           regularization
            %S_TOL
            %                           Scalar controling the tollerance
            %
            % Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
            %                           the number of nodes and each column
            %                           contains the estimate of the graph
            %                           function
            %
            
            %Create full matrix from samples
            m_T=zeros(s_numberOfVertices,size(m_samples,1));
            m_T(m_positions)=m_estimate;
            
            while s_dif>s_tol
                %min norm(W,fro)^2+alpha*variation(Z)
                %+beta*nuclearNorm(X)+gama*norm(E,1)+indicator(Cm);
                while s_dif>s_tol
                    %min beta*nuclearNorm(X)+eta/2
                    %*norm(T-X-W-E-C-Y1,fro)^2 +eta/2*norm(X-Z-Y2,fro)^2
                    %
                    t=backtrackingX();
                    m_X=shrinkSingularValues(m_X+t(m_T-m_W-m_E-m_C-(1/s_eta)*(m_Y1+m_Y2)-m_Z),s_beta/s_eta);
                end
                m_W=s_eta*(m_T-m_X-m_E-m_C-(1/s_eta)*m_Y1)/(s_eta+2);
                while s_dif>s_tol
                    t=backtrackingE();
                     m_E=shrinkElements(m_X+t*(m_T-m_X-m_W-m_C-(1/s_eta)*m_Y1),s_gama/s_eta);
                end
                m_Z;
            end
            
        end
        function m_theta=shrinkElements(m_X,s_tao)
            m_theta=m_X;
            m_theta(m_X>=s_tao)=m_X(m_X>=s_tao)-s_tao;
            m_theta(m_X<=-s_tao)=m_X(m_X<=-s_tao)+s_tao;
            m_theta(-s_tao<=m_X<=s_tao)=0;
        end
        function m_delta=shrinkSingularValues(m_X,s_tao)
            [m_U,m_S,m_Q]=svd(m_X);
            m_delta=m_U*shrinkElements(m_S,s_tao)*(m_Q');
        end
        function t=backtrackingX()
            t=0;
        end
        function t=backtrackingE()
            t=0;
        end
    end
    
    methods
        function N = getNumOfVertices(obj)
            N = obj.graph.getNumberOfVertices();
        end
    end
end
