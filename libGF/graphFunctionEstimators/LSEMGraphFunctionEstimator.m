classdef LSEMGraphFunctionEstimator< GraphFunctionEstimator
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
        ch_name = 'SEMGraphFunctionEstimator';
        s_mu;
    end
    
    methods
        
        function obj = LSEMGraphFunctionEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
              
   end
    
    methods
        function m_estimate = estimate(obj,c_samples,c_positions,m_adjacency)
            %   c_positions cell array containing L Slx1 vectors corresponding to the positions of the lth
            %   observation
            %         m_adjacency;   N x N matrix containing the adjacency
            %   c_samples cell array containg L Slx1 vectors corresponding to the lth
            %   observation
            %   This function solves 0.5|m_F - m_F*m_A|_F^2 + 0.5mu1\sum_{m=1}^M
            %   | v_ym - m_Sm*v_fm|^2_2
            %   mu1 : regularization parameters for the fitting term
            %   the uncostrained optimization problem has a closed form
            %   solution
            % Output:                   N x S_NUMBEROFREALIZATIONS matrix. N is
            %                           the number of nodes and each column
            %                           contains the estimate of the graph
            %                           function
            %
            s_mu=obj.s_mu;
            s_numberOfVertices = size(m_adjacency,1);
            s_numberOfRealizations = size(c_samples,2);
            m_K=inv((eye(s_numberOfVertices)-m_adjacency)'*(eye(s_numberOfVertices)-m_adjacency));
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            for realizationCounter = 1:s_numberOfRealizations
                v_ind=c_positions{realizationCounter};
                m_subK=m_K(v_ind,v_ind);
                v_y=c_samples{realizationCounter};
                %  f1=((eye(size(m_A))-m_A)'*(eye(size(m_A))-m_A)+s_mu1/size(v_ind,2)*m_S'*m_S)^(-1)*s_mu1/size(v_ind,2)*m_S'*v_y;
                m_estimate(:,realizationCounter)=m_K(:,v_ind)*(((size(v_ind,2)/s_mu)*eye(size(m_subK))+m_subK)\v_y);                
            end
            
            
        end
        
    end
    
end
