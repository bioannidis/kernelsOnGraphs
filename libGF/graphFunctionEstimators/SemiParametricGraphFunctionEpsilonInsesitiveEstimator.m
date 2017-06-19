classdef SemiParametricGraphFunctionEpsilonInsesitiveEstimator< GraphFunctionEstimator
    % This was written by Vassilis
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'SEMIPARAMETRIC';
        m_kernels;   % N x N matrix containing the kernel function evalueated at each pair of nodes
        m_basis; % N x B matrix containing the basis functions evalueated at each node
        
    end
    
    methods
        
        function obj = SemiParametricGraphFunctionEpsilonInsesitiveEstimator(varargin)
            obj@GraphFunctionEstimator(varargin{:});
        end
        
    end
    
    methods
        function m_basisSamp=get_proper_basis(obj,v_x)
            %when we have no samples from a specific category
            %we must drop the corresponding collumns from the parameters
            m_basisSamp=obj.m_basis;
            Bsamp=m_basisSamp(v_x,:);
            k=0;
            %here I discard the columns of B if I have no sample from them
            for i=1:size(Bsamp,2)
                if(Bsamp(:,i)==zeros(size(Bsamp,1),1))
                    i;
                else
                    k=k+1;
                    ind(k)=i;
                end
            end
            if(exist('ind')~=0)
                m_basisSamp=m_basisSamp(:,ind);
            end
        end
        function m_estimate = estimate(obj,m_samples,m_positions,s_lambda,s_epsilon)
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
            
            s_numberOfVertices = size(obj.m_kernels,1);
            s_numberOfRealizations = size(m_samples,2);
            s_numberOfSampledVertices= size(m_positions,1);
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            
            for realizationCounter = 1:s_numberOfRealizations
                %find appropriate subbasis
                m_SubBasis=obj.get_proper_basis(m_positions(:,realizationCounter) );
                m_SubBasisSamp=m_SubBasis(m_positions(:,realizationCounter),:);
                %   [C,IA,IC] = UNIQUE(A,'rows') also returns index vectors IA and IC such
                %   that C = A(IA,:) and A = C(IC,:). 
                [m_SubBasisSamp,v_iA,v_iC]=unique(m_SubBasisSamp','rows');
                m_SubBasisSamp=m_SubBasisSamp';
                m_SubBasis=m_SubBasis(:,v_iA);
                
                %find subKernel
                m_subK=obj.m_kernels(m_positions(:,realizationCounter),m_positions(:,realizationCounter));
                s_sizeOfBasis=size(m_SubBasisSamp,2);
                %quadprog
                %%
                cvx_begin
				%cvx_solver sedumi
                variables v_alphas(s_numberOfSampledVertices) v_betas(s_sizeOfBasis) v_ksi(s_numberOfSampledVertices) v_ksi_tilda(s_numberOfSampledVertices)
                minimize ((1/s_numberOfSampledVertices) *(v_ksi+v_ksi_tilda)'*ones(s_numberOfSampledVertices,1)+s_lambda*v_alphas'*m_subK*v_alphas);
                subject to
                v_ksi>=0;
                v_ksi_tilda>=0;
                v_ksi+s_epsilon>=m_samples(:,realizationCounter)-(m_subK*v_alphas+m_SubBasisSamp*v_betas);
                v_ksi_tilda+s_epsilon>=-m_samples(:,realizationCounter)+(m_subK*v_alphas+m_SubBasisSamp*v_betas);
                cvx_end 
				
                m_estimate(:,realizationCounter) = obj.m_kernels(:,m_positions(:,realizationCounter))*v_alphas+ m_SubBasis*v_betas;
                
            end
            
            
        end
        
    end
    
    methods
        function N = getNumOfVertices(obj)
            N = size(obj.m_kernel,1);
        end
    end
    
end
