classdef SemiParametricGraphFunctionEstimator< GraphFunctionEstimator
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
        s_lambda;
    end
    
    methods
        
        function obj = SemiParametricGraphFunctionEstimator(varargin)
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
            %REMOVE AS AN ARGUMENT
            %s_lambda=obj.s_lambda
			
            s_numberOfVertices = size(obj.m_kernels,1);
            s_numberOfRealizations = size(m_samples,2);
            %s_epsilon is used to invert a singular subBasis
            s_epsilon=0;
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
                r=rank(m_SubBasisSamp);
                %find subKernel
                m_subK=obj.m_kernels(m_positions(:,realizationCounter),m_positions(:,realizationCounter));

                m_P=m_SubBasisSamp*((m_SubBasisSamp'*m_SubBasisSamp+eye(size(m_SubBasisSamp,2))*s_epsilon)\(m_SubBasisSamp'));
                m_H=(eye(size(m_P))-m_P)'*(eye(size(m_P))-m_P);
                v_alphas=(m_subK'*(m_H*m_subK+s_lambda*size(m_subK,1)*eye(size(m_subK))))\(m_subK'*(m_H*m_samples(:,realizationCounter)));
                v_betas=(m_SubBasisSamp'*m_SubBasisSamp)\(m_SubBasisSamp'*(m_samples(:,realizationCounter)-m_subK*v_alphas));
                m_estimate(:,realizationCounter) = obj.m_kernels(:,m_positions(:,realizationCounter))*v_alphas+ m_SubBasis*v_betas;
            end
            
          
        end
        
        function m_estimate = estimateGD(obj,m_samples,m_positions,s_lambda)
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
            %REMOVE AS AN ARGUMENT
            %s_lambda=obj.s_lambda
			
            s_numberOfVertices = size(obj.m_kernels,1);
            s_numberOfRealizations = size(m_samples,2);
            %s_epsilon is used to invert a singular subBasis
            s_epsilon=0;
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            for realizationCounter = 1:s_numberOfRealizations
                %find appropriate subbasis
                m_SubBasis=obj.get_proper_basis(m_positions(:,realizationCounter) );
                m_subBasisSamp=m_SubBasis(m_positions(:,realizationCounter),:);
                %   [C,IA,IC] = UNIQUE(A,'rows') also returns index vectors IA and IC such
                %   that C = A(IA,:) and A = C(IC,:). 
                [m_subBasisSamp,v_iA,v_iC]=unique(m_subBasisSamp','rows');
                m_subBasisSamp=m_subBasisSamp';
                m_SubBasis=m_SubBasis(:,v_iA);
                r=rank(m_subBasisSamp);
                %find subKernel
                m_subK=obj.m_kernels(m_positions(:,realizationCounter),m_positions(:,realizationCounter));
                %%
                %GD 
                s_stepSizeA=5;
                s_stepSizeB=5;
                b_bool=1;
                s_iter=0;
                s_maxIter=10^5;
                s_sens=10^-5;
                s_betaArm=0.4;
                s_s=10^2;
                s_sigmaArm=0.4;
                s_numberOfSamples=size(m_subK,1);
                v_alphas=randn(s_numberOfSamples,1);
                v_betas=randn(size(m_subBasisSamp,2),1);
                while b_bool && (s_iter<s_maxIter)
                    s_iter=s_iter+1;
                    v_directAlpha=-obj.gradientA(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    v_directBeta=-obj.gradientB(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    %s_step=obj.armijoRule(s_betaArm,s_sigmaArm,s_s,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter),v_directAlpha,v_directBeta);
                    v_alphasNew=v_alphas+s_stepSizeA*v_directAlpha;
                    v_betasNew=v_betas+s_stepSizeB*v_directBeta;
                   
                    v_funcVal(s_iter)=obj.functionEvaluate(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    if(s_iter>1)&&((v_funcVal(s_iter-1)-v_funcVal(s_iter))<0)
                        s_stepSizeA=s_stepSizeA*0.5;
                        s_stepSizeB=s_stepSizeB*0.5;
                    end
                    if (norm(v_alphasNew-v_alphas)<s_sens)&&(norm(v_betasNew-v_betas)<s_sens)
                            b_bool=0;
                    end
                    v_alphas=v_alphasNew;
                    v_betas=v_betasNew;
                    

                end
                
                
%                 m_P=m_subBasisSamp*((m_subBasisSamp'*m_subBasisSamp+eye(size(m_subBasisSamp,2))*s_epsilon)\(m_subBasisSamp'));
%                 m_H=(eye(size(m_P))-m_P)'*(eye(size(m_P))-m_P);
%                 v_alphas1=(m_subK'*(m_H*m_subK+s_lambda*size(m_subK,1)*eye(size(m_subK))))\(m_subK'*(m_H*m_samples(:,realizationCounter)));
%                 v_betas1=(m_subBasisSamp'*m_subBasisSamp)\(m_subBasisSamp'*(m_samples(:,realizationCounter)-m_subK*v_alphas1));
                m_estimate(:,realizationCounter) = obj.m_kernels(:,m_positions(:,realizationCounter))*v_alphas+ m_SubBasis*v_betas;
            end
            
          
        end
        function m_estimate = estimateGDArm(obj,m_samples,m_positions,s_lambda)
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
            %REMOVE AS AN ARGUMENT
            %s_lambda=obj.s_lambda
			
            s_numberOfVertices = size(obj.m_kernels,1);
            s_numberOfRealizations = size(m_samples,2);
            %s_epsilon is used to invert a singular subBasis
            s_epsilon=0;
            m_estimate = zeros(s_numberOfVertices,s_numberOfRealizations);
            for realizationCounter = 1:s_numberOfRealizations
                %find appropriate subbasis
                m_SubBasis=obj.get_proper_basis(m_positions(:,realizationCounter) );
                m_subBasisSamp=m_SubBasis(m_positions(:,realizationCounter),:);
                %   [C,IA,IC] = UNIQUE(A,'rows') also returns index vectors IA and IC such
                %   that C = A(IA,:) and A = C(IC,:). 
                [m_subBasisSamp,v_iA,v_iC]=unique(m_subBasisSamp','rows');
                m_subBasisSamp=m_subBasisSamp';
                m_SubBasis=m_SubBasis(:,v_iA);
                r=rank(m_subBasisSamp);
                %find subKernel
                m_subK=obj.m_kernels(m_positions(:,realizationCounter),m_positions(:,realizationCounter));
                %%
                %GD 
                s_stepSizeA=1;
                s_stepSizeB=1;
                b_bool=1;
                s_iter=0;
                s_maxIter=10^5;
                s_sens=10^-5;
                s_betaArm=0.4;
                s_s=10;
                s_sigmaArm=0.4;
                s_numberOfSamples=size(m_subK,1);
                v_alphas=randn(s_numberOfSamples,1);
                v_betas=randn(size(m_subBasisSamp,2),1);
                while b_bool && (s_iter<s_maxIter)
                    s_iter=s_iter+1;
                    v_directAlpha=-obj.gradientA(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    v_directBeta=-obj.gradientB(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    s_step=obj.armijoRule(s_betaArm,s_sigmaArm,s_s,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter),v_directAlpha,v_directBeta);
                    v_alphasNew=v_alphas+s_step*v_directAlpha;
                    v_betasNew=v_betas+s_step*v_directBeta;
                   
                    v_funcVal(s_iter)=obj.functionEvaluate(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,m_samples(:,realizationCounter));
                    if (norm(v_alphasNew-v_alphas)<s_sens)&&(norm(v_betasNew-v_betas)<s_sens)
                            b_bool=0;
                    end
                    v_alphas=v_alphasNew;
                    v_betas=v_betasNew;
                    

                end
                
                
%                 m_P=m_subBasisSamp*((m_subBasisSamp'*m_subBasisSamp+eye(size(m_subBasisSamp,2))*s_epsilon)\(m_subBasisSamp'));
%                 m_H=(eye(size(m_P))-m_P)'*(eye(size(m_P))-m_P);
%                 v_alphas1=(m_subK'*(m_H*m_subK+s_lambda*size(m_subK,1)*eye(size(m_subK))))\(m_subK'*(m_H*m_samples(:,realizationCounter)));
%                 v_betas1=(m_subBasisSamp'*m_subBasisSamp)\(m_subBasisSamp'*(m_samples(:,realizationCounter)-m_subK*v_alphas1));
                m_estimate(:,realizationCounter) = obj.m_kernels(:,m_positions(:,realizationCounter))*v_alphas+ m_SubBasis*v_betas;
            end
            
          
        end
        function s_step=armijoRule(obj,s_betaArm,s_sigmaArm,s_s,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations,v_directAlpha,v_directBeta)
            while(obj.functionEvaluate(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)-...
                   obj.functionEvaluate(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas+s_s*v_directAlpha,v_betas+s_s*v_directBeta,s_lambda,v_observations)...
                   <-s_sigmaArm*s_s*(obj.gradientA(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)'*v_directAlpha...
                   +obj.gradientB(s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)'*v_directBeta))
               s_s=s_s*s_betaArm;       
            end
            s_step=s_s;
        end
        function v_grad=gradientA(obj,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)
            v_grad=(-2/s_numberOfSamples)*m_subK*(v_observations-m_subK*v_alphas-m_subBasisSamp*v_betas)+2*s_lambda*m_subK*v_alphas;
        end
        function v_grad=gradientB(obj,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)
            v_grad=(-2/s_numberOfSamples)*m_subBasisSamp'*(v_observations-m_subK*v_alphas-m_subBasisSamp*v_betas);
        end
         function v_val=functionEvaluate(obj,s_numberOfSamples,m_subK,m_subBasisSamp,v_alphas,v_betas,s_lambda,v_observations)
            v_val=(1/s_numberOfSamples)*norm(v_observations-m_subK*v_alphas-m_subBasisSamp*v_betas)^2+s_lambda*v_alphas'*m_subK*v_alphas;
        end
        
        function N = getNumOfVertices(obj)
            N = size(obj.m_kernels,1);
        end
    end
    
end
