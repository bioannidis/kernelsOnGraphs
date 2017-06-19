classdef EigDistMultiKernelKrigingCovEstimator < MultiKernelKrigingCovEstimator
    %
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'l2MultiKernelKrigingCovEstimator';
        s_lambda;     %regularizing parameter
        s_obsNoiseVar;     %observation noise variance
    end
    
    methods
        
        function obj = EigDistMultiKernelKrigingCovEstimator(varargin)
            obj@MultiKernelKrigingCovEstimator(varargin{:});
        end
        
        
        
    end
    
    methods
        function v_theta=estimateCoeffVector(obj,t_residualCovariance,m_positions)
            
            %% this function estimates the vectors of coefficients for the kernels
            % The metric in (4) A Metric for Covariance Matrices is used
            % Assumption: Covariance and LK share the same eigenvectors..
            % t_residualCovariance is an s_monteCarlo x N x N tensor with
            % the residual covariance estimat
            % M_POSITIONS               S x s_monteCarlo matrix
            %                           containing the indices of the vertices
            %                           where the samples were taken
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(m_positions,2));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            s_numberOfSampledVertices=size(m_positions,1);
            for s_monteCarloInd=1:size(m_positions,2)
                m_eigsCovKernInv=zeros(size(obj.t_kernelDictionary,1),size(m_positions,1));
                t_reducedSizeDictionary=zeros(s_numberOfKernels,...
                    size(m_positions,1),size(m_positions,1));
                if size(m_positions,2)~=1;
                    m_residualCovariance=squeeze(t_residualCovariance...
                        (:,:,s_monteCarloInd));
                    %subtract obs noise
                    m_residualCovariance=m_residualCovariance-obj.s_obsNoiseVar*eye(size(m_positions,1));
                    s_maxElCov=max(max(m_residualCovariance));
                    m_residualCovariance=m_residualCovariance/s_maxElCov;
                else
                    m_residualCovariance=t_residualCovariance;
                      %subtract obs noise
                    m_residualCovariance=m_residualCovariance-obj.s_obsNoiseVar*eye(size(m_positions,1));
                    s_maxElCov=max(max(m_residualCovariance,'fro'));
                    m_residualCovariance=m_residualCovariance/s_maxElCov;
                end
                for s_dictionaryInd=1:s_numberOfKernels
                    t_reducedSizeDictionary(s_dictionaryInd,:,:)=...
                        obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd));
                    m_eigsCovKernInv(s_dictionaryInd,:)=...
                        real(eig(squeeze(obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd)))/m_residualCovariance));
                end
                %quadprog
                %%
               
%                 v_ones=ones(s_numberOfSampledVertices,1);
%                 cvx_begin
% 				cvx_solver sedumi
%                 variables v_theta(1,s_numberOfKernels)
%                 minimize log(v_theta*m_eigsCovKernInv)*(log(v_theta*m_eigsCovKernInv)')
%                 subject to
%                 v_theta>=0;
%                 cvx_end 
                cvx_begin
				%cvx_solver sedumi
                variables v_theta(1,s_numberOfKernels)
                m_linearComb=zeros(size(m_residualCovariance));
                for s_kernelInd=1:s_numberOfKernels
                    m_linearComb=m_linearComb+squeeze(v_theta(s_kernelInd)*t_reducedSizeDictionary(s_kernelInd,:,:));
                end
                minimize sum_square(log(eig(m_linearComb*inv(m_residualCovariance))))+obj.s_lambda*sum_square(v_theta)
                subject to
                v_theta>=0;
                cvx_end 
                
               
                m_theta(:,s_monteCarloInd)=v_theta;
                %zscoring creates sparse solutions...
                %m_theta(:,s_monteCarloInd)=zscore(m_theta(:,s_monteCarloInd));
                %or
                %m_theta(:,s_monteCarloInd)=m_theta(:,s_monteCarloInd)/norm(m_theta(:,s_monteCarloInd));
                
            end
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
        end
        
        
        function m_kernel = getNewKernelMatrix(obj,graph)
            obj.m_laplacian = graph.getLaplacian();
            %obj.m_laplacian = graph.getNormalizedLaplacian();
            m_kernel = obj.generateKernelMatrix();
        end
        
    end
    
end
