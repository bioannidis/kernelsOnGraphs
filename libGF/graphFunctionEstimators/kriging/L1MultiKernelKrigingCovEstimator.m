classdef L1MultiKernelKrigingCovEstimator < MultiKernelKrigingCovEstimator
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
        
        function obj = L1MultiKernelKrigingCovEstimator(varargin)
            obj@MultiKernelKrigingCovEstimator(varargin{:});
        end
        
        
        
    end
    
    methods
        function v_theta=estimateCoeffVectorCVX(obj,t_residualCovariance,m_positions)
            %% this function estimates the vectors of coefficients for the kernels
            % The minimization of the Frob
            % distance between the residual covariance
            % and the S sum( theta_m Kernel_m )S' and a l2 regularizer on
            % theta This results to a system of M equations that has closed
            % form solution
            % t_residualCovariance is an s_monteCarlo x N x N tensor with
            % the residual covariance estimat
            % M_POSITIONS               S x s_monteCarlo matrix
            %                           containing the indices of the vertices
            %                           where the samples were taken
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(m_positions,2));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            for s_monteCarloInd=1:size(m_positions,2)
                t_reducedSizeDictionary=zeros(s_numberOfKernels,...
                    size(m_positions,1),size(m_positions,1));
                v_a=zeros(s_numberOfKernels,1);
                if size(m_positions,2)~=1;
                    m_residualCovariance=squeeze(t_residualCovariance...
                        (:,:,s_monteCarloInd));
                    %subtract obs noise
%                     m_residualCovariance=m_residualCovariance-obj.s_obsNoiseVar*eye(size(m_positions,1));
%                     s_maxElCov=max(max(m_residualCovariance));
%                     m_residualCovariance=m_residualCovariance/s_maxElCov;
                else
                    m_residualCovariance=t_residualCovariance;
%                     s_maxElCov=max(max(m_residualCovariance,'fro'));
%                     m_residualCovariance=m_residualCovariance/s_maxElCov;
                end
                for s_dictionaryInd=1:s_numberOfKernels
                    t_reducedSizeDictionary(s_dictionaryInd,:,:)=...
                        obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd));
                   
                end
             
                 m_linearComb=zeros(size(m_residualCovariance));
                s_numberOfSamples=size(m_residualCovariance,1);
                m_eye=eye(s_numberOfSamples);
                 cvx_begin
				cvx_solver sedumi
                variables v_theta(1,s_numberOfKernels) m_difference(s_numberOfSamples,s_numberOfSamples)
                minimize square_pos(norm(m_difference,'fro'))+obj.s_lambda*norm(v_theta,1)
                subject to
                v_theta>=0;
                m_linearComb=0;               
                for s_kernelInd=1:s_numberOfKernels
                    m_linearComb=m_linearComb+squeeze(v_theta(s_kernelInd)*t_reducedSizeDictionary(s_kernelInd,:,:));
                end
                %m_difference==m_linearComb/(m_residualCovariance+0.01*m_eye)-diag(diag(m_linearComb/(m_residualCovariance+0.01*m_eye)));
                %m_difference==m_linearComb/(m_residualCovariance+0.01*m_eye)-m_eye;
                m_difference==m_linearComb-m_residualCovariance;
                cvx_end 
                
                m_theta(:,s_monteCarloInd)=v_theta';
                %zscoring creates sparse solutions...
                %m_theta(:,s_monteCarloInd)=zscore(m_theta(:,s_monteCarloInd));
                %or
                %m_theta(:,s_monteCarloInd)=m_theta(:,s_monteCarloInd)/norm(m_theta(:,s_monteCarloInd));
                
            end
            
            
            
            
            v_theta=mean(m_theta,2);
%             v_theta(v_theta < 0) = 0;
        end
        
        
        function m_kernel = getNewKernelMatrix(obj,graph)
            obj.m_laplacian = graph.getLaplacian();
            %obj.m_laplacian = graph.getNormalizedLaplacian();
            m_kernel = obj.generateKernelMatrix();
        end
        
    end
    
end
