classdef L2MultiKernelKrigingCovEstimator < MultiKernelKrigingCovEstimator
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
        
        function obj = L2MultiKernelKrigingCovEstimator(varargin)
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
                    %%subtract obs noise
                    %m_residualCovariance=m_residualCovariance-obj.s_obsNoiseVar*eye(size(m_positions,1));
                    %s_maxElCov=max(max(m_residualCovariance));
                    %m_residualCovariance=m_residualCovariance/s_maxElCov;
                else
                    m_residualCovariance=t_residualCovariance;
                    %s_maxElCov=max(max(m_residualCovariance,'fro'));
                    %m_residualCovariance=m_residualCovariance;
                end
                for s_dictionaryInd=1:s_numberOfKernels
                    t_reducedSizeDictionary(s_dictionaryInd,:,:)=...
                        obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd));
%                     v_a(s_dictionaryInd)=2*trace...
%                         (squeeze(t_reducedSizeDictionary(s_dictionaryInd,:,:))'*...
%                         m_residualCovariance);
                end
%                 m_C=zeros(s_numberOfKernels,s_numberOfKernels);
%                 for s_dictionaryInd1=1:s_numberOfKernels
%                     for s_dictionaryInd2=1:s_numberOfKernels
%                         if (s_dictionaryInd1~=s_dictionaryInd2)
%                             m_C(s_dictionaryInd1,s_dictionaryInd2)=trace...
%                                 (squeeze(t_reducedSizeDictionary(s_dictionaryInd1,:,:))'*...
%                                 squeeze(t_reducedSizeDictionary(s_dictionaryInd2,:,:)));
%                         elseif (s_dictionaryInd1==s_dictionaryInd2)
%                             m_C(s_dictionaryInd1,s_dictionaryInd2)=2*trace...
%                                 (squeeze(t_reducedSizeDictionary(s_dictionaryInd1,:,:))'*...
%                                 squeeze(t_reducedSizeDictionary(s_dictionaryInd2,:,:)))...
%                                 +2*obj.s_lambda;
%                         end
%                     end
%                 end
                m_linearComb=zeros(size(m_residualCovariance));
                s_numberOfSamples=size(m_residualCovariance,1);
                m_eye=eye(s_numberOfSamples);
                 cvx_begin
				cvx_solver sedumi
                variables v_theta(1,s_numberOfKernels) m_difference(s_numberOfSamples,s_numberOfSamples)
                minimize square_pos(norm(m_difference,'fro'))+obj.s_lambda*sum_square(v_theta)
                subject to
                v_theta>=0;
                m_linearComb=0;               
                for s_kernelInd=1:s_numberOfKernels
                    m_linearComb=m_linearComb+squeeze(v_theta(s_kernelInd)*t_reducedSizeDictionary(s_kernelInd,:,:));
                end
                %m_difference==m_linearComb/(m_residualCovariance+0.01*m_eye)-diag(diag(m_linearComb/(m_residualCovariance+0.01*m_eye)));
                %m_difference==m_linearComb/(m_residualCovariance)-m_eye;
                m_difference==m_linearComb-m_residualCovariance;
                %sum(v_theta,1)==1;
                cvx_end 
                %m_theta(:,s_monteCarloInd)=m_C\v_a;
                m_theta(:,s_monteCarloInd)=v_theta';
                %zscoring creates sparse solutions...
                %%m_theta(:,s_monteCarloInd)=zscore(m_theta(:,s_monteCarloInd));
                %or
                m_theta(:,s_monteCarloInd)=m_theta(:,s_monteCarloInd);%/norm(m_theta(:,s_monteCarloInd));
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
        end
        

          function v_theta=estimateCoeffVectorGD(obj,t_residualCovariance,m_positions)
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
                m_reducedSizeDictionary=zeros(size(m_positions,1)*size(m_positions,1),s_numberOfKernels);
                if size(m_positions,2)~=1;
                    v_residualCovariance=vec(squeeze(t_residualCovariance...
                        (:,:,s_monteCarloInd)));
                else
                    v_residualCovariance=vec(t_residualCovariance);
                end
                for s_dictionaryInd=1:s_numberOfKernels
                    m_reducedSizeDictionary(:,s_dictionaryInd)=...
                        vec(obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd)));
                end
                m_theta(:,s_monteCarloInd)=m_theta(:,s_monteCarloInd);%/norm(m_theta(:,s_monteCarloInd));
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=ones(s_numberOfKernels,1);
                s_sens=10^-8;
                s_stepSize=0.001;
                while norm(v_thetap-v_theta)>s_sens
                    v_thetap=v_theta;
                    v_grad=2*obj.s_lambda*v_theta+2*(m_reducedSizeDictionary')*m_reducedSizeDictionary*v_theta-2*(m_reducedSizeDictionary')*v_residualCovariance;
                    v_theta=v_theta-s_stepSize*v_grad;
                    v_theta(v_theta<0)=0;
                end
                 m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end
        
        
          function v_theta=estimateCoeffVectorGDWithInit(obj,t_residualCovariance,m_positions,v_thetaInit)
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
                m_reducedSizeDictionary=zeros(size(m_positions,1)*size(m_positions,1),s_numberOfKernels);
                if size(m_positions,2)~=1;
                    v_residualCovariance=vec(squeeze(t_residualCovariance...
                        (:,:,s_monteCarloInd)));
                else
                    v_residualCovariance=vec(t_residualCovariance);
                end
                for s_dictionaryInd=1:s_numberOfKernels
                    m_reducedSizeDictionary(:,s_dictionaryInd)=...
                        vec(obj.t_kernelDictionary(s_dictionaryInd,...
                        m_positions(:,s_monteCarloInd),...
                        m_positions(:,s_monteCarloInd)));
                end
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=v_thetaInit;
                s_sens=10^-6;
                s_stepSize=0.001;
                while norm(v_thetap-v_theta)>s_sens
                    v_thetap=v_theta;
                    v_grad=2*obj.s_lambda*v_theta+2*(m_reducedSizeDictionary')*m_reducedSizeDictionary*v_theta-2*(m_reducedSizeDictionary')*v_residualCovariance;
                    s_stepSize=L2MultiKernelKrigingCovEstimator.armijoStepSizeRule(obj.s_lambda,m_reducedSizeDictionary,v_theta,v_residualCovariance);
                    v_theta=v_theta-s_stepSize*v_grad;
                    v_theta(v_theta<0)=0;
                end
                 m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
        end
        
        
        function m_kernel = getNewKernelMatrix(obj,graph)
            obj.m_laplacian = graph.getLaplacian();
            %obj.m_laplacian = graph.getNormalizedLaplacian();
            m_kernel = obj.generateKernelMatrix();
        end
    end
     methods(Static)
        function s_step= armijoStepSizeRule(s_lambda,m_reducedSizeDictionary,v_theta,v_residualCovariance)
            s_sigma=0.25;
            s_s=1;
            s_beta=0.5;
            s_alpha=s_s*s_beta;
            v_grad=2*s_lambda*v_theta+2*(m_reducedSizeDictionary')*m_reducedSizeDictionary*v_theta-2*(m_reducedSizeDictionary')*v_residualCovariance;
            v_d=-v_grad;
            v_thetanew=v_theta+s_alpha*v_d;
            while L2MultiKernelKrigingCovEstimator.f(s_lambda,m_reducedSizeDictionary,v_theta,v_residualCovariance)...
                    -L2MultiKernelKrigingCovEstimator.f(s_lambda,m_reducedSizeDictionary,pos(v_thetanew),v_residualCovariance)...
                    <-s_sigma*s_alpha*v_grad'*v_d
                s_alpha=s_alpha*s_beta;
                v_thetanew=v_theta+s_alpha*v_d;
            end
            s_step=s_alpha;
        end
        function s_val=f(s_lambda,m_reducedSizeDictionary,v_theta,v_residualCovariance)
           s_val=norm(v_residualCovariance-m_reducedSizeDictionary*v_theta)^2+s_lambda*norm(v_theta)^2;
        end
     end
    
end
