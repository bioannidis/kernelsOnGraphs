classdef L2MultiKernelKrigingCorEstimator < MultiKernelKrigingCovEstimator
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
        
        function obj = L2MultiKernelKrigingCorEstimator(varargin)
            obj@MultiKernelKrigingCovEstimator(varargin{:});
        end
        
        
        
    end
    
    methods
        function v_theta=estimateCoeffVectorCVX(obj,t_correlations)
            %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_correlations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            for s_monteCarloInd=1:size(t_correlations,3)

                v_a=zeros(s_numberOfKernels,1);
                if size(t_correlations,3)~=1;
                    m_correlations=squeeze(t_correlations...
                        (:,:,s_monteCarloInd));
                else
                    m_correlations=t_correlations;
                end
                s_lam=obj.s_lambda;
                %s_lam*sum_square(v_theta)
                m_linearComb=zeros(size(m_correlations));
                m_sqrtcorr=sqrtm(m_correlations);
                s_numberOfVertices=size(m_correlations,1);
                v_theta=ones(s_numberOfKernels);
                
%                 t1=trace(m_sqrtcorr*inv(v_theta(1)*squeeze(obj.t_kernelDictionary(1,:,:))...
%                     +v_theta(2)*squeeze(obj.t_kernelDictionary(2,:,:)))*m_sqrtcorr)+1*norm(v_theta)^2;

                cvx_begin
				cvx_solver sedumi
                variables v_theta(1,s_numberOfKernels) t 
                minimize t +s_lam*sum_square(v_theta)
                subject to
                %sum_square(v_theta)<=1/s_lam
                v_theta>=0;
                %v_theta(1)>=0.1;
                t>=0;
                m_linearComb=0;               
                for s_kernelInd=1:s_numberOfKernels
                    m_linearComb=m_linearComb+squeeze(v_theta(s_kernelInd)*obj.t_kernelDictionary(s_kernelInd,:,:));
                end
               
                [m_linearComb m_sqrtcorr;...
                    m_sqrtcorr' (t/s_numberOfVertices)*eye(s_numberOfVertices)]>=0;
                %sum(v_theta)==1;
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
        

          function v_theta=estimateCoeffVectorGD(obj,t_transformedCorrelations,m_eigenvalues)
           %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_transformedCorrelations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            for s_monteCarloInd=1:size(t_transformedCorrelations,3)
                m_transformedCorrelations=t_transformedCorrelations(:,:,s_monteCarloInd);
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=ones(s_numberOfKernels,1);
                s_sens=10^-8;
                s_stepSize=0.1;
                s_it=1;
                
                while norm(v_thetap-v_theta)>s_sens
                    v_grad=obj.evaluateGrad(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_thetap=v_theta;
                    s_stepSize=obj.armijoStepSizeRuleM(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_theta=pos(v_theta-s_stepSize*v_grad);
                    s_val=obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta);
                    s_dif=norm(v_thetap-v_theta);
                    s_it=s_it+1;
                    if s_it==1000
                    end
                end
                s_it
                 m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end
          function v_theta=estimateCoeffVectorNewton(obj,t_transformedCorrelations,m_eigenvalues)
           %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_transformedCorrelations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            tic
            for s_monteCarloInd=1:size(t_transformedCorrelations,3)
                m_transformedCorrelations=t_transformedCorrelations(:,:,s_monteCarloInd);
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=ones(s_numberOfKernels,1);
                s_sens=10^-6;
                s_stepSize=1;
                s_it=1;
                m_thetaIter=zeros(s_numberOfKernels);
                s_recalchesperiod=10;
                while norm(v_thetap-v_theta)>s_sens
                    v_grad=obj.evaluateGrad(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_thetap=v_theta;
                    if mod(s_it,s_recalchesperiod)==0||s_it==1;
                        m_hes=obj.evaluateHes(m_transformedCorrelations,m_eigenvalues,v_theta);
                        m_invhes=inv(m_hes);
                    end
                    v_theta=pos(v_theta-s_stepSize*m_invhes*v_grad);
                    s_val=obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta);
                    s_dif=norm(v_thetap-v_theta);
                     m_thetaIter(:,s_it)=v_theta';
                    s_it=s_it+1;
                    if s_it==1000
                    end
                end
                %obj.plotOrigObj(m_thetaIter,m_transformedCorrelations,m_eigenvalues)
                m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            gd=toc
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end
          
          function v_theta=estimateCoeffVectorNewtonWithInit(obj,t_transformedCorrelations,m_eigenvalues,v_thetaInit)
           %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_transformedCorrelations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            tic
            for s_monteCarloInd=1:size(t_transformedCorrelations,3)
                m_transformedCorrelations=t_transformedCorrelations(:,:,s_monteCarloInd);
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=v_thetaInit;
                s_sens=10^-6;
                s_stepSize=1;
                s_it=1;
                m_thetaIter=zeros(s_numberOfKernels);
                s_recalchesperiod=10;
                while norm(v_thetap-v_theta)>s_sens
                    v_grad=obj.evaluateGrad(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_thetap=v_theta;
                    if mod(s_it,s_recalchesperiod)==0||s_it==1;
                        m_hes=obj.evaluateHes(m_transformedCorrelations,m_eigenvalues,v_theta);
                        m_invhes=inv(m_hes);
                    end
                    v_theta=pos(v_theta-s_stepSize*m_invhes*v_grad);
                    s_val=obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta);
                    s_dif=norm(v_thetap-v_theta);
                     m_thetaIter(:,s_it)=v_theta';
                    s_it=s_it+1;
                    if s_it==1000
                    end
                end
                %obj.plotOrigObj(m_thetaIter,m_transformedCorrelations,m_eigenvalues)
                m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            gd=toc
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end
          
          function v_theta=estimateCoeffVectorOnlyDiagNewton(obj,t_transformedCorrelations,m_eigenvalues)
           %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_transformedCorrelations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            for s_monteCarloInd=1:size(t_transformedCorrelations,3)
                m_transformedCorrelations=t_transformedCorrelations(:,:,s_monteCarloInd);
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=ones(s_numberOfKernels,1);
                s_sens=10^-6;
                s_stepSize=0.0001;
                s_it=1;
                while norm(v_thetap-v_theta)>s_sens
                    v_grad=obj.evaluateGrad(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_thetap=v_theta;
                    m_hes=obj.evaluateOnlyDiagHes(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_theta=pos(v_theta-s_stepSize*(m_hes)\v_grad);
                    s_val=obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta)
                    s_dif=norm(v_thetap-v_theta)
                   

                    s_it=s_it+1;
                    if s_it==1000
                    end
                end
                m_theta(:,s_monteCarloInd)=v_theta';
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end

          function v_theta=estimateCoeffVectorGDWithInit(obj,t_transformedCorrelations,m_eigenvalues,v_thetaInit)
             %% this function estimates the vectors of coefficients for the kernels
            % The minimization of pbj for kernel matching in the paper
            % t_correlations is an s_monteCarlo x N x N tensor 
            
            m_theta=zeros(size(obj.t_kernelDictionary,1),size(t_transformedCorrelations,3));
            s_numberOfKernels=size(obj.t_kernelDictionary,1);
            for s_monteCarloInd=1:size(t_transformedCorrelations,3)
                m_transformedCorrelations=t_transformedCorrelations(:,:,s_monteCarloInd);
                v_thetap=zeros(s_numberOfKernels,1);
                v_theta=v_thetaInit;
                s_sens=10^-6;
                s_stepSize=0.1;
                s_it=1;
                while norm(v_thetap-v_theta)>s_sens
                    v_grad=obj.evaluateGrad(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_thetap=v_theta;
                    s_stepSize=obj.armijoStepSizeRuleM(m_transformedCorrelations,m_eigenvalues,v_theta);
                    v_theta=pos(v_theta-s_stepSize*v_grad);
                    s_val=obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta);
                    s_dif=norm(v_thetap-v_theta);
                    
                    if s_it==1000
                    end
                    s_it=s_it+1;
                end
                %s_it
                 m_theta(:,s_monteCarloInd)=v_theta';
                 
                
            end
            
            
            
            %different thetas how to handle that?
            
            v_theta=mean(m_theta,2);
            v_theta(v_theta < 0) = 0;
          end
          function m_hes=evaluateHes(obj,m_transformedCorrelations,m_eigenvalues,v_theta)
            v_eigenvalues=m_eigenvalues*v_theta;
            v_inveigenvalues=1./(v_eigenvalues);
            m_hes=zeros(size(v_theta,1),size(v_theta,1));
            v_diagTransf=diag(m_transformedCorrelations);
            for s_ind1=1:size(m_hes,1)
                for s_ind2=1:size(m_hes,1)
                m_hes(s_ind1,s_ind2)=2*sum(v_diagTransf.*m_eigenvalues(:,s_ind1).*m_eigenvalues(:,s_ind2).*(v_inveigenvalues).^3);
                if s_ind1==s_ind2
                    m_hes(s_ind1,s_ind2)=m_hes(s_ind1,s_ind2)+2*obj.s_lambda;
                end
                end
            end
          end
      
          function m_hes=evaluateOnlyDiagHes(obj,m_transformedCorrelations,m_eigenvalues,v_theta)
            v_eigenvalues=m_eigenvalues*v_theta;
            v_inveigenvalues=1./(v_eigenvalues);
            m_hes=zeros(size(v_theta,1),size(v_theta,1));
            v_diagTransf=diag(m_transformedCorrelations);
            for s_ind1=1:size(m_hes,1)
                m_hes(s_ind1,s_ind1)=2*sum(v_diagTransf.*m_eigenvalues(:,s_ind1).*m_eigenvalues(:,s_ind1).*(v_inveigenvalues).^3);
                m_hes(s_ind1,s_ind1)=m_hes(s_ind1,s_ind1)+2*obj.s_lambda;
            end
          end

          function s_step= armijoStepSizeRuleM(obj,m_transformedCorrelations,m_eigenvalues,v_theta)
            s_sigma=0.25;
            s_s=1;
            s_beta=0.5;
            s_alpha=s_s*s_beta;
            v_grad=evaluateGrad(obj,m_transformedCorrelations,m_eigenvalues,v_theta);
            v_d=-v_grad;
            v_thetanew=v_theta+s_alpha*v_d;
            while obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,v_theta)...
                    -obj.evaluateObj(m_transformedCorrelations,m_eigenvalues,pos(v_thetanew))...
                    <-s_sigma*s_alpha*v_grad'*v_d
                s_alpha=s_alpha*s_beta;
                v_thetanew=v_theta+s_alpha*v_d;
            end
            s_step=s_alpha;
        end
          function v_grad=evaluateGrad(obj,m_transformedCorrelations,m_eigenvalues,v_theta)
            v_eigenvalues=m_eigenvalues*v_theta;
            v_inveigenvalues=1./(v_eigenvalues);
            v_auxVec=zeros(size(v_theta));
            for s_ind=1:size(v_auxVec)
                v_auxVec(s_ind)=-trace(diag(m_eigenvalues(:,s_ind).*(v_inveigenvalues.^2))*m_transformedCorrelations);
            end
            v_grad= v_auxVec+2*obj.s_lambda*v_theta;
          end
          function s_val=evaluateObj(obj,m_transformedCorrelations,m_eigenvalues,v_theta)
            v_eigenvalues=m_eigenvalues*v_theta;
            v_inveigenvalues=1./(v_eigenvalues);
            s_val= trace(diag(v_inveigenvalues)*m_transformedCorrelations)+obj.s_lambda*(norm(v_theta)^2);
          end
        
          function plotOrigObj(obj,m_thetaIter,m_transformedCor,m_eigenvalues)
              v_val=zeros(size(m_thetaIter,2),1);
              v_val2=zeros(size(m_thetaIter,2),1);
              for s_it=1:size(m_thetaIter,2)
                  m_linearComb=0;               
                for s_kernelInd=1:size(m_thetaIter,1)
                    m_linearComb=m_linearComb+squeeze(m_thetaIter(s_kernelInd,s_it)*obj.t_kernelDictionary(s_kernelInd,:,:));
                end
                v_val2(s_it)=obj.evaluateObj(m_transformedCor,m_eigenvalues,m_thetaIter(:,s_it));
                v_val(s_it)=trace(inv(m_linearComb)*m_transformedCor)+obj.s_lambda*norm(m_thetaIter(:,s_it))^2;
              end
              plot(v_val);
              plot(v_val2);
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
