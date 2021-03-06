classdef BatchEmpiricalMeanCovEstimator < EmpiricalMeanCovarianceEstimator
    %
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'BatchEmpiricalMeanCovEstimator';
        s_stepSize; %for nonstationary covariances
    end
    
    methods
        
        function obj = BatchEmpiricalMeanCovEstimator(varargin)
            obj@EmpiricalMeanCovarianceEstimator(varargin{:});
        end
        
        
        
    end
    
    methods
        %calculates the empirical covariance of a function in batch form
        function t_functionCov=calculateResidualCovBatch(obj,t_functionEstimates,s_numberOfSamples,s_monteCarloSimulations,s_trainTimePeriod)
            %t_functionEstimates is a s_numberOfVerticesxs_monteCarloSimulationsxs_trainTimePeriod
            %                   tensor containing the functions that their cov
            %                   will be estimated
            
            t_auxCovForSignal=zeros(s_numberOfSamples,s_numberOfSamples,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_residual=squeeze(t_functionEstimates(:,s_monteCarloSimulations,:));
                v_meanqAuxForSignal=mean(m_residual,2);
                m_meanqAuxForSignal=repmat(v_meanqAuxForSignal,[1,size(m_residual,2)]);
                t_auxCovForSignal(:,:,s_realizationCounter)=(1/s_trainTimePeriod-1)*...
                    (m_residual-m_meanqAuxForSignal)*(m_residual-m_meanqAuxForSignal)';
            end
            t_functionCov=t_auxCovForSignal;
        end
        %calculates the empirical correlation of a function in batch form
        function t_functionCor=calculateResidualCorBatch(obj,t_functionEstimates,s_numberOfVertices,s_monteCarloSimulations,s_timeWindow)
            %t_functionEstimates is a s_numberOfVerticesxs_monteCarloSimulationsxs_trainTimePeriod
            %                   tensor containing the functions that their cov
            %                   will be estimated
            
            t_auxCorForSignal=zeros(s_numberOfVertices,s_numberOfVertices,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_residual=squeeze(t_functionEstimates(:,s_monteCarloSimulations,:));
                m_cormat=zeros(size(m_residual,1));
                for s_ind1=1:size(m_residual,2);
                    m_cormat=m_cormat+m_residual(:,s_ind1)*m_residual(:,s_ind1)';
                end
                m_cormat=(1/s_timeWindow)*...
                    (m_cormat+eye(s_numberOfVertices));
                %regularized correlation mat
%                 m_cormat1=(1/s_timeWindow)*...
%                 (...
%                     (m_residual)*(m_residual)'+...
%                     eye(s_numberOfVertices));
                %[d,v]=eig(m_cormat);
                t_auxCorForSignal(:,:,s_realizationCounter)=m_cormat;
            end
            t_functionCor=t_auxCorForSignal;
        end
        
        %calculates the empirical mean of a function in batch form
        function m_functionMean=calculateResidualMeanBatch(obj,t_functionEstimates,s_numberOfSamples,s_monteCarloSimulations)
            m_auxMeanForSignal=zeros(s_numberOfSamples,s_monteCarloSimulations);
            for s_realizationCounter=1:s_monteCarloSimulations
                m_residual=squeeze(t_functionEstimates(:,s_monteCarloSimulations,:));
                m_auxMeanForSignal(:,s_realizationCounter)=mean(m_residual,2);
            end
            m_functionMean=m_auxMeanForSignal;
        
        end
    end
end
