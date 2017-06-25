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
