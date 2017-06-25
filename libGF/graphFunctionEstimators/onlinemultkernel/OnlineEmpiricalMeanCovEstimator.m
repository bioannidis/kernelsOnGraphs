classdef OnlineEmpiricalMeanCovEstimator < EmpiricalMeanCovarianceEstimator
    %
    properties(Constant)
    end
    
    properties % Required by superclass Parameter
        c_parsToPrint    = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties
        ch_name = 'onlineEmpiricalCovEstimator';
        s_stepSize; %for nonstationary covariances
    end
    
    methods
        
        function obj = OnlineEmpiricalMeanCovEstimator(varargin)
            obj@EmpiricalMeanCovarianceEstimator(varargin{:});
        end
        
        
        
    end
    
    methods
        
        function [t_functionCov,m_functionMean]=incrementalCalcResCovMean...
                (obj,t_functionCov,t_function,s_timeInd,m_functionMean)
            if obj.s_stepSize==0
                s_stepSize=1/(s_timeInd-1);
            else
                s_stepSize=obj.s_stepSize;
            end
            for s_realizationCounter=1:size(m_functionMean,2)
                v_vt=(t_function(:,s_realizationCounter)-m_functionMean(:,s_realizationCounter));
                m_functionMean(:,s_realizationCounter)=m_functionMean(:,s_realizationCounter)+(1/s_timeInd)*v_vt;
                t_functionCov(:,:,s_realizationCounter)=t_functionCov(:,:,s_realizationCounter)+...
                    s_stepSize*(((s_timeInd-1)/s_timeInd)*v_t*v_t'-t_functionCov(:,:,s_realizationCounter));
            end
        end

        

    end
end
