classdef SEMGraphGraphFunctionEstimator< JointGraphGraphFunctionEstimator
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
        m_graphFunctionInitialize;%initialize non convex algo
        s_lambda1;
        s_lambda2;
        s_maxIterSEMInit;
        s_maxIterJoint;
        s_rho;
        s_tolSEM;
        s_mu;
        s_numberOfVertices;
        s_tolJoint;
        s_maxIterSEMLongRun;
    end
    
    methods
        
        function obj = SEMGraphGraphFunctionEstimator(varargin)
            obj@JointGraphGraphFunctionEstimator(varargin{:});
        end
        
    end
    
    methods
        function [m_adjacencyEstimate,m_graphFunctionEstimate] = estimate(obj,c_samples,c_positions)
               %   c_positions cell array containing L Slx1 vectors corresponding to the positions of the lth
            %   observation
            %   c_samples cell array containg L Slx1 vectors corresponding to the lth
            %   observation
            
            lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',obj.s_lambda1,'s_lambda2',obj.s_lambda2,'s_maxIter',obj.s_maxIterSEMInit,'s_rho',obj.s_rho,'s_tol',obj.s_tolSEM);
            lSEMGraphFunctionEstimator=LSEMGraphFunctionEstimator('s_mu',obj.s_mu);
            s_iter=0;
            % initialize
            s_flagConver=0;
            m_graphFunctionEstimate=zeros(obj.s_numberOfVertices,size(c_samples,2));
            for s_sampleInd=1:size(c_samples,2)
                m_graphFunctionEstimate(c_positions{s_sampleInd},s_sampleInd)=c_samples{s_sampleInd};
            end
            m_adjacencyEstimate=zeros(obj.s_numberOfVertices);
            while s_iter<obj.s_maxIterJoint && (~s_flagConver)
                s_iter=s_iter+1;
                if s_iter>4
                    lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',obj.s_lambda1,'s_lambda2',obj.s_lambda2,'s_maxIter',obj.s_maxIterSEMLongRun,'s_rho',obj.s_rho,'s_tol',obj.s_tolSEM);
                end
                m_adjacencyEstimateNew=lSEMGraphEstimator.estimate(m_graphFunctionEstimate);
                m_graphFunctionEstimateNew=lSEMGraphFunctionEstimator.estimate(c_samples,c_positions,m_adjacencyEstimateNew);
                v_errAdj(s_iter)=norm(m_adjacencyEstimate-m_adjacencyEstimateNew,'fro');
                v_errFun(s_iter)=norm(m_graphFunctionEstimate-m_graphFunctionEstimateNew,'fro');

                if v_errAdj(s_iter) <= obj.s_tolJoint && v_errFun(s_iter) <= obj.s_tolJoint 
                    s_flagConver = 1;
                end
                m_graphFunctionEstimate=m_graphFunctionEstimateNew;
                m_adjacencyEstimate=m_adjacencyEstimateNew;
            end
            
        end
         function [m_adjacencyEstimate,m_graphFunctionEstimate] = estimateRandInit(obj,c_samples,c_positions)
               %   c_positions cell array containing L Slx1 vectors corresponding to the positions of the lth
            %   observation
            %   c_samples cell array containg L Slx1 vectors corresponding to the lth
            %   observation
            
            lSEMGraphEstimator=LSEMGraphEstimator('s_lambda1',obj.s_lambda1,'s_lambda2',obj.s_lambda2,'s_maxIter',obj.s_maxIterSEM,'s_rho',obj.s_rho,'s_tol',obj.s_tolSEM);
            lSEMGraphFunctionEstimator=LSEMGraphFunctionEstimator('s_mu',obj.s_mu);
            s_iter=0;
            % initialize
            s_flagConver=0;
            m_graphFunctionEstimate=obj.m_graphFunctionInitialize;
            m_adjacencyEstimate=zeros(obj.s_numberOfVertices);
            while s_iter<obj.s_maxIterJoint && (~s_flagConver)
                s_iter=s_iter+1;
                
                m_adjacencyEstimateNew=lSEMGraphEstimator.estimate(m_graphFunctionEstimate);
                m_graphFunctionEstimateNew=lSEMGraphFunctionEstimator.estimate(c_samples,c_positions,m_adjacencyEstimateNew);
                v_errAdj(s_iter)=norm(m_adjacencyEstimate-m_adjacencyEstimateNew,'fro');
                v_errFun(s_iter)=norm(m_graphFunctionEstimate-m_graphFunctionEstimateNew,'fro');

                if v_errAdj(s_iter) <= obj.s_tolJoint && v_errFun(s_iter) <= obj.s_tolJoint 
                    s_flagConver = 1;
                end
                m_graphFunctionEstimate=m_graphFunctionEstimateNew;
                m_adjacencyEstimate=m_adjacencyEstimateNew;
            end
            
        end
        
    end
    
end
