classdef SVARGraphGraphFunctionEstimator< JointGraphGraphFunctionEstimator
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
        t_graphFunctionInitialize;%initialize non convex algo
        s_lambda10;
        s_lambda20;
        s_lambda11;
        s_lambda21;
        s_maxIterSEMInit;
        s_maxIterJoint;
        s_rho;
        s_tolSEM;
        b_plot=0;
        s_mu;
        s_numberOfVertices;
        m_adjInit;
        s_boolInit=1; %1 initialize with adj 0 with function 
        m_adjtempInit;
        s_maximumTime
        s_tolJoint;
        s_maxIterSEMLongRun;
        t_initialErrorCov; %for RTS
        m_initialState; 
    end
    
    methods
        
        function obj = SVARGraphGraphFunctionEstimator(varargin)
            obj@JointGraphGraphFunctionEstimator(varargin{:});
        end
        
    end
    
    methods
        function [m_adjacency,m_timeAdjacency,t_graphFunctionEstimate] = estimate(obj,t_samples,t_positions)
            % this initializes the graph function to the observed values
            
            
            s_iter=0;
            % initialize
            s_flagConver=0;
            if obj.s_boolInit==0 
                t_graphFunctionEstimate=obj.t_graphFunctionInitialize;                
            else
                m_adjacency=obj.m_adjInit;
                m_timeAdjacency=obj.m_adjtempInit;
            end
            if obj.s_boolInit==0
                m_adjacency=zeros(obj.s_numberOfVertices);
                m_timeAdjacency=zeros(obj.s_numberOfVertices);
                t_graphFunctionEstimate=zeros(obj.s_numberOfVertices,obj.s_maximumTime,size(obj.m_initialState,2));
                while s_iter<obj.s_maxIterJoint && (~s_flagConver)
                    s_iter=s_iter+1;
                    lSVARGraphEstimator=LSVARGraphEstimator('s_lambda10',obj.s_lambda10,'s_lambda11',obj.s_lambda11,'s_lambda20',obj.s_lambda20,'s_lambda21',obj.s_lambda21...
                         ,'s_maxIter',obj.s_maxIterSEMInit,'s_rho',obj.s_rho,'s_tol',obj.s_tolSEM,'v_initGraphSignal',mean(t_graphFunctionEstimateNew(:,1,:),3));
                    [m_adjacencyNew,m_timeAdjacencyNew]=lSVARGraphEstimator.estimate(t_graphFunctionEstimate);
                    m_adjTilde=eye(size(m_adjacencyNew))-m_adjacencyNew;
                    m_transMat=(m_adjTilde)\m_timeAdjacencyNew;
                    m_transCov=inv((m_adjTilde)'*(m_adjTilde));
                    rTSSmootherFunctionEstimator=RTSSmootherFunctionEstimator('s_maximumTime',size(t_graphFunctionEstimate,2),...
                    'm_initialState',obj.m_initialState,'t_initialErrorcov',obj.t_initialErrorCov,...
                    'm_transMat',m_transMat,'m_transCov',m_transCov,'s_sigma',1/sqrt(obj.s_mu));

                    v_errAdj(s_iter)=norm(m_adjacency-m_adjacencyNew,'fro');
                    v_errTimeAdj(s_iter)=norm(m_timeAdjacency-m_timeAdjacencyNew,'fro');

                    v_errFun(s_iter)=norm(t_graphFunctionEstimate-t_graphFunctionEstimateNew,'fro');

                    if v_errAdj(s_iter) <= obj.s_tolJoint && v_errFun(s_iter) <= obj.s_tolJoint  && v_errTimeAdj(s_iter)<= obj.s_tolJoint
                        s_flagConver = 1;
                    end
                    t_graphFunctionEstimate=t_graphFunctionEstimateNew;
                    m_adjacency=m_adjacencyNew;
                    m_timeAdjacency=m_timeAdjacencyNew;
                end
            else
              t_graphFunctionEstimate=zeros(obj.s_numberOfVertices,obj.s_maximumTime,size(obj.m_initialState,2));
              while s_iter<obj.s_maxIterJoint && (~s_flagConver)
                    s_iter=s_iter+1;
                    m_adjTilde=eye(size(m_adjacency))-m_adjacency;
                    m_transMat=(m_adjTilde)\m_timeAdjacency;
                    m_transCov=pinv((m_adjTilde)'*(m_adjTilde));
                    rTSSmootherFunctionEstimator=RTSSmootherFunctionEstimator('s_maximumTime',obj.s_maximumTime,...
                    'm_initialState',obj.m_initialState,'t_initialErrorcov',obj.t_initialErrorCov,...
                    'm_transMat',m_transMat,'m_transCov',m_transCov,'s_sigma',1/sqrt(obj.s_mu));
                    [m_initialGraphSignal,t_graphFunctionEstimateNew]=rTSSmootherFunctionEstimator.estimate(t_samples,t_positions);
                    lSVARGraphEstimator=LSVARGraphEstimator('s_lambda10',obj.s_lambda10,'s_lambda11',obj.s_lambda11,'s_lambda20',obj.s_lambda20,'s_lambda21',obj.s_lambda21...
                         ,'s_maxIter',obj.s_maxIterSEMInit,'s_rho',obj.s_rho,'s_tol',obj.s_tolSEM,'v_initGraphSignal',mean(m_initialGraphSignal,2));
                    [m_adjacencyNew,m_timeAdjacencyNew]=lSVARGraphEstimator.estimate(mean(t_graphFunctionEstimateNew,3));
         

                    
                    v_errAdj(s_iter)=norm(m_adjacency-m_adjacencyNew,'fro');
                    v_errTimeAdj(s_iter)=norm(m_timeAdjacency-m_timeAdjacencyNew,'fro');

                    v_errFun(s_iter)=frobenious(obj,t_graphFunctionEstimate,t_graphFunctionEstimateNew);

                    if v_errAdj(s_iter) <= obj.s_tolJoint && v_errFun(s_iter) <= obj.s_tolJoint  && v_errTimeAdj(s_iter)<= obj.s_tolJoint
                        s_flagConver = 1;
                    end
                    t_graphFunctionEstimate=t_graphFunctionEstimateNew;
                    m_adjacency=m_adjacencyNew;
                    m_timeAdjacency=m_timeAdjacencyNew;
                end
            end
            if obj.b_plot==1
               disp(['Joint LSVAR ADMM finished after ',num2str(s_iter),' iterations']);
                figure;
                hold all;
                plot(v_errAdj(1:s_iter-1));
                plot(v_errTimeAdj(1:s_iter-1));
                plot(v_errFun(1:s_iter-1));
                legend({'errAdj','errTimeAdj','errFun'});
            end
            
            
        end
        function s=frobenious(obj,m,t)
        s=0;
            for s_it=1:size(m,3)
            s=s+norm(m(:,:,s_it)-t(:,:,s_it),'fro');
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
