classdef TimeVaryingGraphGenerator < GraphGenerator
	properties % required by parent classes
		c_parsToPrint  = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	
	properties(Constant)
		ch_name = 'TimeVaryingGraphGenerator';
	end
	
	properties
		m_adjacency; % contains the initial adjacency
		s_maximumTime; %contains the maximum time to gen adjacencies
		s_timePeriodCh; %contains the period that the connectivity changes
        s_timePeriodDel; %contains the period that the connectivity deletes
		
	end
	methods
		function obj = TimeVaryingGraphGenerator(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
        end
        function t_adj=generateAdjForDifferentTBasedOnProb(obj,s_probabilityOfEdgeChange,s_mean,s_std)
            %s_probabilityOfEdgeChange; contains the propability by which an edge will change
            %s_mean, s_std; mean std of the random change to the adjentry
            m_adj=obj.m_adjacency;
            s_maxT=obj.s_maximumTime;
            s_timP=obj.s_timePeriodCh;
            m_lastAdj=m_adj;
            for s_timeInd=1:s_maxT
                t_adj(:,:,s_timeInd)=m_lastAdj;
                if mod(s_timeInd,s_timP)==0
                %modify adjacency
                m_ran=rand(size(m_adj));
                m_ran=m_ran<s_probabilityOfEdgeChange;
                m_change=s_mean+s_std*randn(size(m_adj));
                m_lastAdj=m_lastAdj+m_ran.*m_change;
                m_lastAdj=pos(m_lastAdj);
                m_lastAdj=m_lastAdj-diag(diag(m_lastAdj));
                m_lastAdj=1/2*(m_lastAdj+m_lastAdj');
                end
            end
        end
         function t_adj=generateAdjForDifferentTBasedOnDifProbWithDel(obj,s_probabilityOfEdgeDelet,s_mean,s_std)
            %s_probabilityOfEdgeChange; contains the propability by which an edge will change
            %s_mean, s_std; mean std of the random change to the adjentry
            m_adj=obj.m_adjacency;
            s_maxT=obj.s_maximumTime;
            s_timP=obj.s_timePeriodCh;
            s_timDel=obj.s_timePeriodDel;

            m_lastAdj=m_adj;
            for s_timeInd=1:s_maxT
                t_adj(:,:,s_timeInd)=m_lastAdj;
                m_prob= TimeVaryingGraphGenerator.constructProb(m_lastAdj);
                if mod(s_timeInd,s_timP)==0
                %modify adjacency
                m_ran=rand(size(m_adj));
                m_ran=m_ran<m_prob;
                m_change=s_mean+s_std*randn(size(m_adj));
                m_lastAdj=m_lastAdj+m_ran.*abs(m_change);
                m_lastAdj=pos(m_lastAdj);
                m_lastAdj=m_lastAdj-diag(diag(m_lastAdj));
                m_lastAdj=1/2*(m_lastAdj+m_lastAdj');
                end
                if mod(s_timeInd,s_timDel)==0
                %modify adjacency
                m_ran=rand(size(m_adj));
                m_ran=~(m_ran<s_probabilityOfEdgeDelet);
                m_ran=m_ran+m_ran';
                m_ran=(m_ran~=0);
                m_lastAdj=m_lastAdj.*m_ran;
                end
            end
        end
		function [graph] = realization(obj)
			%initialization
			m_adj=obj.m_adjacency;
			m_train=obj.m_training_data;
            s_maxT=obj.s_maximumTime;
            s_timP=obj.s_timePeriodCh;
            s_probEd=obj.s_probabilityOfEdgeChange;
			%sparsity = sum(m_adj(:))/(numel(m_adj)-size(m_adj,1))
            error('Not implemented');
		end
	end
	methods (Static)
		function m_covInv = learnInverseCov( m_sampleCov , m_adjacency )
			% Learns the inverse covariance of a normal distribution
			% m_adjacency is optional. m_covInv is such
			% that m_covInv(i,j) = 0 if m_covInv(i,j) = 0  (i~=j)If given, then
			%
			
			d = size(m_sampleCov,1);
			m_adjacency = m_adjacency + triu(ones(d));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			variable S(d,d) symmetric
			minimize( -log_det(S) +trace(S*m_sampleCov) )
			subject to
			S(m_mask) == 0;
			cvx_end
			
			m_covInv = S;
			
		end
        function m_prob= constructProb(m_adj)
            sumWeights=sum(sum(m_adj));
            m_prob=zeros(size(m_adj));
            for s_adjind1=1:size(m_adj,1)
                for s_adjind2=1:size(m_adj,1)
                    if s_adjind1~=s_adjind2
                        m_prob(s_adjind1,s_adjind2)=(sum(m_adj(s_adjind1,:))+sum(m_adj(:,s_adjind2)))/sumWeights;
                    end
                end
            end
        end
		function m_laplacian = approximateWithLaplacian(m_input,m_adjacency)
			% m_laplacian is the best Laplacian matrix approximating matrix
			% m_input in the Frobenius norm
			% m_adjacency is an optional parameter. m_laplacian is such
			% that m_laplacian(i,j) = 0 if m_adjacency(i,j) = 0  (i~=j)
			%
			s_nodeNum = size(m_input,1);
			if nargin<2
				m_adjacency = ones(s_nodeNum);
			end
			m_adjacency = m_adjacency + triu(ones(s_nodeNum));
			m_mask = (m_adjacency == 0);
			
			cvx_begin
			variable L(s_nodeNum,s_nodeNum) symmetric
			minimize( norm(L - m_input,'fro') )
			subject to
			L*ones(s_nodeNum,1) == zeros(s_nodeNum,1);
			triu(L,1) <= 0;
			L(m_mask) == 0;
			cvx_end
			
			m_laplacian = L;
		end
	end
end
