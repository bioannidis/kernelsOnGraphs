classdef GraphLearningSmoothSignalGraphGenerator < GraphGenerator
	properties % required by parent classes
		c_parsToPrint  = {};
		c_stringToPrint  = {};
		c_patternToPrint = {};
	end
	
	properties(Constant)
		ch_name = 'Graph-Learning-For-Smooth-Signal';
	end
	
	properties
		m_observed; %contains the observed signals
		s_niter;    %maximum number of iterations
		s_alpha;
		s_beta; %positive regularization parameters0000......
		m_missingValuesIndicator;   %is a matrix with one if the corresponding entry in m_observed is
		%observed and 0 otherwise. If [] is used then assumed that all values observed
		m_constraintLaplacian; %indicator matrix if entry 1 the equivelant
		%entry in the Laplacian is constrained to be
		%zero must be offdiagonal element
		s_dont_estimate_the_signal=0;
		
	end
	methods
		function obj = GraphLearningSmoothSignalGraphGenerator(varargin)
			% Constructor
			obj@GraphGenerator(varargin{:});
			
		end
		function [graph,m_estimated,v_objective] = realization(obj)
			%initialization
			m_missingValuesIndicator=obj.m_missingValuesIndicator;
			m_constraintLaplacian=obj.m_constraintLaplacian;
			m_observed=obj.m_observed;
			s_alpha=obj.s_alpha;
			s_beta=obj.s_beta;
			s_niter=obj.s_niter;
			v_objective=0;
			if(isempty(m_missingValuesIndicator)) %#ok<*PROP>
				m_missingValuesIndicator=ones(size(m_observed));
			end
			if(isempty(m_constraintLaplacian))
				m_constraintLaplacian=zeros(size(m_observed,1));
			end
			m_estimated=randn(size(m_observed));
			%m_dublication converts the vectorized form of lower triangular
			%part of L to the vectorized form of the full matrix
			m_dubplication=GraphLearningSmoothSignalGraphGenerator.dup(rand(size(m_estimated,1)));
			
			if obj.s_dont_estimate_the_signal
				m_laplacian=GraphLearningSmoothSignalGraphGenerator.graphLaplUpd(s_alpha,s_beta,m_estimated,m_dubplication,m_constraintLaplacian);
			else
				%alternating optimization
				for t=1:s_niter
					m_laplacian=GraphLearningSmoothSignalGraphGenerator.graphLaplUpd(s_alpha,s_beta,m_estimated,m_dubplication,m_constraintLaplacian);
					m_laplacian(m_constraintLaplacian==1)=0;
					m_estimated=GraphLearningSmoothSignalGraphGenerator.signalUpd(m_observed,m_laplacian,s_alpha,m_missingValuesIndicator);
					%contains the objective values of the function
					v_objective(t)=(norm(m_missingValuesIndicator.*(m_observed-m_estimated),'fro')^2)+s_alpha*trace((m_estimated)'*m_laplacian*(m_estimated))+s_beta*norm(m_laplacian,'fro')^2;
					if(t>1)
						funcDecrease=-v_objective(t)+v_objective(t-1)
					end
					if t>1&&(0<(-v_objective(t)+v_objective(t-1)))&&((-v_objective(t)+v_objective(t-1))<10^-6)
						break;
					end
				end
			end
			%             figure(1);
			%             semilogy(1:t,v_obj);
			%keep only meaningful values
			%Prune insignificant edges
			s_alpha=(m_laplacian>-10^-4);
			s_beta=m_laplacian>10^-4;
			s_alpha=~s_alpha;
			c=s_alpha|s_beta;
			m_laplacian(~c)=0;
			
			
			m_adjacency=Graph.createAdjacencyFromLaplacian(m_laplacian);
			graph = Graph('m_adjacency',m_adjacency);
		end
	end
	methods (Static)
		%f
		function m_laplacian=graphLaplUpd(s_alpha,s_beta,m_estimated,m_duplication,m_constraintLaplacian)
			%arg min a*vec(Y*Y')'*Mdup*vech(L)+b*vech(L)'*Mdup'*Mdup*vech(L)
			%wrt vech(L)
			%st A*vech(L)=0;
			%   B*vech(L)<=0;
			k=size(m_estimated,1);
			%vectorize the constraints
			vh_constraints=GraphLearningSmoothSignalGraphGenerator.vech(m_constraintLaplacian);
			%Is used for the equality constraints of the opt problem
			m_A=GraphLearningSmoothSignalGraphGenerator.getLinEqualities(rand(size(m_estimated,1)));
			m_B=GraphLearningSmoothSignalGraphGenerator.getLinInequalities(rand(size(m_estimated,1)));
			s_n=size(GraphLearningSmoothSignalGraphGenerator.vech(rand(size(m_estimated,1))),1);
			cvx_begin quiet
			    variable v(s_n)
			minimize( s_alpha*vec((m_estimated)*(m_estimated)')'*m_duplication*v+s_beta*v'*(m_duplication')*m_duplication*v)
			subject to
			    m_A*v==[k;zeros(size(m_estimated,1),1)];
			    m_B*v<=0;
			    vh_constraints.*v==0;
			cvx_end
			m_laplacian=GraphLearningSmoothSignalGraphGenerator.my_ivech(v,m_duplication);
		end
		
		function v=getDiagIndices(n)
			v=zeros(n,1);
			for i=1:n
				if i==1
					v(i)=1;
				else
					v(i)=i+v(i-1);
				end
			end
		end
		
		function m_A=getLinEqualities(m_laplacian)
			X=GraphLearningSmoothSignalGraphGenerator.vech(m_laplacian);
			% %A contais the following info:
			% % tr(L)=n
			% % diag elements are in the positions prev_diag+diag_index
			v=GraphLearningSmoothSignalGraphGenerator.getDiagIndices(size(m_laplacian,1));
			indic=zeros(size(X,1),1);
			indic(v)=1;
			indic=indic';
			% %indic contains the tr(L)
			% % L*1=0
			
			m_A=zeros(size(m_laplacian,1)+1,size(X,1));
			v=v';
			for i=1:size(m_laplacian,1)
				if(i~=1)
					ind=[v(i)-(0:i-1),v(i:end-1)+i];
				else
					ind=horzcat(v(i),v(i:end-1)+i);
				end
				m_A(i+1,ind)=1;
			end
			m_A(1,:)=indic;
			%A*X
		end
		
		function m_B=getLinInequalities(m_laplacian)
			%B contains info about the Lij <= of zero
			%so must contain a line for each of these elements of X
			X=GraphLearningSmoothSignalGraphGenerator.vech(m_laplacian);
			v=GraphLearningSmoothSignalGraphGenerator.getDiagIndices(size(m_laplacian,1));
			m_B=zeros(size(X,1)-size(m_laplacian,1),size(X,1));
			for i=1:size(X,1)
				if ismember(i,v)
					i=i-1;
				else
					m_B(i,i)=1;
				end
			end
			%B*X<=0
		end
		
		function m_estimated=signalUpd(m_H,m_laplacian,s_alpha,m_w)
			%min norm(W*(X-Y),'fro')^2+a*tr(Y'*L*Y)
			%wrt Y
			%convex has closed form solution
			m_estimated=zeros(size(m_H));
			if m_w==ones(size(m_w))
				m_estimated=(eye(size(m_laplacian))+s_alpha*m_laplacian)\m_H;
			else
				for m=1: size(m_estimated,2)
					m_estimated(:,m)=(diag(m_w(:,1))+s_alpha*m_laplacian)\(diag(m_w(:,1))*m_H(:,m));
				end
			end
		end
		function L=my_ivech(x,M)
			K2=size(x,1);
			K=(-1+sqrt(1+8*K2))/2;
			d=GraphLearningSmoothSignalGraphGenerator.getDiagIndices(K);
			vec=M*x;
			L=reshape(vec,K,K);
		end
		
		function m_matrixData=ivech(v_stackedData)
			%converts the vectorized form of lower triangular part of a matrix back
			%to the initial matrix
			if size(v_stackedData,2)>size(v_stackedData,1)
				v_stackedData=v_stackedData';
			end
			
			if size(v_stackedData,2)~=1
				error('STACKED_DATA must be a column vector.')
			end
			
			K2=size(v_stackedData,1);
			K=(-1+sqrt(1+8*K2))/2;
			
			if floor(K)~=K
				error(['The number of elemeents in STACKED_DATA must be conformable to' ...
					'the inverse vech operation.'])
			end
			% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			% % %Input Checking
			% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			% % %Initialize the output data
			m_matrixData=zeros(K);
			
			% % %Use a logical trick to inverse the vech
			pl=tril(true(K));
			m_matrixData(pl)=v_stackedData;
			diag_matrixData=diag(diag(m_matrixData));
			m_matrixData=m_matrixData+m_matrixData'-diag_matrixData;
		end
		
		function m_dublication=dup(m_laplacian)
			%converts the vectorized form of lower triangular
			%part of L to the vectorized form of the full matrix
			%M*vech(L)=vec(L)
			
			X=GraphLearningSmoothSignalGraphGenerator.vech(m_laplacian);
			m_dublication=zeros(size(m_laplacian,1)^2,size(X,1));
			for i=1:size(m_laplacian,1)
				if i==1
					v(i)=1;
				else
					v(i)=i+v(i-1);
				end
			end
			for i=1:size(m_laplacian,1)
				if(i~=1)
					ind=[v(i)-(0:i-1),v(i:end-1)+i];
				else
					ind=horzcat(v(i),v(i:end-1)+i);
				end
				ind=sort(ind);
				for j=1:size(m_laplacian,1)
					m_dublication(j+(i-1)*size(m_laplacian,1),ind(j))=1;
				end
			end
		end
		
		function v_vech=vech(m_X)
			%contains the vectorized form of lower triangular part of m_X
			m_Y=m_X';
			v_vech= m_Y(triu(true(size(m_Y))));
		end
		
		function v_vec=vec(m_X)
			%contains the vectorized form of m_X
			[m,n] = size(m_X);
			v_vec = reshape(m_X,m*n,1);
		end
	end
end
