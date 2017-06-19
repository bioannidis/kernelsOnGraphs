classdef SmoothSignalGraphGenerator < GraphGenerator
    % SmoothSignalGraphGenerator     learning graph laplacian from graph
    % signals so that these signals are smooth on the learned graph
    %
    % Reference: Dong et al. "Learning Laplacian Matrix in Smooth Graph
    %            Signal Representation", arXiv:1406.7842v3, Feb 2016.
    %
    % Meng MA
    % SPiNCOM, DTC
    % University of Minnesota
    % Last Modified: April 20, 2016
    
    properties % required by parent classes
        c_parsToPrint  = {};
        c_stringToPrint  = {};
        c_patternToPrint = {};
    end
    
    properties(Constant)
        ch_name = 'Learn Graph from signals';
		DEBUG = true; % set this paramter to true for debug information
        CVX = true;  % set CVX=true to use cvx for solving this problem
    end
    
    properties
        m_observed;             % observed signals on some graph
        s_niter = 1000;       % maximum number of iterations
		s_alpha = 1;            % regularization parameter: alpha*tr(Y'LY)
        s_beta = 100;           % regularization: beta * norm(Y,'fro')^2
    end
    
    
    methods
        function obj = SmoothSignalGraphGenerator(varargin)
            obj@GraphGenerator(varargin{:});     
		end
        
		% call this method to learn graph from signals such that these
		% signals are smooth on this graph
		% Graph object is returned
        function graph = realization(obj)
            L = obj.learnLaplacian();  
            m_adjacency=Graph.createAdjacencyFromLaplacian(L);
            graph = Graph('m_adjacency',m_adjacency);
		end
		
		function m_laplacian = learnLaplacian(obj)
            % learning laplacian matrix L by alternating minimization
			%   First, fix Y minimize L
            % 
            N = size(obj.m_observed,1); % number of nodes
            X = obj.m_observed;         % observed signals, in columns
			Y = X;                      % initialize Y = X
            M = obj.getMdup(N);         % M duplication matrix: M*vech(L) = vec(L)
            m_A = obj.getA(N);
            m_B = obj.getB(N);
            lengthVech = N*(N+1)/2;
            
            % for quadratic programming
            H = obj.s_beta/2 * (M'*M);
            A = m_B; b = zeros(N*(N-1)/2,1);
            Aeq = m_A; beq = zeros(N+1,1); beq(end) = N;
			
            %history = NaN(obj.s_maxIter, 1);    % record objective value
			for iter = 1 : obj.s_niter
                Y_old = Y;
                
                % 1. minimization wrt L
                if obj.CVX  % use CVX to solve L minimization problem
                    cvx_begin quiet
                    variable vech_L(lengthVech)
                    minimize( obj.s_alpha * vec(Y*Y')' * M * vech_L + ...
                        obj.s_beta * vech_L' * (M'*M) * vech_L )
                    subject to
                    m_A * vech_L == [zeros(N,1); N];
                    m_B * vech_L <= 0;
                    cvx_end
                else
                    % update quadratic programming paramter
                    W = Y*Y'; w = W(:);
                    f = obj.s_alpha*M'*w;
                    vech_L = quadprog(H,f,A,b,Aeq,beq);
                end
				
                % 2. minimization wrt Y
				m_laplacian = obj.unvech(vech_L);
				Y = (eye(size(m_laplacian))+obj.s_alpha*m_laplacian) \ X;
                
                % print debug information
                history(iter) = norm(Y - Y_old, 'fro')/norm(Y_old,'fro');
                if obj.DEBUG
					fprintf('Iterations:%2d \tobjective value: %f\n',...
						iter, history(iter) );
				end
                
				if history(iter) < 1e-6
					break;
                end
			end
            % if DEBUG = true, plot the history of objective value
			% set DEBUg = false to suppress this output
			if obj.DEBUG && length(history) > 1
				figure(101)
				plot(history,'linewidth',2)
				xlabel('iterations')
				ylabel('||Y_{t+1} - Y_t||')
				title('objective value history')
                grid on
			end
        end
    end
    
    
    methods (Static)   
        function m_A = getA(N)
			% get matrix A which take care of equality constraints
			% The last row of A takes care of tr(L) = N
			% while the first N rows take care of L*1 = 0
			% Algorithm here is to first find the index of every entry of L
			% in vech(L), then set the corresponding rows.
			%
			% Example. When N = 3, the index matrix is 
			%     1  2  3
			%     2  4  5
			%     3  5  6
			%    where each number identify the position of that entry in
			%    vech(L). (2,3) = 5 means L(2,3) is located in vech(L)(5).
            m_A = zeros(N+1, N*(N+1)/2);
            v_diagIndex = SmoothSignalGraphGenerator.getDiagIndexInVech(N);
			indexL = zeros(N,N);
			for row = 1 : N
                indexL(row,row:N) = v_diagIndex(row) : v_diagIndex(row) + N-row;
                indexL(row:N, row) = v_diagIndex(row) : v_diagIndex(row) + N-row ;
			end
			
			for row = 1 : N
				m_A(row, indexL(row,:)) = 1;
			end
            m_A(N+1,v_diagIndex) = 1;
        end
        
        function m_B = getB(N)
            % get matrix B which takes care of inequality constraint
            % pick all the off-diagonal entries from vech(L)
            m_B = eye(N*(N+1)/2);
            diagIndex = SmoothSignalGraphGenerator.getDiagIndexInVech(N);
            m_B(diagIndex, diagIndex) = 0;
            m_B( sum(m_B,2) == 0, : ) = [];
        end
        
        function v_diagIndex = getDiagIndexInVech(N)
            % for a graph of given size N
            % find the index of diagonal elements in vector vech(L)
            v_diagIndex = 1 + [0 cumsum(N:-1:2)];
        end
        
        function M = getMdup(s_laplacianSize)
            % generate duplication matrix M such that
            % M * vech(L) = vec(L)
            N = s_laplacianSize;
            %M = zeros( N^2, N*(N+1)/2 );
            yIndex = NaN(N^2,1);    % use sparse matrix to speedup
            
            % convert index of vec(L) into matrix coordinates
            % then switch row and columns to ensure it's in lower triangle
            % then convert matrix coordinates into index of vech(L)
            colIndex = [0 cumsum(N:-1:1)];
            for idx = 1 : N^2
                coord = [mod(idx-1, N)+1, floor((idx-1)/N)+1];
                row = max(coord);
                col = min(coord);
                idxh = colIndex(col) + row - col + 1;
                %M(idx, idxh) = 1;
                yIndex(idx) = idxh;
            end
            M = sparse( 1:N^2, yIndex, ones(N^2,1), N^2, N*(N+1)/2 );
        end
        
        function m_L = unvech(v_vechL)
            % converts the vectorized form of lower triangular part
            % of a matrix back to full matrix form
            v_vechL = v_vechL(:);        % ensure column vector
            len = length(v_vechL);
            
            N = (-1+sqrt(1+8*len))/2;    % find the size of matrix
            if floor(N) ~= N
                error('Not a valid vector, cannot unvec');
            end
            
            m_L = zeros(N,N);
            m_L(tril(true(N,N))) = v_vechL;
            m_diagonal = diag(diag(m_L));
            m_L = m_L + m_L' - m_diagonal;
        end
        
        function v_vech = vech(m_X)
            % convert only lower triangle part of L into a vector
            v_vech = m_X(tril(true(size(m_X))));
        end
    end
end