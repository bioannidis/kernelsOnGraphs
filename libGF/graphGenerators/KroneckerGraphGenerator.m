classdef KroneckerGraphGenerator < GraphGenerator
    
    
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
        s_kronNum; %number of kronprod
        m_seedInit; % initial seed matrix
    end
    
    
    methods
        function obj = KroneckerGraphGenerator(varargin)
            obj@GraphGenerator(varargin{:});
        end
        
        % call this method to learn graph from signals such that these
        % signals are smooth on this graph
        % Graph object is returned
        function graph = realization(obj)
            m_seedIn=obj.m_seedInit;
            m_seed=m_seedIn;
            s_krNum=obj.s_kronNum;
            for level=1:s_krNum-1
                m_seed=kron(m_seedIn,m_seed);
            end
            
            
            
            
            m_adjacency=binornd(ones(size(m_seed)),m_seed);    % adjacency matrix
            m_adjacency=m_adjacency-diag(diag(m_adjacency));
            m_adjacency=m_adjacency+m_adjacency';
            m_adjacency=logical(m_adjacency);
            graph=Graph('m_adjacency',m_adjacency);
            v_ind=graph.getComponents{1};
            m_adjacency=m_adjacency(v_ind,v_ind);
            graph=Graph('m_adjacency',m_adjacency);
            %A=S.*randn(Nv,Nv);                                      % edge weights
            %A=A-diag(diag(A));
            
            figure(1)
            imagesc(m_adjacency)
            
        end
        
    end
    
    
    methods (Static)
        
    end
end
