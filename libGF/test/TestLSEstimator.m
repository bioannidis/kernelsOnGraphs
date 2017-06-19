classdef TestLSEstimator < matlab.unittest.TestCase
    % Tests to be performed:
    %   1. illegal parameter: m_samples and m_positions not the same size
    %   2. illegal parameter: NaN values in paramter
    %   3. not engouh information: bandwidth not set
    %   4. not engouh information: graph not valid
    %   5. not consistent: m_samples and graph not the same size
    %   6. correct output: given graph, bandwidth, and parameter, valid
    %      corresponding output
    %   7. vector handling: given parameters, if bandwidth is set to be a
    %   vector, then perform estimation with each bandwith
    
    properties
        LSEstimator
        graph
        bandwidth = 5;
        N = 10;     % number of vertices
        p = 0.3;    % edge probability
        graphSignal
    end
    
    methods(TestMethodSetup)
        function createGraph(tc)
            ERGraphGenerator = ErdosRenyiGraphGenerator();
            ERGraphGenerator.s_numberOfVertices = tc.N;
            ERGraphGenerator.s_edgeProbability = tc.p;
            tc.graph = ERGraphGenerator.realization();
        end
        
        function createGraphSignal(tc)
            V = tc.graph.getLaplacianEigenvectors();
            alpha = zeros(tc.N,1);
            alpha(1:tc.bandwidth) = sort(rand(tc.bandwidth,1));
            tc.graphSignal = V*alpha;
        end
              
        function createLSEstimator(tc)
            tc.LSEstimator = LSGraphFunctionEstimator();
            tc.LSEstimator.graph = tc.graph;
            tc.LSEstimator.s_bandwidth = tc.bandwidth;
        end
    end
    
    methods(Test)
        function testGraphNotEmpty(tc)
            tc.verifyNotEmpty(tc.graph);
        end       
        function testGraphSize(tc)
            tc.verifySize(tc.graph.m_adjacency, [tc.N tc.N]);
        end
        function testGraphSignalBandwidth(tc)
            tc.assertNotEmpty(tc.graphSignal);
            V = tc.graph.getLaplacianEigenvectors();
            alpha = V'*tc.graphSignal;
            tc.verifyEqual(alpha(tc.bandwidth+1:end), ...
                zeros(tc.N-tc.bandwidth,1),'abstol',1e-5);
        end
        
        function testIllegalParameterSize(tc)
            m_positions = rand(tc.N,1);
            m_samples = rand(tc.N-1,1);
            tc.verifyError(@() tc.LSEstimator.estimate(m_samples, m_positions),...
                'LSEstimator:IllegalParameter');
        end
        
        function testIllegalParamterSet(tc)
            m_positions = rand(tc.N,1);
            m_samples = rand(tc.N-1,1);
            tc.LSEstimator.s_bandwidth = [];
            tc.verifyError(@() tc.LSEstimator.estimate(m_samples, m_positions), ...
                'LSEstimator:ParameterNotSet');
        end
        
        % m_positions and graph are inconsistent
        function testSampleGraphInconsistent(tc)
            m_samples = rand(tc.N,1);
            m_positions = randi(tc.N, tc.N, 1);
            m_positions(1) = tc.N + 3;
            tc.verifyError(@() tc.LSEstimator.estimate(m_samples, m_positions),...
                'LSEstimator:SampleGraphInconsistent');
        end
        
        % test correct output
        function testOutput(tc)
            V = tc.LSEstimator.graph.getLaplacianEigenvectors();
            B = tc.LSEstimator.s_bandwidth;
            VB = V(:,1:B);
            
            S = floor(tc.N*2/3);
            m_positions = randperm(tc.N,S);
            m_positions = m_positions(:);
            m_samples = rand(S,1);
            I = eye(tc.N);
            Phi = I(m_positions,:);
            
            tc.verifyEqual(tc.LSEstimator.estimate(m_samples, m_positions),...
                VB*( (Phi*VB)\m_samples) );
        end
        
        % test vector input handling
        function testVectorInput(tc)
            V = tc.LSEstimator.graph.getLaplacianEigenvectors();
            B = tc.LSEstimator.s_bandwidth;
            VB = V(:,1:B);
            
            S = floor(tc.N*2/3);
            COL = 5;
            for iCol = 1 : COL
                pos = randperm(tc.N,S); 
                m_positions(:,iCol) = pos(:);
                m_samples(:,iCol) = rand(S,1);
                I = eye(tc.N);
                Phi = I(m_positions(:,iCol),:);
                fhat(:,iCol) = VB*( (Phi*VB)\m_samples(:,iCol));
            end
            
            tc.verifyEqual(tc.LSEstimator.estimate(m_samples, m_positions),...
                fhat );
        end
    end
    
end

