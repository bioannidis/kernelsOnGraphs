function simpleSimulation

% initializeSimGF;

% 1. define graph function generator
graphGenerator = ErdosRenyiGraphGenerator('s_edgeProbability', 0.3,'s_numberOfVertices',100);
graph = graphGenerator.realization;
functionGenerator = BandlimitedGraphFunctionGenerator('graph',graph,'s_bandwidth',30);

% 2. define graph function sampler
sampler = UniformGraphFunctionSampler('s_numberOfSamples',40,'s_SNR',20);

% 3. define graph function estimator
estimator = BandlimitedGraphFunctionEstimator('m_laplacianEigenvectors',functionGenerator.basis);

% Simulation
m_graphFunction = functionGenerator.realization();
[m_samples,m_positions] = sampler.sample(m_graphFunction);
m_graphFunctionEstimate = estimator.estimate(m_samples,m_positions);

% Performance assessment 
error = norm(m_graphFunctionEstimate - m_graphFunction,'fro')^2/size(m_graphFunction,1)

end