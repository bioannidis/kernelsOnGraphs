function NMSE = Simulate( generator, sampler, estimator, MONTE_CARLO, testSetOnly)
% SIMULATE      simulate function estiamtion on graph
% Input:
%       generator       graph function generator matrix
%       sampler         graph function sampler matrix
%       estimator       graph function estiamtor matrix
%       MONTE_CARLO     number of Monte Carlo simulations
% Output:
%       NMSE            normalized mean squared error, whose size is
%                       consistent with generator/sampler/estimator
%
if nargin < 5
	testSetOnly = 0;
end

DEBUG = true;

% get the dimension of result
ROW = max( [size(generator,1), size(sampler,1), size(estimator,1)] );
COL = max( [size(generator,2), size(sampler,2), size(estimator,2)] );

% simulation
NMSE = NaN(ROW,COL);
for iRow = 1 : ROW
    for iCol = 1 : COL
        NMSE(iRow, iCol) = MonteCarloSimulation( ...
            getObjectFromMat(generator,iRow, iCol), ...
            getObjectFromMat(sampler,iRow,iCol), ...
            getObjectFromMat(estimator, iRow, iCol), ...
            MONTE_CARLO, testSetOnly);

        if DEBUG
            fprintf('Simulation progress\t%3.1f%%\n', ...
                100*(iCol+(iRow-1)*COL)/(ROW*COL) );
        end
    end
end

end

function NMSE = MonteCarloSimulation( generator, sampler, estimator, MONTE_CARLO, testSetOnly )
% Monte Carlo simulation

%parfor iMonte = 1:MONTE_CARLO
%m_graphFunction = generator.realization();
if nargin < 5
	testSetOnly = 0;
end

if testSetOnly
	v_nmse = NaN(MONTE_CARLO,1);
    for iMonte = 1:MONTE_CARLO
        m_graphFunction = generator.realization();
        [m_samples, m_positions] = sampler.sample(m_graphFunction);
        m_estimate = estimator.estimate(m_samples, m_positions);
        testSetIndex = true(size(m_graphFunction));
        testSetIndex(m_positions) = false;
        v_nmse(iMonte) = norm(m_estimate(testSetIndex) - m_graphFunction(testSetIndex))^2 / ...
            norm(m_graphFunction(testSetIndex))^2;
	end
	NMSE = mean(v_nmse);
else
	v_nmse_num = NaN(MONTE_CARLO,1);
	v_nmse_den = NaN(MONTE_CARLO,1);
    parfor iMonte = 1:MONTE_CARLO
        m_graphFunction = generator.realization();
        [m_samples, m_positions] = sampler.sample(m_graphFunction);
        m_estimate = estimator.estimate(m_samples, m_positions);
        %v_nmse(iMonte) = norm(m_estimate - m_graphFunction)^2 / norm(m_graphFunction)^2;
		v_nmse_num(iMonte) = norm(m_estimate - m_graphFunction)^2;
		v_nmse_den(iMonte) = norm(m_graphFunction)^2;
	end
	NMSE = mean(v_nmse_num)/mean(v_nmse_den);
end



end

function obj = getObjectFromMat(objMat, row, col)
% this function is used to retrieve the obj (generator/sampler/estiamtor)
% from replicated matrix by specifiying row and col.
% If there is only 
[M,N] = size(objMat);

if M == 1 && N == 1
    % only one object in objMat
    % always return this object
    obj = objMat;
elseif M == 1
    % objMat is a row vector, only column index matters
    obj = objMat(col);
elseif N == 1
    % objMat is a column vector, only row index matters
    obj = objMat(row);
else
    % objMat is a non-trivial matrix, return corresponding element
    obj = objMat(row, col);
end

end

