function figureKKF(onlyplot,fnum_in)
%% This function generates the figures in the paper
% "Kernel-based Reconstruction  of Space-time Functions on Dynamic Graphs"
% by Daniel Romero, and Vassilis N. Ioannidis, and Georgios B. Giannakis.
% For each figure a different configuration of defaultFigureIndex
% and niter is needed

% This is the master file.
%
%    Input:
%        ONLYPLOT   0: Simulate
%                   1: Display results (previously computed)
%        FNUM_IN    index of the figure within the function file pointed by FS below        
%

% Initializations
addpath('./libGF/');
initializeSimGF;
%initCvx;
assert(nargin>=1,'Not enough arguments');

% Figure options (EDIT)
global plot_f_settings
plot_f_settings.docked_on=0; 
plot_f_settings.title_to_caption=1;
plot_f_settings.saveplots=1;  % write figures to files
plot_f_settings.pdf_flag=1;   % write figures to pdf
plot_f_settings.bw_maxpoints=20; 
plot_f_settings.figbasename='';  % if ~='' then figures are stored in the specified location,
                                 % else, the default folder is used. It must end with '/'.
%plot_f_settings.pos=[100.0000  402.0000  0.75*[592.3077  363.4615]]; % good for papers
plot_f_settings.pos=[[100.0000  402.0000]  1.5*[592.3077  363.4615]];
global chars_per_line
chars_per_line = 80;


%% Execution options (EDIT)
%% figure 2

defaultFigureIndex = 2018;
niter =1;

%% figure 3

defaultFigureIndex = 2019;
niter =10;

%% figure 4

% defaultFigureIndex = 2016;
% niter =10;

%% figure 5

% defaultFigureIndex = 2023;
% niter =10;

%% figure 6

% defaultFigureIndex = 5034;
% niter =200;

%% figure 7

% defaultFigureIndex = 3019;
% niter =100;

fs = KFonGSimulations;

% EXECUTION
if nargin < 2
	simt_compute(fs,defaultFigureIndex,niter,onlyplot)
else
	simt_compute(fs,fnum_in,niter,onlyplot)
end


end




function initCvx

cf = '../lib';
addpath([cf '/cvx'])
addpath([cf '/cvx/structures'])
addpath([cf '/cvx/lib'])
addpath([cf '/cvx/functions'])
addpath([cf '/cvx/commands'])
addpath([cf '/cvx/builtins'])


end

