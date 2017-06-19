classdef simFunctionSet
	%
	% An object of this class is a file which contains multiple functions
	% whose name starts by the string in funbasename.
	%
	% These functions return a variable F, which can either be the input 
	% for plot_f or they can be an object (possibly vector valued) of the 
	% class F_figure 
	%
	properties(Constant)
		funbasename = 'compute_fig_';
		
	end
		
	properties
		classname
		figbasename
	end
	
	methods(Access=public)
		
		function obj = simFunctionSet
			obj.classname=class(obj);
			obj.figbasename = [obj.folder obj.classname ];
		end
		
		
		
		function simt_compute(fs,fnum,niter,onlyplot)
						
			if ~onlyplot
				niter
			else
				niter=-1;
			end
			
			% -------------
			tic
			
			fs.invoke_functions(fnum,niter)
			
			tm = toc;
			% -------------
			fprintf('Elapsed time: %s\n',print_time(tm));
			
			% if (~onlyplot) && 0
			% 	beep; pause(1); beep; pause(1); beep;
			% end
			
		end

		
		
		function invoke_functions(obj,fig_vector,niter)
			% if niter == -1, then onlyplot is activated
			% otherwise, it is the number of iterations
			fprintf('starting %s\n',obj.classname);
			for k=1:length(fig_vector)
				obj.invoke_function(fig_vector(k),niter);
			end
		end
		
		function str = filename(obj,fnum)
			filebasename=[obj.folder obj.classname '_'];  % name and location for mat files
			str=sprintf('%s%d',filebasename,fnum);
			
		end
		
		function str = folder(obj)
			str = ['libGF/simulations/' obj.classname '_data/'];
		end
		
		function F = load_F_structure(obj,fig)
			matfile = obj.filename(fig)
			load(matfile,'F');
		end
				
		function update_F_structure(obj,fig,indx)
			% you first have to create an instance, not of this class, but
			% the class producing the simulations.
			%
			% this function modifies the .mat file to include in the F-structure the
			% changes made in the legend, the title and axis limits
			%
			% fig :  the code of the simulation, not the figure in which it is displayed
			% indx:  index of the figure within the F-struct array
			%
			%   We assume that this simulation is going to be displayed on figure( fig  + indx -1)
			% Eg: the second (indx = 2 ) figure of simulation 23 is displayed on figure 24.
			
			%classname = 'fig_part_5'; % name of the class that generated the figures	
			%matfile = sprintf('%s_data/%s_%d.mat', classname,classname,fig)
			
			matfile = obj.filename(fig);
			
			% b) figure options
			global plot_f_settings
			%saveplots figbasename bwplot pos bw_maxpoints docked_on pdf_flag;
			plot_f_settings.def_grid_on=1;
			plot_f_settings.docked_on=0;
			plot_f_settings.title_to_caption=0; % better to zero, cause otherwise we can erase the title
			plot_f_settings.saveplots=0;  % better to zero and save plots with simt. otherwise the name of the pdf may change when there are several figures, no?
			plot_f_settings.pdf_flag=1;
			plot_f_settings.bw_maxpoints=20;
			plot_f_settings.figbasename='';  % if ~='' then figures are stored in the specified location,
			plot_f_settings.figbasename= obj.figbasename;
			
			ff = figure(fig+indx -1);
			load(matfile)
			
			
			% legend
% 			if 0
% 				ll = legend;
% 				F(indx).leg = get(ll,'String');
% 			end
% 			
% 			% title
% 			if 0
% 				str = get(get(gca,'Title'),'String');
% 				F(indx).tit = str;
% 			end
% 			
% 			% axes limits
% 			if 0
% 				F(indx).xlimit = xlim;
% 				F(indx).ylimit = ylim;
% 			end
% 			
% 			% styles
% 			if 0
% 				curves = get(get(ff,'CurrentAxes'),'Children');
% 				ncurves = length(curves)
% 				for k = 1:ncurves
% 					styles{k} = get(curves(k),'LineStyle');
% 				end
% 				F(indx).styles = styles;
% 				F(indx).styles
% 			end
% 			%%
% 			
% 			% XLabel and YLabel
% 			if 0
% 				xl =get(get(get(ff,'CurrentAxes'),'XLabel'));
% 				F(indx).xlab = xl.String;
% 				yl =get(get(get(ff,'CurrentAxes'),'YLabel'));
% 				F(indx).ylab = yl.String;
% 			end
			
			if ~isfield(F(indx),'figfileismaster');
				F(indx).figfileismaster = 0;
			end
			fig_file_is_master = F(indx).figfileismaster;
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			keyboard
			% Editing the figure with the command line:
			% Edit the F-structure F(indx)
			%
			% Editing with the graphical tool:
			% set the variable "fig_file_is_master" to 1 if you want to
			% make the fig file have the master figure information. The F
			% structure will not be used. 
				
			% write "return" to exit
			
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			F(indx).figfileismaster = fig_file_is_master;
			if ~fig_file_is_master
				% if manual editting we show the results of editing
				plot_f(F,fig)
			else				
				disp('Master file is fig file');
			end
			
			s=input('do you want to save (y/n)??\n','s');
			if strcmp(s,'y')	
				save(matfile,'F');
				if fig_file_is_master 
					name = sprintf('%s_%d-%d',obj.figbasename,fig,indx);
					saveas(gcf,[name '.fig']);					
					plot_f(F,fig);
				end
				
				
			end
				
		
			
			
		end
		
		
		
	end
	
	
	
	methods(Access=private)
		
		function invoke_function(obj,fgnum,niter)
			
			% nothing to edit from here on
			global rnum tnum  
			global plot_f_settings
			%global filebasename funbasename
			rnum=0;    % results txt file number
			tnum=0;    % table number
			
			
			plot_f_settings.figbasename= obj.figbasename;
			
			
			if (~exist(obj.folder,'file'))
				mkdir(obj.folder);
			end
			
			funname =sprintf('%s%d',obj.funbasename,fgnum);
			if ~ismethod(obj,funname)
				error(sprintf('Simulation %d does not exist',fgnum));
			end
			
			if niter==-1
				% load results from previous executions
				if (~exist([obj.filename(fgnum) '.mat'],'file'))
					error(sprintf('You must run the simulation first. Use simt(0,%d)',fgnum));
				end
				load(obj.filename(fgnum),'F');
			else
				% call the function				
				F=feval(funname,obj,niter);
				%F=obj.compute_fig_1;
				% save results
				save(obj.filename(fgnum),'F');				
			end
			
			global running_on_server
			if running_on_server
				return
			end
			
			if isa(F,'F_figure')
				F.plot(fgnum);
			end
			
		end
		
% 		
% 		function name = figbasename(obj)
% 			
% 			folder=[obj.classname '_data/'];
% 			filebasename=[folder obj.classname '_'];  % name and location for mat files
% 			%if (isempty(figbasename))
% 			name = [folder obj.classname ];
% 			%else
% 			%	plot_f_settings.figbasename=[figbasename obj.classname ];
% 			%end
% 			
% 		end
% 		
% 		
		
		
	end
	
	
	
	
	
	
end

