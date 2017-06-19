classdef F_figure
	%
	% A vector of objects F_figure is used to represent one figure per element of this vector. 
	%
	% Each object of this vector corresponds to a figure, with one or more axes (subplots). It may 
	% directly contain the figure in its fields 'X' and 'Y', or the subplots may be contained in the
	% field 'multiplot_array'. By default, a subplot is created with the same dimensions as the matrix
	% multiplot_array. If we want to write several plots on the same axes (with the "hold on" command)
	% then use the third dimension of multiplot_array.
	% 
	% Only the deepest level can have 3rd dim > 1. However, only the first element along the 3rd dimension
	% is considered for parameters other than 'X','Y' and 'Z'
	%
	% A Nfigs-length vector of objects of the class F_figure corresponds to Nfigs different figures
	
	properties(Constant)
		
	end
	
	
	properties
		fignum = 1; % number of the first figure		

		% Multiplot fields
		multiplot_type = 'subplot';
		            % this can be:
					%     'subplot' : a subplot is made with the same dimensions
					%               as the matrix of objects of class F_figure
					%               stored in multiplot_array
					%     'sequence' : each element of multiplot_array is represented sequentially
					%               and a pause command is executed between the display of
					%               two consecutive elements. Each element of the sequence
					%               can have subplots.
		
        multiplot_array;  % Matrix/vector of objects of class F_figure
						  % If multiplot_array is not empyt, then X, Y and Z should 
						  % be empty
		
		
		% Figure fields (the following variables are disregarded in multiplot mode)
		X = [];     % Ncurves x Npoints matrix with the X-coordinates/
		            % 1 x Npoints vector with the X coordinates 
	    Y = [];     % Ncurves x Npoints matrix with the Y-coordinates
		Z = [];     % similar to X and Y, but 
		
		xlimit = [];
		ylimit = [];
		
		EB = [];    % Ncurves x Npoints matrix with the semi-length of the error bars
		            % If an element is NaN, no error bar is represented for the corresponding
					% point. 
		
		% 3D plots
		caxis_vector =[];   % Color axis. This is the argument to the caxis command
		                    % if it is empty, the default [min max] is applied.
		view_from_top = 1;  % if ~= 0, the plot is viewed from the top -> only colors
		                    % but the Z axis becomes visible by manually rotating the
							% axes in the MATLAB figure.
		
		logx = 0;     % set to 1 to have a logarithmic x-axis
        logy = 0;     % set to 1 to have a logarithmic x-axis
							
		% text
		tit = '';
		tit_font_size = [];  
		xlab = '';
		ylab = '';
		zlab = '';
		leg = {};
		leg_pos = '';      % it can be 'northwest', etc
		leg_pos_vec = [];  % this must be a vector with four components. 
		                   % you may use the function get_leg_pos(fig)
						   % to get the vector that a figure is using
		
		% colors and styles
		colorset =[0 0 0;1 0 0;0 0 .9 ;0 .65 0; .9 0 .9 ;.5 .5 0;0 .7 .7;.5 0 1; 1 .5 0;  1 0 .5; 0 1 .5;0 1 0];
		colorp = 12;     % color period. the color of the curves repeat every colorp curves. At most it
		                 % can be size(colorset,1)
			  % NOTE: color can also be set in the style strings, e.g. '-w' makes a solid white line. 
			  % For plot3 this is the only possibility with the current implementation.
						 
		styles = {'-','-','-','-','-','-','-','-','-','-','-','-',...
			'--','--','--','--','--','--','--',...
			'-.','-.','-.','-.','-.','-.','-.',...
			':',':',':',':',':',':',':'};
		plot_type_2D = 'plot'; % other possibilities: 'stem' and 'bar'
		plot_type_3D = 'imagesc'; % other possibilities: 'plot3','surf' (make selection imagesc/surf automatic)
		
		
		gstyle = '--';           % style for the grid. Other values are '' and ':'
		pos = [];               % Set e.g. to 
		                        % [100.0000  402.0000  0.75*[592.3077  363.4615]];
		
		caption = [];            % if it is set, it is printed when representing and it is saved
		                         % to a txt file. If it is empty and the value of title_to_caption 
								 % is set different from 0, then its value is the title.
			
        % properties of the representation	-> delete, this should not be properties of F							 
        % title_to_caption = 0;	
		% saveplots = 0;
		%figbasename = '';
		%pdf_flag = 0;
		docked_on = 0;
		bw_maxpoints = 100;
		translation_table = {};  % if this is a N x 2 cell array of strings. we replace the
		                         % occurences of translation_table(i,1) by
								 % translation_table(i,2) for all i
								 %
								 % if it is N x 3, we replace translation_table(i,1) by
								 % translation_table(i,3) in the title and caption
								 % translation_table(i,2) in the legend, xlabel and ylabel
								 %
								 % The table used is the one corresponding to F(1)
		inhibit_pause = 0;       % if one, no pause commands are issued when displaying sequences (see above)
		figure;% figure object if not null then plot it
	end
	
	
	methods
		
		function obj = F_figure(varargin)
			%
			% function obj = F_figure(field,value,field,value...)
			%
			%
			%
			obj = obj.read_globals;
			if nargin>0
								
				if mod(length(varargin),2)~=0
					error('the number of arguments must be even');
				end
				
				for k=1:2:length(varargin)-1
					obj.(varargin{k})=varargin{k+1};
				end
								
			end
		end
		
		function obj = read_globals(obj)
			
			global plot_f_settings

			% parsing globals
			if ~isempty(plot_f_settings)
				if isfield(plot_f_settings,'gstyle')
					obj.gstyle = plot_f_settings.def_gstyle;
				end
								
				if isfield(plot_f_settings,'bw_maxpoints')
					obj.bw_maxpoints= plot_f_settings.bw_maxpoints;
				end
				%if isfield(plot_f_settings,'pdf_flag')
				%	obj.pdf_flag= plot_f_settings.pdf_flag;
				%end
				if isfield(plot_f_settings,'docked_on')
					obj.docked_on= plot_f_settings.docked_on;
				end
				if isempty(obj.translation_table)&&isfield(plot_f_settings,'translation_table')
					obj.translation_table= plot_f_settings.translation_table;
				end
				%if isfield(plot_f_settings,'title_to_caption')
				%	obj.title_to_caption = plot_f_settings.title_to_caption;
				%end
				%if isfield(plot_f_settings,'inhibit_pause')
				%	obj.inhibit_pause = plot_f_settings.inhibit_pause;
				%end
			end
				
		end
		
		function plot(F_vec,fgnum)		
			% F_vec : object to plot
			% fgnum : number of the first figure

			% number
			
			if nargin>1
				function_number = fgnum;
			else
				function_number = F_vec(1,1,1).fignum;
			end			
			
			% we change the vector F_vec to column vector but respecting the
			% third dimension
			if (size(F_vec,2)>1)
				if size(F_vec,1)>1
					error('F_vec must be a vector, not a matrix')
				end
				F_vec = permute(F_vec,[2 1 3]);
			end
	
			if ~isempty(F_vec(1,1,1).translation_table)
				F_vec = F_vec.translate;
			end
			
			for k=1:size(F_vec,1)
				
				F_now = F_vec(k,1,:);
				
				check_parameters(F_now(1,1,1));
				figure(function_number+k-1)
				if ~isempty(F_now.figure)
				plot(F_now.figure)
				else
				plot_scalar_F(F_now,function_number,k); % it may have a third dimension
	
				end
								
												
			end
			
			
		end
		
		function plot_scalar_F(F,function_number,fig_index)
			% F may have a third dimension. 
			% F corresponds to a figure.
			%
			% The name of the file where the figure is stored is
			%    figbasename_function_number-fig_index.pdf
			% where figbasename is given by the globals

			global plot_f_settings

			% parsing globals
			% plotting options
			
			if F.docked_on
				set(gcf,'WindowStyle','docked');
			else
				if (isempty(F.pos))&&(~isempty(plot_f_settings))&&isfield(plot_f_settings,'pos')
					posi= plot_f_settings.pos;					
				else
					posi = F.pos;
				end
				if ~isempty(posi)
					set(gcf,'position',posi);
				end
			end			
			%subplot(1,1,1)
			
			% subplots
			if isempty(F(1,1,1).multiplot_array)
				represent_axis(F);
				F(1,1,1).print_caption;				
				F(1,1,1).save_figure(function_number,fig_index);
			else
				switch(F(1,1,1).multiplot_type)
					case 'subplot'						
						nrows = size(F(1,1,1).multiplot_array,1);
						ncols = size(F(1,1,1).multiplot_array,2);
						for row = 1:nrows
							for col = 1:ncols
								subplot(nrows,ncols,(row-1)*ncols + col );
								represent_axis(F(1,1,1).multiplot_array(row,col,:));
							end
						end
						F(1,1,1).print_caption;
						F(1,1,1).save_figure(function_number,fig_index);
					case 'sequence'
						Fseq = F(1,1,1).multiplot_array(:);
						for k=1:length(Fseq)
% 							if ~isempty(Fseq(k).caption)
% 								fprintf('%s\n',Fseq(k).caption);
% 							end	
                            %Fseq(k).saveplots = 0;
							plot_scalar_F(Fseq(k),function_number,fig_index);
							
							%change the following line for one using globals
							%if ~F(1,1,1).inhibit_pause
								pause();
							%end							
						end
								
					otherwise
						error('F.multiplot_type does not contain a valid string');
				end
				
			end
		end
		
		function represent_axis(F)
			% representation of the curves
			
			for k=1:size(F,3)
%keyboard				
				if isempty(F(1,1,1).Z)
					represent_axis_2D(F(1,1,k));
				else
					represent_axis_3D(F(1,1,k));
				end
				hold on
			end
			hold off
			
			
			% axis
			if ~isempty(F(1,1,1).xlimit)
				xlim(F(1,1,1).xlimit);
			end
			if ~isempty(F(1,1,1).ylimit)
				ylim(F(1,1,1).ylimit);
			end
			
			% text	
			title(F(1,1,1).tit);

			if ~isempty(F(1,1,1).tit_font_size)
				set(get(gca,'Title'),'FontSize',F(1,1,1).tit_font_size);
			end

			xlabel(F(1,1,1).xlab);
			ylabel(F(1,1,1).ylab);
			zlabel(F(1,1,1).zlab);
			if ~isempty(F(1,1,1).leg)
				if ~isempty(F(1,1,1).leg_pos)
					legend(F(1,1,1).leg,'Location',F(1,1,1).leg_pos);
				else
					legend(F(1,1,1).leg);
				end
				if ~isempty(F(1,1,1).leg_pos_vec)
					if ~isempty(F(1,1,1).leg_pos)
						warning('ignoring F.leg_pos');
					end
					cv = get(gcf,'Children');
					set(cv(1),'Position',F(1,1,1).leg_pos_vec);
					
				end
				
			end
			
		end
		
		function represent_axis_3D(F)
			if isempty(F.Z)
				error('Z cannot be empty in 3D figures')
			end
			switch(F.plot_type_3D)
				% note that some of these figures may not display correctly in
				% pdf. This is a bug and can be found on export_fig documentation.
				% the same happens if we export through matlab standard commands.
				% A solution is to export to eps.
				case 'imagesc'
					pcolor(F.X,F.Y,F.Z);
					colormap(gcf,jet(100));
				case 'surf'
					surf(F.X,F.Y,F.Z);					
				case 'plot3'
					plot3(F.X,F.Y,F.Z,F.styles{1},'LineWidth',2);
			end
					
			if F.view_from_top 
				view([0 90]);
			end
			shading interp
			if ~isempty(F.caxis_vector)
				caxis(F.caxis_vector);
			end
			
		end
		
		
		function represent_axis_2D(F)
			ncurves = size(F.Y,1);
			assert(length(F.styles)>=ncurves);
			if isempty(F.X)
				F.X = 1:size(F.Y,2);
			end
			if size(F.X,1) ~= ncurves
				if size(F.X,1) == 1
					F.X = ones(ncurves,1)*F.X;
				else
					error('size(X,1) must be either 1 or size(Y,1)');
				end
			end
			
			for kr=1:ncurves
				assert(~isempty(F.styles{kr}));
				switch(F.plot_type_2D)
					case 'plot'
						if F.logx
							if F.logy
								loglog(F.X(kr,:),F.Y(kr,:),F.styles{kr},'LineWidth',2);
							else
								semilogx(F.X(kr,:),F.Y(kr,:),F.styles{kr},'LineWidth',2);
							end
						else
							if F.logy
								semilogy(F.X(kr,:),F.Y(kr,:),F.styles{kr},'LineWidth',2);
							else
								plot(F.X(kr,:),F.Y(kr,:),F.styles{kr},'LineWidth',2);
							end
						end
					case 'stem'
						stem(F.X(kr,:),F.Y(kr,:),F.styles{kr});
					case 'bar'
						bar(F.X(kr,:),F.Y(kr,:));
					otherwise
						error('unrecognized plot type')
				end
				hold on
			end
			hold off
			if strcmp(F.plot_type_2D,'bar')  % bar color cannot be set like plot objects
				return
			end
			% colors
			chldv=get(gca,'Children');
			for kr=1:size(F.Y,1)
				% the following line returns an error if we are trying to set the
				% color of something that is not a curve
				get(chldv(length(chldv)-kr+1),'XData');
				set(chldv(length(chldv)-kr+1),'Color',F.colorset(mod(kr-1,F.colorp)+1,:));
			end
			
			if ~isempty(F.gstyle)
				grid on
				set(gca,'GridLineStyle',F.gstyle);
			end
			
			if ~isempty(F.EB)
				F.plot_error_bars;
			end
			
			set(gcf,'PaperPositionMode','auto');
			
			
			
			
		end
		
		function plot_error_bars(F)
						
			hold on
			s_bar_width = (max(F.X(:))-min(F.X(:)))/100;  % same as in errorbar.m
			
			for m = 1:size(F.EB,1)
				vx = [];
				vy = [];
				for n = 1:size(F.EB,2)
					x = F.X(m,n);
					y = F.Y(m,n);
					slen = F.EB(m,n);
					vx_now = [x       x       NaN   x-s_bar_width   x+s_bar_width   NaN   x-s_bar_width   x+s_bar_width   NaN];   
					vy_now = [y+slen  y-slen  NaN   y+slen          y+slen          NaN   y-slen          y-slen          NaN];
					
					vx = [vx vx_now];
					vy = [vy vy_now];
				end
				plot(vx,vy,'-k')
			end
			
			hold off

			% colors set as the figures
			chldv=get(gca,'Children');
			for kr=1:size(F.Y,1)
				get(chldv(size(F.Y,1)-kr+1),'XData');
				set(chldv(size(F.Y,1)-kr+1),'Color',F.colorset(mod(kr-1,F.colorp)+1,:));
			end
			
		end
		
		
		function check_parameters(F)
			% this function is for scalar objets of class F_figure
			
			assert(numel(F)==1);
			if ~isempty(F.multiplot_array)
				assert(isempty(F.X)&&isempty(F.Y)&&isempty(F.Z));
			end
			
		end
		
		
		function print_caption( F )
			if ~isempty(F.caption)
				fprintf('%s\n',F.caption);
			end
		end
		
		function save_figure( F , function_number,fig_index)
			
			global plot_f_settings

			% parsing globals
			if ~isempty(plot_f_settings)
				if isfield(plot_f_settings,'saveplots')
					saveplots= plot_f_settings.saveplots;
				else
					saveplots = 0;
				end
				if isfield(plot_f_settings,'figbasename')
					figbasename= plot_f_settings.figbasename;
				end
				if isfield(plot_f_settings,'pdf_flag')
					pdf_flag= plot_f_settings.pdf_flag;
				end
				if isfield(plot_f_settings,'title_to_caption')
					title_to_caption = plot_f_settings.title_to_caption;
				end
			else
				% we do not support saving files without globals beind defined
				return
			end
			
			if ~saveplots
				return
			end
			
			name = sprintf('%s_%d-%d',figbasename,function_number,fig_index);

			set(gcf,'color',[1 1 1]);
			if pdf_flag
				%set(gcf,'color','none');				
				fprintf('Saving to %s\n', [name '.pdf']);
				%export_fig(name,'-pdf');				
				export_fig(name,'-pdf');
			else
				%print(fg,'-dpsc',sprintf('%sk%d',plot_f_settings.figbasename,fgnum));
				%print(fg,'-dpsc',name);
				fprintf('Saving to %s\n', [name '.eps']);
				export_fig(name,'-eps');
			end
			set(gcf,'color',0.8*[1.0 1.0 1.0])
			saveas(gcf,[name '.fig']);
			
			caption_text = F.caption;
			
			if  (title_to_caption)&&(isempty(caption_text))
				if isfield(F,'tit')
					caption_text = [F.tit '.'];
				else
					caption_text = '';
				end
			end
			if ~isempty(caption_text)
				caption_fn = [name '.txt'];
				fprintf('saving caption to %s\n',caption_fn);
				fdes=fopen(caption_fn,'w');
				%F.tit  = regexprep( ['$' tit '$'],',','$, $');  % chapuza
				%caption  = regexprep(  caption ,'=','$=$');  % chapuza
				%if ~isempty(caption)
				%	caption = [caption '.'];
				%end
				%caption
				fprintf(fdes,caption_text);
				fclose(fdes);
			end	
			
			
			
				
				
		
			
			
		end
		
		
		function F = translate_axis(F,translation_tab)
						
			% splitting a possible N x 3 table into two Nx2 tables
			if isempty( translation_tab )
				return;
			elseif size(translation_tab,2) == 2
				title_table = translation_tab;
				common_table = translation_tab;
			elseif size(translation_tab,2) == 3
				rows = size(translation_tab,1);
				title_table = cell(rows,2);
				common_table = cell(rows,2);
				for k=1:rows
					common_table{k,1} = translation_tab{k,1};
					common_table{k,2} = translation_tab{k,2};
					title_table{k,1} = translation_tab{k,1};
					title_table{k,2} = translation_tab{k,3};
				end
			elseif ~isempty(translation_tab)
				error('the translation table must have 2 or 3 columns');
			end
			
			%%%% translating
			
			if (~isempty(F.tit))
				F.tit = F_figure.translate_string(F.tit,title_table);
			end
			if (~isempty(F.xlab))
				F.xlab = F_figure.translate_string(F.xlab,common_table);
			end
			if (~isempty(F.ylab))
				F.ylab = F_figure.translate_string(F.ylab,common_table);
			end
			if (~isempty(F.leg))
				F.leg = F_figure.translate_string(F.leg,common_table);
			end				
			if (~isempty(F.caption))
				F.caption = F_figure.translate_string(F.caption,title_table);
			end	
		end
		
		function F_vec = translate(F_vec)
			
			tt = F_vec(1,1,1).translation_table;
			for k1=1:size(F_vec,1)
				for k3=1:size(F_vec,3)
					if ~isempty( F_vec(k1,1,k3).multiplot_array )
						
						for n1 = 1:size( F_vec(k1,1,k3).multiplot_array , 1)
							for n2 = 1:size( F_vec(k1,1,k3).multiplot_array , 2)
								for n3 = 1:size( F_vec(k1,1,k3).multiplot_array , 3)
									 F_vec(k1,1,k3).multiplot_array(n1,n2,n3) = F_vec(k1,1,k3).multiplot_array(n1,n2,n3).translate_axis(tt);
								end
							end
						end
						
					end
					F_vec(k1,1,k3) = F_vec(k1,1,k3).translate_axis(tt);
					
				end
			end
			
			
		end
		
	end	
	
	
	methods(Static)
		
		function test
			
		    simple_test = 0;			
			if simple_test
				
				Ncurves = 4;
				Npoints = 40;
				
				X = linspace(3,10,Npoints);
				Y = randn(Ncurves,Npoints);
				
				my_F = F_figure('X',X,'Y',Y);
				
				my_F.plot;
			end
			
			sequence_test = 1;
			if sequence_test
				
				Ncurves = 4;
				Npoints = 40;
				
				X = linspace(3,10,Npoints);
				
				
				for k = 1:10
					Y = randn(Ncurves,Npoints);
					Fseq(k) = F_figure('X',X,'Y',Y);					
				end
				
				
				my_F = F_figure('multiplot_array',Fseq,'multiplot_type','sequence');
				
				my_F.plot;
				
				
			end
			
		end
		
		
		function str_out = translate_string(str_in,translation_table)
			assert( iscell(translation_table)&&(size(translation_table,2)==2));
			for k = 1:size(translation_table,1)
				str_in = regexprep(str_in,translation_table{k,1},translation_table{k,2});
			end
			str_out = str_in;
			
		end
		
	end
		
end