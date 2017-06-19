classdef Parameter < matlab.mixin.Heterogeneous
	% This class allows obtaining information about parameters of
	% descendants (e.g. generators, samplers, estimators) in text form.
	% This is useful to make figures. 
	%
	% all classes extending Parameter shall allow their constructors to be
	% called without any parameter.
        %
        % 

	properties(Abstract)
		% the n-th property of an object of class Parameter will be printed
		% as   my_sprintf( obj.c_patternToPrint{n} , obj.c_stringToPrint{n} , getfield(obj,obj.c_parsToPrint{n}) );		
		c_parsToPrint    % cell array of strings containing the name of the parameters to print
		                 % For example c_parsToPrint =
		                 % {'par1','par2','par3'}
						 % To customize the way of printing the value of
						 % the parameter, just create a method of the form
						 %    function  str = par1_print(obj)
						 % that returns the desired string.
						 %
		c_stringToPrint  % cell array of strings containing the string for printing each parameter (titles, axes and legends)
		c_patternToPrint % cell array of strings containing the string pattern for sprintf in titles (e.g. {'%s = %d'})
	end 
	
	properties
		c_replicatedVerticallyAlong = {};   % property names in this list 
		% are used to make legends In an array of objects of class 
		% Parameter, only the values of this property for the objects in the first
		% column are considered. If this property is empty for the entry
		% (n,1), then the one corresponding to the entry (m,1) is used,
		% where m is the largest index less than n such that the element
		% (m,1) has a non-empty c_replicatedVerticallyAlong field.
		%
		% If all legend entries have to have the same structure, just set
		% this field for the (1,1) element.
		
		c_replicatedHorizontallyAlong = {}; % property name in this list is 
		% used to make the label of the x-axis. For arrays of objects of
		% class Parameter, the only considered is the (1,1) entry.
	end
	
	properties(Constant)
		def_chars_per_line = 200;
    end
	
	methods
		
		function obj = Parameter(varargin)
			% varargin is a sequence of (property_name,property_value)
			% pairs
			% Example:
			%     car = Parameter('numberOfWheels',4,'year',2009,'color','blue')
			obj =  assignParametersByName(obj,varargin{:});
			
		end
		
		
		function str = getParameterByNumber( obj1, par_number )
			obj1 = obj1(1,1);
			assert(numel(obj1)==1);
			parameterName = obj1.c_parsToPrint{par_number};
			if ismethod(obj1,[parameterName '_print'])
				str = eval(['obj1.' parameterName '_print']);
			else			
				str = my_sprintf( obj1.c_patternToPrint{par_number} , obj1.c_stringToPrint{par_number} , my_getfield(obj1,obj1.c_parsToPrint{par_number}) );
			end
		end
		
		function str = getParameterByName( obj1, name )
			par_number = obj1.parameterIndex(name);
			str =  getParameterByNumber( obj1, par_number );
		end
		
		function ind = parameterIndex(obj,str)
			for k=1:length(obj.c_parsToPrint)
				if strcmp(obj.c_parsToPrint{k},str)
					ind = k;
					return
				end
			end
			str = sprintf('parameter ''%s'' does not exist in %s.c_parsToPrint',str,class(obj));
			error(str);
		end
		
	end

	
	methods
				
		function obj_mat = replicate(obj,fieldname_1,fieldvalues_1,fieldname_2,fieldvalues_2)
			% input 
			%    FIELDVALUES_1: cell array with M values for the file in
			%                   Parameter called FIELDNAME_1
			%    FIELDNAME_1  : string with the name of the field
			%    
			%    FIELDVALUES_2: cell array with N values for the file in
			%                   Parameter called FIELDNAME_1
			%    FIELDNAME_2  : string with the name of the field
			%
			% output
			%    obj_mat      : MxN matrix where the (m,n)-th element has
			%                   all the values equal to those of the
			%                   current object except from the field
			%                   FIELDNAME_1, which has value
			%                   FIELDVALUES_1{m} and the field
			%                   FIELDNAME_2, which has value
			%                   FIELDVALUES_2{n}, that is, FIELDNAME_1
			%                   replicates the object vertically, and
			%                   FIELDNAME_2 does it horizontally
			%
			
			assert(numel(obj)==1,'REPLICATE not implemented for input array objects');
			% this  function is implemented for scalar objects. see
			% replicate_horizontally above
			
			% check that fieldvalues are cells. You can use num2cell
			assert( (isempty(fieldvalues_1)||iscell(fieldvalues_1))&&(isempty(fieldvalues_2)||iscell(fieldvalues_2) ));
			
			obj.c_replicatedVerticallyAlong=[obj.c_replicatedVerticallyAlong {fieldname_1}];
			obj.c_replicatedHorizontallyAlong=[obj.c_replicatedHorizontallyAlong {fieldname_2}];
			M = max(length(fieldvalues_1),1);
			N = max(length(fieldvalues_2),1);
			obj_mat = repmat(obj,M,N);
			for m=1:M
				for n=1:N
					obj_mat(m,n)=obj;
					if ~isempty(fieldname_1)
						obj_mat(m,n).(fieldname_1) = fieldvalues_1{m};
					end
					if ~isempty(fieldname_2)
						obj_mat(m,n).(fieldname_2) = fieldvalues_2{n};
					end
				end
			end
			
		end
				
		function d = is_c_replicatedVerticallyAlong(obj,str)
			
			obj = obj(1,1);
			
			if isempty( obj.c_replicatedVerticallyAlong )
				d = 0;
			else
				d = 0;
				for k=1:length( obj.c_replicatedVerticallyAlong )
					if strcmp( str , obj.c_replicatedVerticallyAlong{k} )
						d=1;
						return
					end
				end
			end
			
		end
				
		function d = is_c_replicatedHorizontallyAlong(obj,str)
			
			obj = obj(1,1);
			
			if isempty( obj.c_replicatedHorizontallyAlong )
				d = 0;
			else
				d = 0;
				for k=1:length( obj.c_replicatedHorizontallyAlong )
					if strcmp( str , obj.c_replicatedHorizontallyAlong{k} )
						d=1;
						return
					end
				end
			end
			
		end
				
	end
	
	
	methods(Static)
		

		
		
		function leg = getLegend(varargin)
			
			list = Parameter.getLegendList(varargin{:});
			
			leg ={};
			for k = 1:size(list,1)
				leg{k} = Parameter.strListToText( {list{k,:}},1000 );
			end
			
		end
		
		function leg = getLegendList(varargin)
			% it makes a 2D cell array with the entries for the legend
			%
			% leg = getLegendList(OBJ_1,OBJ_2,...,OBJ_N)
			%
			%   OBJ_n   : array of objects of class Parameter
			%
			%   leg     : cell array where the m-th row is a cell array
			%             with strings printing the values of the
			%             parameters whose names are in the field
			%             c_replicatedVerticallyAlong of OBJ_n(m,1). Those
			%             parameters are supposed to be common to all
			%             objects in the m-th row, but this is not checked.
			%             
			%
			last_col = 0;
			leg = {};
			for k = 1:nargin
				obj_array_now = varargin{k};
				obj_now = obj_array_now(1,1);
				leg_pars = obj_now.c_replicatedVerticallyAlong;		
				if (size(obj_array_now,1)<2)&&(~isempty(leg_pars)&&(~isempty(leg_pars{1})))
					str = sprintf('Property c_replicatedVerticallyAlong is non-empty for an array of class %s not truly replicated vertically',class(obj_array_now));
					warning(str);
				end
				
				for row = 1:size(obj_array_now,1)
					obj_now = obj_array_now(row,1);
					if ~isempty(obj_now.c_replicatedVerticallyAlong)
						leg_pars = obj_now.c_replicatedVerticallyAlong;
					end
					
					for par = 1:length(leg_pars)
						par_now = leg_pars{par};
						if isempty(par_now)
							continue
						end						
						str = obj_now.getParameterByName(par_now);
						leg{row,last_col + par} = str;
					end
					
				end
				
				last_col = size(leg,2);
				
			end
		end
				
		function tit = getTitleFilter(no_list,varargin)
			global chars_per_line
			if isempty(chars_per_line)
				chars_per_line = Parameter.def_chars_per_line;
			end
			
			list = Parameter.getTitleList(varargin{:});
			assert(max(no_list)<=length(list));
			inds = setxor(1:length(list),no_list);
			list = list(inds);
			tit = [];
			tit_len = 0;
			for k=1:length(list)
				% end of line
				if tit_len + length(list{k}) > chars_per_line
					tit = [tit sprintf('\n')];
					tit_len = length(list{k});										
				else					
					tit_len = tit_len + length(list{k});					
				end
				tit = [tit list{k}];
				% comma
				if k~=length(list)
					tit = [tit ', '];
				end
			end

		end
				
		function list = getTitleList(varargin)
			% cell array with the strings to put in the title
			% arguments are objects of the class Parameter

			list = {};
			for nn = 1:nargin
				obj1 = varargin{nn};				
				obj1 = obj1(1,1);
				for k=1:length(obj1.c_parsToPrint)
					par_name = obj1.c_parsToPrint{k};
					if (~is_c_replicatedVerticallyAlong(obj1,par_name) )&&(~is_c_replicatedHorizontallyAlong(obj1,par_name) )
						str = obj1.getParameterByNumber(k);
						if ~isempty(str)
							list = {list{:},str};
						end
					end
				end
			end
		end
				
		function tit = getTitle(varargin)
			global chars_per_line
			if isempty(chars_per_line)
				chars_per_line = Parameter.def_chars_per_line;
			end
			
			list = Parameter.getTitleList(varargin{:});
			tit = Parameter.strListToText(list,chars_per_line);

		end				
		
		function tit = strListToText(list,chars_per_line)
			% this function takes a list of strings and concatenates them
			% with commas in between and inserting an EOL when the length
			% of the current line exceeds CHARS_PER_LINE
			
			tit = [];
			tit_len = 0;
			for k=1:length(list)
				if isempty(list{k})
					continue
				end
				% end of line
				if tit_len + length(list{k}) > chars_per_line
					tit = [tit sprintf('\n')];
					tit_len = length(list{k});										
				else					
					tit_len = tit_len + length(list{k});					
				end
				tit = [tit list{k}];
				% comma
				if k~=length(list)
					tit = [tit ', '];
				end
			end

		end
				
		
		function [xlab,xvalues] = getXLabel(varargin)
			
			for k = 1:length(varargin)
				obj_array_now = varargin{k};				
				obj1=obj_array_now(1,1);
				if isempty(obj1.c_replicatedHorizontallyAlong) || ...
                        isempty(obj1.c_replicatedHorizontallyAlong{1})
					continue;					
				end
				ind = obj1.parameterIndex(obj1.c_replicatedHorizontallyAlong{1});
				fieldname = obj1.c_parsToPrint{ind};
				xlab = sprintf('%s',obj1.c_stringToPrint{ind});
				xvalues =[ obj_array_now(1,:).(fieldname)];
				return
			end
			error('Object array not replicated horizontally');
		end
		
		
		function [xvalues] = getXAxis(varargin)
			
			for k = 1:length(varargin)
				obj_array_now = varargin{k};				
				obj1=obj_array_now(1,1);
				if isempty(obj1.c_replicatedHorizontallyAlong) || ...
                        isempty(obj1.c_replicatedHorizontallyAlong{1})
					continue;					
				end
				ind = obj1.parameterIndex(obj1.c_replicatedHorizontallyAlong{1});
				fieldname = obj1.c_parsToPrint{ind};
				xvalues =[ obj_array_now(1,:).(fieldname)];
				return
			end
			error('Object array not replicated horizontally');
		end
		
		
	end
	
	
end

