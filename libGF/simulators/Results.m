classdef Results
	% Class to encapsulate results of estimators using a single object.
	% Fields can be set and recovered using methods of this class. 
	%
	% Example
	%
	%>>    car = Results('number_of_windows',4,'parts',{'wheels','seats','mirrors'})
	%>>    car.getByName('number_of_windows')
	%         ans =
    %                4 
    %
	properties
		field_1
		name_1
		field_2
		name_2
		field_3
		name_3
		field_4
		name_4
		field_5		
		name_5
		field_6		
		name_6
	end
	
	methods
		
		function obj = Results( varargin )
			if mod(length(varargin),2)~=0
				error('The number of parameters must be even');
			end
			nps = length(varargin)/2;
			for k=1:nps;
				obj.(sprintf('name_%d',k)) = varargin{2*k-1};
				obj.(sprintf('field_%d',k)) = varargin{2*k};
			end
		end
		
		%delete this function
		function obj = frmr_Results( name1, val1 )
			
			obj.field_1 = val1;
			obj.name_1 = name1;
		end
		
		function val = getByName( obj , str )
			% if obj is a single object, val returns the value of the field
			% with name STR
			% if obj is a matrix:
			%      - if that field has scalar values, then the output is a
			%        vector obtained by concatenating those values
			%      - if that field does not have scalar values, then the
			%        output is a cell array with those values
			%
			if numel(obj) == 1
				val = getByNameElement(obj, str );
			else
				if isscalar(obj(1).getByNameElement( str ))
					val = obj(1).getByNameElement( str );	
					for ii = 1:size(obj,1)
						for jj = 1:size(obj,2)
							val(ii,jj) = obj(ii,jj).getByNameElement( str );
						end
					end
				else
					for ii = 1:size(obj,1)
						for jj = 1:size(obj,2)												
							val{ii,jj} = obj(ii,jj).getByNameElement( str );
						end
					end
				end
			end
		end
		
		function val = frmr_getByName( obj , str )
			% if obj is a single object, val returns the value of the field
			% with name STR
			% if obj is a vector:
			%      - if that field has scalar values, then the output is a
			%        vector obtained by concatenating those values
			%      - if that field does not have scalar values, then the
			%        output is a cell array with those values
			%
			if length(obj) == 1
				val = getByNameElement(obj, str );
			else
				val = obj(1).getByNameElement( str );	
				if isscalar(val)
					for k = 1:length(obj)
						val(k) = obj(k).getByNameElement( str );
					end
				else
					for k = 1:length(obj)
						val{k} = obj(k).getByNameElement( str );
					end
				end
			end
		end
		
		function val = getByNameElement(obj, str )
			switch str
				case {obj.name_1}
					val = obj.field_1;
					return
				case {obj.name_2}
					val = obj.field_2;
					return
				case {obj.name_3}
					val = obj.field_3;
					return
				case {obj.name_4}
					val = obj.field_4;
					return
				case {obj.name_5}
					val = obj.field_5;
					return
				case {obj.name_6}
					val = obj.field_6;
					return
				otherwise
					str
					obj
					error('property does not exist');					
			end
				
			
		end	
		
		function avg = averageWithoutOutliers(obj,str)
			vv = obj.getByName(str);
			[~,~,inliers] = Results.getOutliers(vv);
			avg = mean( vv(inliers) );
			
		end
		
		function res_out = histogram(obj,str,varargin)
			nrows = size(obj,1);
			ncols = size(obj,2);
			res_out(nrows,ncols) = Results();
			for ii = 1:nrows
				for jj = 1:ncols
					data = obj(ii,jj).getByNameElement(str);
					[ncount,xaxis ] = hist(data,varargin{:});
					res_out(ii,jj) = Results('ncount',ncount,'xaxis',xaxis);
				end
			end
			
		end
		
		function res_out = samplePdf(obj,str,varargin)
			% varargin are the parameters for the hist function
			res_hist = histogram(obj,str,varargin{:});
			nrows = size(res_hist,1);
			ncols = size(res_hist,2);
			res_out(nrows,ncols) = Results();
			for ii = 1:nrows
				for jj = 1:ncols
					ncount = res_hist(ii,jj).getByNameElement('ncount');
					xaxis = res_hist(ii,jj).getByNameElement('xaxis');
					lims = obj.x_hist_limits(xaxis);
					lengths = lims(2:end)-lims(1:end-1);
					spdf = ncount/sum(ncount)./lengths;
					res_out(ii,jj) = Results('spdf',spdf,'xaxis',xaxis);
				end
			end
		
			
		end
		
		function mat = fieldToMatrix(obj,str)
			% vertical concatenation of the field STR corresponding to all
			% elements in the array OBJ
			obj=obj(:);
			vals = obj.getByName(str);
			
			if iscell(vals)
				mat = fill_mat_f(vals{:});
			else
				mat = vals;
			end

		end
		
		
		function mat = fieldToMatrixHz(obj,str)
			% Horizontal concatenation of the field STR corresponding to all
			% elements in the array OBJ
			obj=obj(:);
			vals = obj.getByName(str);

			for k= 1:length(vals)
				vals{k} = vals{k}.';
			end
			if iscell(vals)
				mat = fill_mat_f(vals{:});
			else
				mat = vals;
			end

			mat = mat.';
		end
		
	end
	
	methods(Static)
		
		function [outl,inds,indscomp] = getOutliers( vec )
			% only for the case when the outlier is greater than the other
			% values.
			thresh = 5;

			[vs,sort_order] = sort(vec);
			nstdvs = floor(length(vec)/2);
			for k=1:nstdvs
				stdv(k) = std( vs(1:length(vec)-nstdvs + k)  );
			end			
			stdv_diff = stdv - [0 stdv(1:end-1)];
			
			
			inds_over_thresh = find(stdv_diff > thresh) + length(vec)-nstdvs;
			if isempty(inds_over_thresh)
				outl = [];
				inds = [];
				indscomp = 1:length(vec);
			else
				plot([stdv' stdv_diff']);
				outl_inds = inds_over_thresh(1):length(vec);
				fprintf('%d outliers were found\n',length(outl_inds));				
				inds = sort_order(outl_inds);
				outl = vec(inds);
				oo = ones(1,length(vec));
				oo(inds) = 0;
				indscomp = find(oo==1);				
			end

			
			
		end
		
	end
	
	methods(Static,Access=private)
		
		function lims = x_hist_limits(centers)
			% centers should be increasing
			assert( centers(2)-centers(1) > 0 );
			lims = zeros(1,length(centers)+1);
			lims(1) = centers(1) - ( centers(2)-centers(1) );
			lims(end) = centers(end) + ( centers(end)-centers(end-1) );
			lims(2:length(centers)) = (centers(1:end-1)+centers(2:end))/2;						
		end
		
	end
	
end

