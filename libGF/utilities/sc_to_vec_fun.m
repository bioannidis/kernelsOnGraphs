function y = sc_to_vec_fun(sfun,varargin)
%  this turns a sfunction whose input is scalar-valued into a sfunction whose
%  input is vector-valued
%
% sfunction FUN takes scalar arguments. here we just execute the sfunction
% for all columns of the input arguments. 
%
%  sfun           can be a sfunction name or a handle
%  varargin{n}   is a scalar or 1xN vector
%  y             is a 1xN vector
%
% example:
%     >>  sc_to_vec_fun(@(x,y)(x^2+y^2),[1:4],[1:4])

assert( isa(sfun,'function_handle') || isa(sfun,'char')  );
% number of outputs
for n=length(varargin):-1:1
	sz(n) = size(varargin{n},2);
end
N = max(sz);
% for n=1:length(varargin)
% 	if size(varargin{n},2)==1
% 		varargin{n} = varargin{n}*ones(1,N);
% 	end
% end



%
for n=N:-1:1
	for k=1:length(varargin)
		if sz(k) == N
			args_now{k} = varargin{k}(:,n);
		elseif sz(k) == 1
			args_now{k} = varargin{k};
		else
			error('arguments must have N or 1 columns');
		end
	end
	
	y(:,n) = feval(sfun,args_now{:});
	
end


end