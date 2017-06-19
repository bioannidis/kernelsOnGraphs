function I = mv_quadgk(fun, a , b , tol)
% multivariate quadgk
% this function integrates a function FUN on a hyperrectangle of R^N
% defined by the Nx1 vectors a and b. Some components of a and b may take
% infinite values. FUN is a function from R^N into R
%
% In general, integration in more than 2 dimensions need a loooooot of time
%
%
% TOL is optional
%


if nargin<4
	tol = 3;
end

%fun =  @(x,y)(1/(x^2+y^2));
%fun =  @(x)((x(1)^2+x(2)^2));

% fun =  @(x)(norm(x)^2);
% a=[0;0;0];
% b = [1 2 3];
% example
%     mv_quadgk( @(x)(norm(x)^2) , [0;0;0]  , [1 2 3]  )
% 

%%%%%5

 fhand_out = int_last_var( fun , a(end) , b(end) , tol);
 
 % debug
%  fhand_out = int_last_var( fhand_out , a(2) , b(2) );
%  tic
%  quadgk(@(x)tvin(fhand_out,x) ,0,1,'AbsTol',3)
%  toc
 %
 
 
 for k = length(a)-1:-1:1 
	 fhand_out = int_last_var( fhand_out , a(k) , b(k) , tol );
 end
 
% tic
 I = fhand_out([]);%
%toc
end

function fhand_out = int_last_var( fhand_in , a , b , tol)
% fhand_out is the handle to the function result of integrate fhand_in
% w.r.t. the last component between a and b. fhand_in is the handle of a
% scalar valued function of a (column) vector variable.

%tol =3;

% beware of the size of x_small
fhand_out = @( x_small )quadgk( @(y)tvin( @(y_aux)fhand_in([ x_small ; y_aux]) , y ) , a , b ,'AbsTol',tol);



end


% 
% function I_now = int_y(x,func,a,b)
% 	tol = 1e-2;
% 	I_now = quadgk(  @(y)tvin(	func, x, y ) , a , b ,'AbsTol',tol);
% 
% 	
% end
% 


function y = tvin(sfun,x)
% sfun has a scalar input. It turns it in vector input
%
%   x is an Nx1 input vector
%   y is an Nx1 output vector 

for n=length(x):-1:1	
	y(n) = sfun(x(n));
end


end





