function str=icomplex_to_str(z)
% integer complex to string
if real(z)==0
	if imag(z)==0
		str=0;
	else
		if imag(z)==1
			str=sprintf('j');
		elseif imag(z)==-1
			str=sprintf('-j');
		else
			if imag(z)>0
				str=sprintf('j%d',imag(z));
			else
				str=sprintf('-j%d',abs(imag(z)));
			end
		end
	end
else
	if imag(z)==0
		str=sprintf('%d',real(z));
	else
		if imag(z)==1
			str=sprintf('%d+j',real(z));
		elseif imag(z)==-1
			str=sprintf('%d-j',real(z));
		else
			if imag(z)>0
				str=sprintf('%d+j%d',real(z),imag(z));
			else
				str=sprintf('%d-j%d',real(z),abs(imag(z)));
			end
		end
	end
end

return