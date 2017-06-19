function str_out = str_remove_char(str_in,chr)

inds_remove = strfind( str_in , chr );

inds_keep = setxor(inds_remove,1:length(str_in) );
str_out = str_in(inds_keep);


end