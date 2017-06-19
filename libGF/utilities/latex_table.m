function fd_out=latex_table(order,fd_in,tnum,enum,caption,row)
% LATEX_TABLE writes one table to a LaTeX file. The label will
% be sprintf(tab%d:%d,enum,tnum). It is only needed to include this file
% with "input{}".
%  ->order can be:
%       -'new': creates a new file and begins the table. also it writes the
%       first row, usually a row of strings with the sense of each col.
%       -'addrow': adds another row.
%       -'end': finishes the table and closes the file.
%  ->fd_in: with order='new' fd_in can be anything. with other options it 
%  must be the fd_out provided by the function when order='new'.
%  ->row: must be a cell array when order=='new'

N=length(row);
switch order
    case 'new'
        if (~iscell(row))
            disp('error: first row must be a cell array');
        return
        end
        fd=fopen(sprintf('exercise%dt%d.tex',enum,tnum),'w');
        fprintf(fd,'\\begin{table}[bhtp]\n');
        fprintf(fd,'\\begin{center}\n');
        fprintf(fd,'\\begin{tabular}{');
        for k=1:N
            fprintf(fd,'|c ');
        end
        fprintf(fd,'|}\n     \\hline\n');
        for k=1:N-1
      	    fprintf(fd,' %s &',row{k});
        end
        fprintf(fd,' %s \\\\ \\hline \\hline\n',row{N});
        fd_out=fd;
    case 'addrow'
        for k=1:N-1
      	    fprintf(fd_in,' %.6f &',row(k));
        end
        fprintf(fd_in,' %.6f \\\\ \\hline \n',row(N));
        fd_out=fd_in;
    case 'end'
        fprintf(fd_in,'  \\end{tabular}\n');
        fprintf(fd_in,'  \\end{center}\n');
        fprintf(fd_in,'  \\caption{%s}\n  \\label{tab%d:%d}\n',caption,enum,tnum); 
        fprintf(fd_in,'\\end{table}\n');
        fclose(fd_in);
        fd_out=fd_in;       
end

return
