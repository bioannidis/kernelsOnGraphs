function A = un_matrix_Ari(Ari)
%       [ real(A)   -imag(A) ]
% Ari = [                    ]
%       [ imag(A)    real(A) ]

M = size(Ari,1)/2;

A = Ari(1:M,1:M)+Ari(M+1:2*M,M+1:2*M) + 1j * (Ari(M+1:2*M,1:M) -Ari(1:M,M+1:2*M) );
A = 0.5 * A;

end
