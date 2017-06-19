function Ari = matrix_Ari(A)
%       [ real(A)   -imag(A) ]
% Ari = [                    ]
%       [ imag(A)    real(A) ]

Ari = [real(A),-imag(A);imag(A),real(A)];

end
