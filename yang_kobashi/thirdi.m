function dndit = thirdi(eeT,nnT);

Tp = 2*(10^-12);
GN = 1.1*(10^-12);
No = 1.1*(10^24);
Ts = 2*(10^-9);
NTH = No+(1/(GN*Tp));
Jth = (NTH/Ts); 
No = 1.1*(10^24);
J  = 1.3*(Jth);

dndit = J - (nnT/Ts) - [GN*(nnT-No)*(abs(eeT)^2)] ;
