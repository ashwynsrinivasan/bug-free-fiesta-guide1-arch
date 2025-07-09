function dedit = firsti(nnT,eeT);

GN = 1.1*(10^-12);
No = 1.1*(10^24);
Tp = 2*(10^-12);
NTH = No+(1/(GN*Tp));
V = 4*(10^-16);
R = 1.71*(10^12);

dedit = (0.5*GN*(nnT-NTH)*eeT) +  (R/(2*V*eeT));
