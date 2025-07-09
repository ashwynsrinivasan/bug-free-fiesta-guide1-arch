clc; close all; clear all;

format long g;
V = 4*(10^-16);
tspan = 0:10*(10^-12):81.92*(10^-9);
R = 1.71*(10^12);
XI = 1:1:8192;
k = R*10*(10^-12);
SIGMA = sqrt(k/(2*V));
MU = 0;
ER(1:8192) = normrnd(MU,SIGMA,1,8192);
Ws = 2.5*(10^9);
elfa = 3;
GN = 1.1*(10^-12);
No = 1.1*(10^24);
Ts = 2*(10^-9);
Tp = 2*(10^-12);
Tin = 8*(10^-12);
Tow = 1*(10^-9);
NTH = No+1/(GN*Tp);
Jth = (NTH/Ts);
J  = 1.3*(Jth);
K = 0.02;
X = 1.45;
Delt = 10*(10^-12);
Wo = 7*(10^9);
ERin = ER(1);
T = 0;
io = 1;
pico = deg2rad(-82.7) + mod((3*X),(2*pi));
Nbar = NTH + ([-2*(X/Tow)*cos(pico)]/GN);
Ebar = sqrt([[GN*(Nbar - No)]^-1]*(J-(Nbar/Ts)));
SIOG = SIGMA/Ebar;
LASEI(1:8192) = normrnd(0,SIOG,1,8192);

%FIRST rungekUTTA METHOD FOR TIME TILL NO FEEDBACK EXISTS

nnT = NTH;
eeT = sqrt([(J - nnT)/(GN*(nnT - No))]);
pit = 6.263;

dedit(1) = rungeit1(nnT,eeT,T);

dpdit(1) = rungeit2(pit,nnT,eeT,T);
eeTL(1) = eeT + dedit(1);
pitL(1) =  pit + dpdit(1);

dndit(1) = rungeit3(nnT,eeTL,T);

nnTL(1) =  nnT + dndit(1);

for ui = 2:1:101
    
    dedit(ui) = rungeit1(nnTL(ui-1),eeTL(ui-1),T);
    
    dpdit(ui) = rungeit2(pit,nnTL(ui-1),eeTL(ui-1),T);
    eeTL(ui) = eeTL(ui-1) + dedit(ui);
    pitL(ui) =  pitL(ui-1) + dpdit(ui);
    
    dndit(ui) = rungeit3(nnTL(ui-1),eeTL(ui),T);
    
    nnTL(ui) =  nnTL(ui-1) + dndit(ui);
    
end

%SECOND rungekUTTA METHOD AFTER FEEDBACK STARTS.

format long g;
ET = eeTL(1:101);
Nt = nnTL(1:101);
PHITY = pitL(1:101);

DEDT(1) = rungek1(Nt(101),ET(101),ET(1),ERin,T,PHITY(101),PHITY(1)); %//////////\\\\\\\\\

LASEI1 = LASEI(1);
DPIDT(1) = rungek2(Nt(101),ET(101),ET(1),Delt,T,PHITY(101),PHITY(1),LASEI1); %//////////\\\\\\\\\\

ETW = ET(101) + DEDT;
DNDT(1) = rungek3(ETW,Nt(101),Delt);  %//////////\\\\\\\\

ETL(1) = ET(101) + DEDT(1);
PHITL(1) = PHITY(101) + DPIDT(1);
NtL(1) = Nt(101) + DNDT(1);

LI = 2;
KI = 1;

WRI = sqrt(GN*(NtL(1)-No)*ETL(1));

for ni = 2:1:2000
    if ni <= 101
        
        DEDT(ni)  = rungek1(NtL(ni-1),ETL(ni-1),ET(LI),ER(ni),T,PHITL(ni-1),PHITY(LI));
        
        DPIDT(ni) = rungek2(NtL(ni-1),ETL(ni-1),ET(LI),Delt,T,PHITL(ni-1),PHITY(LI),LASEI(ni));
        
        SPHIF(ni) = abs(DPIDT(ni))^2 *((2*pi*(12*10^6))^2)/(81.92*(10^-9));
        SPHIFL(ni) = SPHIF(ni)*abs(([exp(-j*(2*pi*(12*10^6)*81.92*(10^-9)))] - [1])/(-j*2*pi*(12*10^6)));
        
        ETL(ni) = ETL(ni-1) + DEDT(ni);                 %FINALE - ELECTRIC FIELD
        PHITL(ni) = PHITY(ni-1) + DPIDT(ni);            %FINALE - PHASE VARIATION
        
        DNDT(ni) = rungek3(ETL(ni),NtL(ni-1),Delt);
        
        NtL(ni) = NtL(ni-1) + DNDT(ni);                 %FINALE - CHARGE DENSITY
        
        ACKATI(ni) =ETL(ni)*exp(i*((Wo*tspan(ni))+(PHITL(ni))));
        ACKAL(ni) = exp(i*((Wo*tspan(ni))+(PHITL(ni))));
        io = io + 1;
        LI = LI + 1;
    else
        
        DEDT(ni) = rungek1(NtL(ni-1),ETL(ni-1),ETL(KI),ER(ni),T,PHITL(ni-1),PHITL(KI));
        
        DPIDT(ni) = rungek2(NtL(ni-1),ETL(ni-1),ETL(KI),Delt,T,PHITL(ni-1),PHITL(KI),LASEI(ni));
        
        SPHIF(ni) = abs(DPIDT(ni))^2 *((2*pi*(12*10^6))^2)/(81.92*(10^-9));
        SPHIFL(ni) = SPHIF(ni)*abs(([exp(-j*(2*pi*(12*10^6)*81.92*(10^-9)))] - [1])/(-j*2*pi*(12*10^6)));
        
        ETL(ni) = ETL(ni-1) + DEDT(ni);                    %FINALE - ELECTRIC FIELD
        PHITL(ni) = PHITL(ni-1) + DPIDT(ni);               %FINALE - PHASE VARIATION
        
        DNDT(ni) = rungek3(ETL(ni),NtL(ni-1),Delt);
        
        NtL(ni) = NtL(ni-1) + DNDT(ni);                    %FINALE - CHARGE DENSITY
        ACKATI(ni) = ETL(ni)*exp(j*((Wo*tspan(ni))+(PHITL(ni))));
        ACKAL(ni) = exp(i*((Wo*tspan(ni))+(PHITL(ni))));
        KI = KI+1;
    end
end

WRIL = sqrt(GN*(NtL(ni)-No)*ETL(ni));

figure(1);
%[Pxx,w] = pwelch(DPIDT,[],[],2048,200000000000);
%[Pxx,f] = periodogram(DPIDT(2:length(DPIDT)),[],512,200000000000);
[Pxx,f] = periodogram(DEDT(2:length(DEDT)),[],2048,200000000000);
spectrum(Pxx,f);
format long g;

figure(2);
plot(tspan(2:length(DPIDT)),DEDT(2:length(DEDT)));
title('variation of E field increments W.R.T to time');
xlabel('time axis');
ylabel('Electric field increments');

figure(3);
plot(tspan(2:length(DPIDT)),DPIDT(2:length(DPIDT)));
title('variation of phase increments W.R.T to time');
xlabel('time axis');
ylabel('phase density increments');

figure(4);
plot(tspan(2:length(DPIDT)),DNDT(2:length(DNDT)));
title('variation of carrier increments W.R.T to time');
xlabel('time axis');
ylabel('Charge density increments');

figure(5);
plot(tspan(2:length(DPIDT)),(ETL(2:length(ETL))/Ebar));
title('evolution of electric field');
xlabel('time axis');
ylabel('Electric field ');
axis([0*(10^-8) 2*(10^-8) -10 10]);

figure(6);
plot(tspan(2:length(DPIDT)),PHITL(2:length(PHITL)));
title('evolution of phase');
xlabel('time axis');
ylabel('PHASE');

figure(7);
plot(tspan(2:length(DPIDT)),NtL(2:length(NtL)));
title('carrier density evolution');
xlabel('time axis');
ylabel('CARRIER DENSITY');

figure(8);
plot(xcorr(PHITL,PHITL));
title('correlation of phase');
xlabel('time axis');
ylabel('Amplitude axis');

figure(9);
plot(xcorr(ETL,ETL));
title('correlation of Electric field');
xlabel('time axis');
ylabel('Amplitude axis');

figure(10);
plot(xcorr(NtL,NtL));
title('correlation of charge density');
xlabel('time axis');
ylabel('Amplitude axis');

figure(11);
plot(tspan(2:length(DPIDT)),(ETL(2:length(ETL))/Ebar));
title('evolution of electric field');
xlabel('time axis');
ylabel('Electric field ');
axis([0*(10^-8) 2*(10^-8) -10 10]);

figure(12);
plot3(PHITL(2:length(PHITL)),(ETL(2:length(ETL))),NtL(2:length(NtL)) );
title('STRANGE ATTRACTORS');
xlabel('PHASE');
ylabel('ELECTRIC FIELD');
zlabel('CHARGE DENSITY');
grid on;

figure(13);
[Pxx,f] = periodogram((ETL(2:length(DEDT))).^2,[],2048,200000000000);
spectrum(Pxx,f);
title('in EXCELLENT AGREEMENT WITH 2003DECEMBER PAPER');
