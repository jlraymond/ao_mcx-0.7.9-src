function [cwfluence0,cwfluence1,header,det,PRC,Intensity,Intensity0,Intensity1,AvgPhi] = AOI_MCX_Eval(fname,time,dim,absorptions)
%AOI_MCX_EVAL interprets the aoi-mcx output files and produces fluence,
%detector, header, PRC based AO signals, the total energy at each detector,
%and the energy at each frequency at each detector.  0 represents the
%optical frequency, and 1 represents the first acoustic sideband
%
% fname is the name of the file, time is the length of the time window, dim
% is a 3 element vector containing the dimensions of the volume, and
% absorptions is an "m" element vector containing the absorption
% coeffiencient of each medium type
%
% Important Note: The PRC signal here is missing the background signal
% level here! Is important for noise!
%
% Matt Adams
% 9/25/13

% PRC Constants
PRCabs = 1.8;   %Absorption Coefficient (1/cm)
TWM_real = 0.5;      %Two wave mixing coefficient (1/cm)
Lc = 0.7;       %Crystal optical path length (cm)
TWM_im = 0;
Ad = pi*0.25^2;     %Detector Area (cm^2)



% Load files
[det,header]=AOI_loadmch([num2str(fname) '.mch']);
% det is a matrix with the following format:
% Detector number, # of Scattering Events, Magnitude of Modulation, Phase
% Angle of Modulation, partial pathlength in each medium

flux0=loadmc2([num2str(fname) '_0.mc2'],dim);
flux1=loadmc2([num2str(fname) '_1.mc2'],dim);

cwflux0=sum(flux0,4);
cwfluence0=time*cwflux0;    %This is the fluence at the optical frequency

cwflux1=sum(flux1,4);
cwfluence1=time*cwflux1;    %The is the fluence at the acoustic sideband

exponent = zeros(length(absorptions),length(det(:,1)));
for m = 1:length(absorptions)
    exponent(m,:)=-absorptions(m)*det(:,4+m);
end

Ener = exp(sum(exponent,1));
Ener = Ener';
Ener0 = Ener.*(besselj(0,det(:,3)).^2);
Ener1 = Ener*2.*(besselj(1,det(:,3)).^2);

AC = 4*exp(-PRCabs*Lc)*exp(TWM_real*Lc)*sin(TWM_im*Lc)*...
    Ener/Ad.*besselj(1,det(:,3)).*cos(det(:,4));    %This is in Watts (for a 1 W source)
DC = 2*exp(-PRCabs*Lc)*(exp(TWM_real*Lc)*cos(TWM_im*Lc)-1)*...
    Ener/Ad.*(besselj(0,det(:,3))-1);  %This is in Watts (for a 1 W source)

NewAC = zeros(length(header.detnum),1);
NewDC = zeros(length(header.detnum),1);
Intensity = zeros(length(header.detnum),1);
Intensity0 = zeros(length(header.detnum),1);
Intensity1 = zeros(length(header.detnum),1);
AvgPhi = zeros(length(header.detnum),1);

for idx = 1:header.detnum
    NewAC(idx) = sum(AC(find(det(:,1) == idx)));
    NewDC(idx) = sum(DC(find(det(:,1) == idx)));
    Intensity(idx) = sum(Ener(find(det(:,1) == idx)))/Ad;
    Intensity0(idx) = sum(Ener0(find(det(:,1) == idx)))/Ad;
    Intensity1(idx) = sum(Ener1(find(det(:,1) == idx)))/Ad;
    AvgPhi(idx) = sum(det(find(det(:,1) == idx),3))/length(find(det(:,1) == idx));
end

NewAC = NewAC/header.totalphoton;
NewDC = NewDC/header.totalphoton;
Intensity = Intensity/header.totalphoton;
Intensity0 = Intensity0/header.totalphoton;
Intensity1 = Intensity1/header.totalphoton;

% Energy is a (n_det,1) vector containing the total energy measured at each
% detector

%PRC is a (n_det,2) matrix.  The first column is the AC signal at each
%detector, and the second column is the DC signal.
PRC = [NewAC;NewDC];