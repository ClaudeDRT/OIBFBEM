function [P_t_full,P_t_ml,P_t_full_comp,P_t_ml_comp] = Facet_Echo_Model_looped(op_mode,lambda,bandwidth,P_T,h,v,pitch,roll,prf,beam_weighting,G_0,D_0,gamma1,gamma2,N_b,t,PosT,surface_type,sigma_0_snow_surf,sigma_0_snow_vol,kappa_e,tau_snow,c_s,h_s,sigma_0_ice_surf,sigma_0_lead_surf,sigma_0_mp_surf,ice_std_to_snow_std_ratio)

%% Facet-based Radar Altimeter Echo Model for Sea Ice

% Simulates the backscattered echo response of a pulse-limited or synthetic
% aperture radar altimeter from snow-covered sea ice, over a facet-based
% triangular mesh of the sea ice surface topography

%% Input (reference values)
% op_mode = operational mode: 1 = pulse-limited, 2 = SAR (PL-mode only feasible on high memory machines)
% lambda = radar wavelength, 0.0221 m
% bandwidth = antenna bandwidth, Hz
% P_T = transmitted peak power, 2.188e-5 watts
% h = satellite altitude, 720000 m
% v = satellite velocity, 7500 m/s
% pitch = antenna bench pitch counterclockwise, rads (up to ~0.005 rads)
% roll = antenna bench roll counterclockwise, rads (up to ~0.005 rads)
% prf = pulse-repitition frequency, Hz
% beam_weighting = weighting function on beam pattern (1 = rectangular, 2 =
% Hamming)
% G_0 = peak antenna gain, dB
% gamma1 = along-track antenna parameter, 0.0116 rads
% gamma2 = across-track antenna parameter, 0.0129 rads
% N_b = no. beams in synthetic aperture, 64 (1 in PL mode)
% t = time, s
% PosT = surface facet xyz locations (n x 3 matrix)
% surface_type = surface facet type: 0 = lead/ocean, 1 = sea ice, 2 = melt
% pond (n x 3 matrix)
% theta = angular sampling of scattering signatures, rads
% sigma_0_snow_surf = backscattering coefficient of snow surface, dB
% sigma_0_snow_vol = backscattering coefficient of snow volume, dB 
% kappa_e = extinction coefficient of snow volume, Np/m
% tau_snow = transmission coefficient at air-snow interface
% c_s = speed of light in snowpack, m/s
% h_s = snow depth, m
% sigma_0_ice_surf = backscattering coefficient of ice surface, dB
% sigma_0_lead_surf = backscattering coefficient of lead surface, dB
% sigma_0_mp_surf = backscattering coefficient of pond surface, dB

%% Output
% P_t_full = delay-Doppler map (DDM) of single look echoes, watts
% P_t_ml = multi-looked power waveform, watts
% P_t_full_comp = DDM for individual snow surface, snow volume, ice
% surface, and lead surface components, watts
% P_t_ml_comp = multi-looked power waveforms for individual snow surface,
% snow volume, ice surface, and lead surface components, watts

% Based on model equations introduced in Landy et al, TGARS, 2019
% Builing on theory of Wingham et al 2006, Giles et al 2007,
% Makynen et al 2009, Ulaby et al 2014

% Uses the following codes from external sources:
% computeNormalVectorTriangulation.m (David Gingras)

% (c) Jack Landy, University of Bristol, 2018


%% Antenna parameters
%disp('the size of PosT is: ');
%disp(size(PosT));
%disp('done sizing');
c = 299792458; % speed of light, m/s
Re = 6371*10^3; % earth's radius, m

% f_c = c/lambda; % radar frequency, Hz
k = (2*pi)/lambda; % wavenumber

delta_x = v/prf; % distance between coherent pulses in synthetic aperture, m

% delta_x_dopp = (h*prf*c)/(2*N_b*v*f_c); % along-track doppler-beam limited footprint size, m
% delta_x_pl = 2*sqrt(c*(h/((Re+h)/Re))*(1/bandwidth)); % across-track pulse-limited footprint size, m
% A_pl = pi*(delta_x_pl/2)^2; % area of each range ring (after waveform peak), m
N_b = 16;
epsilon_b = lambda/(2*N_b*v*(1/prf)); % angular resolution of beams from full look crescent (beam separation angle) 

% Antenna look geometry
m = -(N_b-1)/2:(N_b-1)/2;

%% Triangulate surface
PosTsnow = PosT;




%%%WARNING!!! I CHANGED THIS SO CHANGE IT BACK:
%%%%PosTsnow(:,3) = PosTsnow(:,3)*0;



% Triangulate
xs_snow = PosTsnow(:,1);
ys_snow = PosTsnow(:,2);
TRIsnow = delaunay(xs_snow,ys_snow);

SURFACE_TYPE = surface_type(TRIsnow(:,1));

% Simplify triangulation to improve speed
% simplication_factor = 0.8; % Fraction of facets remaining after simplification
% figure;DT=trisurf(TRI,PosTx,PosTy,PosTz); set(gcf,'Visible', 'off');
% nfv=reducepatch(DT,simplification_factor);
% TRI = nfv.faces;
% PosT = nfv.vertices;
% PosTx = PosT(:,1); PosTy = PosT(:,2); PosTz = PosT(:,3);

% Compute normal vectors of facets
[NormalVxsnow, NormalVysnow, NormalVzsnow, PosVxsnow, PosVysnow, PosVzsnow]=computeNormalVectorTriangulation(PosTsnow,TRIsnow,'center-cells');

% Compute areas of facets
P0snow = PosTsnow(TRIsnow(:,1),:);
P1snow = PosTsnow(TRIsnow(:,2),:);
P2snow = PosTsnow(TRIsnow(:,3),:);

P10snow = bsxfun(@minus, P1snow, P0snow);
P20snow = bsxfun(@minus, P2snow, P0snow);
Vsnow = cross(P10snow, P20snow, 2);

A_facets = sqrt(sum(Vsnow.*Vsnow, 2))/2;

clear P0snow P1snow P2snow P10snow P20snow Vsnow








%%%WARNING!!! Claude added the next bit (up until the for loop) to make the ice surface differ from the snow

PosTice = PosT;
%Uncomment if you want a flat ice layer:
PosTice(:,3) = PosTice(:,3)*ice_std_to_snow_std_ratio;

% Triangulate
xsice = PosTice(:,1);
ysice = PosTice(:,2);
TRIice = delaunay(xsice,ysice);

SURFACE_TYPE = surface_type(TRIice(:,1));

% Simplify triangulation to improve speed
% simplication_factor = 0.8; % Fraction of facets remaining after simplification
% figure;DT=trisurf(TRI,PosTx,PosTy,PosTz); set(gcf,'Visible', 'off');
% nfv=reducepatch(DT,simplification_factor);
% TRI = nfv.faces;
% PosT = nfv.vertices;
% PosTx = PosT(:,1); PosTy = PosT(:,2); PosTz = PosT(:,3);

% Compute normal vectors of facets
[NormalVxice, NormalVyice, NormalVzice, PosVxice, PosVyice, PosVzice]=computeNormalVectorTriangulation(PosTice,TRIice,'center-cells');

% Compute areas of facets
P0ice = PosTice(TRIice(:,1),:);
P1ice = PosTice(TRIice(:,2),:);
P2ice = PosTice(TRIice(:,3),:);

P10ice = bsxfun(@minus, P1ice, P0ice);
P20ice = bsxfun(@minus, P2ice, P0ice);
Vice = cross(P10ice, P20ice, 2);

A_facets_ice = sqrt(sum(Vice.*Vice, 2))/2;

clear P0ice P1ice P2ice P10ice P20ice Vice










% Construct beam weighting function for azimuth FFT
if op_mode==1 || beam_weighting == 1
    H = ones(1,N_b);
elseif op_mode==2 && beam_weighting == 2    
    H = hamming(N_b);
end

%% Radar Simulator Loop
% Formulated for parallel processing

P_t_full = zeros(length(m),length(t));
sigma_0_tracer = zeros(length(m),length(t),4);
parfor i = 1:length(m)
    % disp(i);
    
    %% Angular geometry of surface
    
    % Antenna location
    %x_0 = h*m(i)*epsilon_b + h*tan(pitch);
    %WARNING!!! CLAUDE ADDED THE FOLLOWING LINE:
    x_0 = m(i)*v/prf + 0;

    y_0 = 0;   %h*tan(roll);
    
    % Calculate basic angles
    R = sqrt((PosVzsnow-h).^2 + ((PosVxsnow-x_0).^2 + (PosVysnow-y_0).^2).*(1 + h/Re));
    THETAsnow = pi/2 + atan2((PosVzsnow-h), sqrt((PosVxsnow-x_0).^2 + (PosVysnow-y_0).^2));
    PHIsnow = atan2((PosVysnow-y_0),(PosVxsnow-x_0));
    
    THETA_Gsnow = pi/2 + atan2((PosVzsnow-R), sqrt((PosVxsnow-x_0+h*tan(pitch)).^2 + (PosVysnow-y_0+h*tan(roll)).^2));
    PHI_Gsnow = atan2((PosVysnow-y_0+h*tan(roll)),(PosVxsnow-x_0+h*tan(pitch)));
    
    %theta_lsnow = atan(-(PosVxsnow-x_0+h*tan(pitch))./(PosVzsnow-h)); % look angle of radar (synthetic beam pattern unaffected by mis-pointing, following beam steering)
    
    %WARNING!!! CLAUDE ADDED THE FOLLOWING LNIE:
    theta_lsnow = atan(-(PosVxsnow-x_0)./(PosVzsnow-h));
    
    %WARNING!!! CLAUDE ADDED THE FOLLOWING LINE:
    theta_lice = atan(-(PosVxice-x_0)./(PosVzice-h));

    % Compute angle between facet-normal vector and antenna-facet vector
    NormalAxsnow = cos(PHIsnow).*cos(pi/2 - THETAsnow);
    NormalAysnow = sin(PHIsnow).*cos(pi/2 - THETAsnow);
    NormalAzsnow = -sin(pi/2 - THETAsnow);
    theta_prsnow = pi - acos((NormalVxsnow.*NormalAxsnow + NormalVysnow.*NormalAysnow + NormalVzsnow.*NormalAzsnow)./(sqrt(NormalVxsnow.^2 + NormalVysnow.^2 + NormalVzsnow.^2).*sqrt(NormalAxsnow.^2 + NormalAysnow.^2 + NormalAzsnow.^2)));
    theta_prsnow(theta_prsnow > pi/2) = pi/2;
    
    %% Compute gain functions
    
    % Antenna gain pattern (based on Cryosat-2)
    G = G_0*exp(-THETA_Gsnow.^2.*(cos(PHI_Gsnow).^2/gamma1^2 + sin(PHI_Gsnow).^2/gamma2^2));
    
    % Synthetic beam gain function
    %P_m = D_0*sin(N_b*(k*delta_x*sin(theta_lsnow+m(i)*epsilon_b))).^2./(N_b*sin(k*delta_x*sin(theta_lsnow+m(i)*epsilon_b))).^2;
    
    %WARNING!!! CLAUDE ADDED THE FOLLOWING LINE:
    P_m = D_0*sin(N_b*(k*delta_x*sin(theta_lice))).^2./(N_b*sin(k*delta_x*sin(theta_lice))).^2;

    %% Compute transmitted power envelope
    
    % Time offset
    % WARNING!!! CLAUDE SET THE FOLLOWING LINE TO ZERO:
    tc = 2*(sqrt(x_0.^2*(1 + h/Re)+h^2)-h)/c; % slant-range time correction


    T = bsxfun(@minus, t + 2*h/c + tc, 2*R/c);
    
    % Power envelope
    P_t = (sin(0.5*bandwidth*pi*T)./(0.5*bandwidth*pi*T)).^2;
    %Claude attempting to make the power envelope look like OIB's:
    %P_t = exp(-(T - 0).^2 / (2 * (1.0820637119102194e-10).^2));

    %% Compute linearized backscattering
    % Surface plus volume echo (following Arthern et al 2001, Kurtz et al 2014 procedure)
    
    theta_PRsnow = bsxfun(@times, ones(size(P_t)), theta_prsnow);


    %%%%WARNINGG!! Claude is trying to definitely definitely get rid of all possible effects that the snow layer would have 
    %on the echo so keep this commented unless you want this to be the case::
    % also if you uncomment these two lines the code literally wont work:P
    %%%%%%%%%%%%%%sigma_0_snow_surf = zeros(size(sigma_0_snow_surf));
    %%%%%%%%%%%%%%sigma_0_snow_vol = zeros(size(sigma_0_snow_vol));


    % Snow volume echo from IEM and Mie extinction
    vu_t_surf = 10.^(ppval(sigma_0_snow_surf,theta_prsnow)/10)*(h_s~=0);
    vu_t_vol = 10.^(ppval(sigma_0_snow_vol,theta_PRsnow(T>=-(2*h_s)/c_s & T<0))/10)*kappa_e.*exp(-c_s*kappa_e*(T(T>=-(2*h_s)/c_s & T<0) + (2*h_s)/c_s));
    
    %%%WARNING!!!!! Claude added this to get rid of the effect that the
    %%%snow has on the waveform shape::
    %Uncomment if you DONT want the snow to contribute:
    %%vu_t_surf = zeros(size(vu_t_surf));
    %%vu_t_vol = zeros(size(vu_t_vol));

    vu_t_surf_tracer = bsxfun(@times,interp1(t - 2*h_s/c_s,P_t',t,'linear')',vu_t_surf); % correct echo for snow depth
    
    vu_t_vol_tracer = zeros(size(P_t)); vu_t_vol_tracer(T>=-(2*h_s)/c_s & T<0) = vu_t_vol;
    vu_t_vol_tracer = interp1(t - 2*h_s/c_s,P_t',t,'linear')'.*vu_t_vol_tracer;
    

    %%%WARNING!!!!! Claude added this to get rid of the effect that the
    %%%snow has on the waveform shape::
    %Uncomment if you DONT want the snow to contribute:
    %%vu_t_surf_tracer = zeros(size(vu_t_surf_tracer));
    %%vu_t_vol_tracer = zeros(size(vu_t_vol_tracer));
    



    % Calculate basic angles
    R = sqrt((PosVzice-h).^2 + ((PosVxice-x_0).^2 + (PosVyice-y_0).^2).*(1 + h/Re));
    THETAice = pi/2 + atan2((PosVzice-h), sqrt((PosVxice-x_0).^2 + (PosVyice-y_0).^2));
    PHIice = atan2((PosVyice-y_0),(PosVxice-x_0));

    THETA_Gice = pi/2 + atan2((PosVzice-R), sqrt((PosVxice-x_0+h*tan(pitch)).^2 + (PosVyice-y_0+h*tan(roll)).^2));
    PHI_Gice = atan2((PosVyice-y_0+h*tan(roll)),(PosVxice-x_0+h*tan(pitch)));

    %theta_lice = atan(-(PosVxice-x_0+h*tan(pitch))./(PosVzice-h)); % look angle of radar (synthetic beam pattern unaffected by mis-pointing, following beam steering)
    %WARNING!!! CLAUDE ADDED THE FOLLOWING LNIE:
    theta_lice = atan(-(PosVxice-x_0)./(PosVzice-h));


    % Compute angle between facet-normal vector and antenna-facet vector
    NormalAxice = cos(PHIice).*cos(pi/2 - THETAice);
    NormalAyice = sin(PHIice).*cos(pi/2 - THETAice);
    NormalAzice = -sin(pi/2 - THETAice);
    theta_price = pi - acos((NormalVxice.*NormalAxice + NormalVyice.*NormalAyice + NormalVzice.*NormalAzice)./(sqrt(NormalVxice.^2 + NormalVyice.^2 + NormalVzice.^2).*sqrt(NormalAxice.^2 + NormalAyice.^2 + NormalAzice.^2)));
    theta_price(theta_price > pi/2) = pi/2;

    % Ice surface echo from IEM and coherent backscattering theory
    mu_t = zeros(size(theta_price));
    mu_t(SURFACE_TYPE==1) = 10.^(ppval(sigma_0_ice_surf,theta_price(SURFACE_TYPE==1))/10).*ppval(tau_snow,theta_price(SURFACE_TYPE==1)).^2*exp(-kappa_e*h_s/2);
    
    mu_t_si_tracer = bsxfun(@times,P_t,mu_t);
    
    % Coherent reflection from water (leads or melt ponds) where necessary
    mu_t(SURFACE_TYPE==0) = 10.^(ppval(sigma_0_lead_surf,theta_price(SURFACE_TYPE==0))/10);
    mu_t(SURFACE_TYPE==2) = 10.^(ppval(sigma_0_mp_surf,theta_price(SURFACE_TYPE==2))/10);
    mu_t(isnan(mu_t)) = 0; % zero backscattered power at higher incidence over smooth water (change NaNs to zeros)
        
    mu_t_ocean_tracer = bsxfun(@times,P_t,mu_t) - mu_t_si_tracer;
    
    % Total echo (pre-convolved with transmitted pulse)
    sigma_0_P_t = vu_t_surf_tracer + vu_t_vol_tracer + mu_t_si_tracer + mu_t_ocean_tracer;
    
    % Backscatter component fraction tracer
    sigma_0_tracer(i,:,:) = [sum(vu_t_surf_tracer,1)./sum(sigma_0_P_t,1);sum(vu_t_vol_tracer,1)./sum(sigma_0_P_t,1);sum(mu_t_si_tracer,1)./sum(sigma_0_P_t,1);sum(mu_t_ocean_tracer,1)./sum(sigma_0_P_t,1)]';
    
    vu_t_surf_tracer = []; vu_t_vol_tracer = []; mu_t = []; mu_t_si_tracer = []; mu_t_ocean_tracer = [];
    
    %% Integrate power contributions from each facet
    
    % Integrate radar equation
    P_r = ((lambda^2*P_T)/(4*pi)^3)*(0.5*c*h)*bsxfun(@times, bsxfun(@rdivide, sigma_0_P_t, R.^4), G.^2.*P_m.*A_facets);
    
    echo_t = nansum(P_r,1);
    
    % Keeps memory use low during loop
    T = []; P_t = []; P_r = []; theta_PRsnow = []; theta_PRice = []; sigma_0_P_t = [];
    
    % Apply weighting to single-look echo stack
    P_t_full(i,:) = real(echo_t)*H(i);
        
end

% Multi-looked echo
P_t_ml = nansum(P_t_full,1)';

% Component echoes
P_t_full_comp = bsxfun(@times, sigma_0_tracer, P_t_full);
P_t_full = P_t_full';
P_t_full_comp = permute(P_t_full_comp,[2 1 3]);

P_t_ml_comp = permute(nansum(P_t_full_comp,2),[1 3 2]);


end

