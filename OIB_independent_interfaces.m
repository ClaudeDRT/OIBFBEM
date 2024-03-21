
for gam_file_no = 1:25

for beta_c = [0.018949175726928103 0.026799568726957428 0.037898351453856206 0.04458629582806613 0.07579670290771241]


%Shell Script for using GAM surfaces in FBEM derived from OIB ATM laser point data
%Claude de Rijke-Thomas
%Uses Jack Landy's Facet-Based Echo Model ((c) J.C. Landy, University of
%Bristol, 2018)
%4th May 2023
ice_std_to_snow_std_ratio = 1

if gam_file_no==1
    gam_file_start = "mesh04032_j100"
    h_s = 0.21599583259509437
elseif gam_file_no==2
    gam_file_start = "mesh04032_j101"
    h_s = 0.24685238010867927
elseif gam_file_no==3
   gam_file_start = "mesh04032_j102"
   h_s = 0.24685238010867927
elseif gam_file_no==4
   gam_file_start = "mesh04032_j103"
   h_s = 0.26742341178440254
elseif gam_file_no==5
   gam_file_start = "mesh04032_j105"
   h_s = 0.3085654751358491  % air-snow interface is stronger than snow-ice interface
elseif gam_file_no==6
    gam_file_start = "mesh04032_j105"
    h_s = 0.16456825340578618
elseif gam_file_no==7
    gam_file_start = "mesh04032_j106"
    h_s = 0.13371170589220127
elseif gam_file_no==8
    gam_file_start = "mesh04032_j107"
    h_s = 0.12342
elseif gam_file_no==9
    gam_file_start = "mesh04032_j108"
    h_s = 0.17485376924364782
elseif gam_file_no==10
    gam_file_start = "mesh04032_j109"
    h_s = 0.17485376924364782
elseif gam_file_no==11
    gam_file_start = "mesh04032_j110"
    h_s = 0.16456825340578618;
elseif gam_file_no==12
    gam_file_start = "mesh04032_j111"
    h_s = 0.13371170589220127
elseif gam_file_no==13
    gam_file_start = "mesh04032_j112";
    h_s = 0.17485376924364782
elseif gam_file_no==14
    gam_file_start = "mesh04032_j113"
    h_s = 0.21599583259509436
elseif gam_file_no==15
    gam_file_start = "mesh04032_j114"
    h_s = 0.2879944434601258
elseif gam_file_no==16
    gam_file_start = "mesh04032_j115"
    h_s = 0.2879944434601258
elseif gam_file_no==17
    gam_file_start = "mesh04032_j116"
    h_s = 0.32913650681157236
elseif gam_file_no==18
    gam_file_start = "mesh04032_j117"
    h_s = 0.32913650681157236
elseif gam_file_no==19
    gam_file_start = "mesh04032_j118"
    h_s = 0.24685238010867927
elseif gam_file_no==20
    gam_file_start = "mesh04032_j119"
    h_s = 0.2777089276222642 %bigger air-snow peak
elseif gam_file_no==21
    gam_file_start = "mesh04032_j121";
    h_s = 0.21599583259509436
elseif gam_file_no==22
    gam_file_start = "mesh04032_j128"
    h_s = 0.13371170589220127
elseif gam_file_no==23
    gam_file_start = "mesh04032_j129";
    h_s = 0.13371170589220127
elseif gam_file_no==24
    gam_file_start = "mesh04032_j137";
    h_s = 0.2879944434601258
elseif gam_file_no==25
    gam_file_start = "mesh04032_j147";
    h_s = 0.41142063351446545

end

snowice_gam_file_start = string(append("ice", gam_file_start)); 

parts = strsplit(gam_file_start, '_j');
subparts = strsplit(parts(1), 'mesh')
fileno = subparts(2)
index = parts(2)
%fileno_and_echono = strsplit(firstsubparts(2), '_')
%fileno = fileno_and_echono(1);
%index = fileno_and_echono(2);


% Included Codes ((c) Jack Landy):
% Facet_Echo_Model.m
% RelDielConst_Brine.m
% rsgene2D_anisotrop.m
% synthetic_topo_shell.m
% snow_backscatter.m
% ice_backscatter.m
% lead_backscatter.m

% Uses the following codes from external sources:
% computeNormalVectorTriangulation.m (David Gingras)
% I2EM_Backscatter_model.m (Fawwaz Ulaby)
% MieExtinc_DrySnow.m (Fawwaz Ulaby)
% RelDielConst_PureIce.m (Fawwaz Ulaby)
% RelDielConst_DrySnow.m (Fawwaz Ulaby)
% RelDielConst_SalineWater.m (Fawwaz Ulaby)
% TVBmodel_HeterogeneousMix.m (Fawwaz Ulaby)
% artifical_surf.m (Mona Mahboob Kanafi)


%The ATM laser data has been altimetrically aligned across multiple
%flyovers to increase the horizontal resolution

% CLEAR WORKSPACE WHEN STARTING A NEW SET OF SIMULATIONS

%% Model Variables (MODIFIABLE)

% Up to 3 variables can be vectors

% Geophysical parameters
sigma_s = [0.0005 0.001 0.00125 0.0015 0.00175 0.002 0.003 0.0045 0.007]; % snow rms height (default = 0.001 m)
l_s = 0.0148; % snow correlation length (default = 0.04 m)
T_s = -22.060638297872337; % snow bulk temperature (default = -20 C)
rho_s = 328.50909090909086; % snow bulk density (default = 350 kg/m^3)
r_s = 0.001; % snow grain size (normal range from 0.0001 to 0.004 m, default 1 mm)



sigma_si = [0.0005 0.001 0.00125 0.0015 0.00175 0.002 0.0025]; % sea ice rms height (default = 0.002 m)
l_si = 0.0148; % sea ice correlation length (default = 0.02 m)
% phi_pr = 1*(pi/180); % polar response angle for Giles et al., 2007 simplified surface scattering function
T_si = -17; % sea ice bulk temperature (default = -15 C)
S_si = 17.21111111111111; % sea ice bulk salinity (default = 6 ppt)
sigma_sw = 0.00001; % lead rms height (default = 0.00001 m)
T_sw = 0; % temperature of seawater (default = 0 C)
S_sw = 34; % salinity of seawater (default = 34 ppt)


kappa_es = [1.2141 2 3 5.67];

% Antenna parameters
lambda = (299792458/(14.75*10^9)); % radar wavelength (default = 0.0221, Ku-band e.g. Cryosat-2)
GP = whos('*'); % all parameters controlling scattering signatures

gammabar = 0.012215368000378016; % Cryosat-2 antenna pattern term 1
gammahat = 0.0381925958945466; % Cryosat-2 antenna pattern term 2
gamma1 = sqrt(2/(2/gammabar^2+2/gammahat^2)); % along-track antenna parameter
gamma2 = sqrt(2/(2/gammabar^2-2/gammahat^2)); % across-track antenna parameter

op_mode = 2; % operational mode: 1 = pulse-limited, 2 = SAR (PL-mode only feasible on high memory machines)
beam_weighting = 2; % weighting on the beam-wise azimuth FFT: 1 = rectangular, 2 = Hamming (default = Hamming)
P_T = 2.188e-5; % transmitted peak power (default = 2.188e-5 watts)

pitch = 0; % antenna bench pitch counterclockwise (up to ~0.001 rads)
roll = 0; % antenna bench roll counterclockwise (up to ~0.005 rads)

h = 515; % satellite altitude (default = 720000 m)
v = 121.13638615818662; % satellite velocity (default = 7500 m/s)
N_b = double(16); % no. beams in synthetic aperture (default = 64, e.g. Cryosat-2)
prf = 1953.125; % pulse-repitition frequency (default = 18182 Hz, e.g. Cryosat-2)
bandwidth = 5776000000.0; % antenna bandwidth (default = 320*10^6 Hz, e.g. Cryosat-2)
G_0 = 42; % peak antenna gain, dB
D_0 = 36.12; % synthetic beam gain, SAR mode

% Number of range bins
N_tb = 150; % (default = 70)

% Range bin at mean scattering surface, i.e. time = 0
t_0 = 60; % (default = 15)

% Time oversampling factor
t_sub = 1;

% Parameters of synthetic topography
topo_type = 1; % type of surface: 1 = Gaussian, 2 = lognormal, 3 = fractal
sigma_surf = 0.1; % large-scale rms roughness height (default = 0.1 m)
l_surf = 5; % large-scale correlation length (default = 5 m)
H_surf = 0.5; % Hurst parameter (default = 0.5)
dx = 0.25; % resolution of grid, m (WARNING use dx>=10 for PL mode and dx>=5 for SAR mode)

% Lead parameters (optional)
L_w = 0; % lead width (default = 100 m)
L_h = 0; % lead depth (default = 0.2 m)
D_off = 0; % distance off nadir (default = 0 m)
L_ang = 0; %lead angle from the across-track direction (between 0 deg and 90 deg)

% Add melt ponds (optional)
T_fw = 0; % temperature of freshwater in pond (default = 0 C)
f_p = 0; % melt pond fraction (default = 0.5)
u_a = 4; % boundary-layer wind speed (default = 4 m/s)


PARAMETERS = whos('*');


save(append('both/IndInters',fileno,"j",index ,"modFBEMBc",strrep(string(round(beta_c,2,"significant")), '.', 'p'),"NoOmegaAndExtraTwo4sOldGAMh_s",strrep(string(round(h_s,2,"significant")), '.', 'p')));

%% Antenna Geometry



%% Loop Echo Model

% Use parallel processing
% parpool

% Identify vector variables


% Loop model over vector variables
P_t_full_range = cell(length(sigma_s),length(sigma_si));
P_t_ml_range = cell(length(sigma_s),length(sigma_si));
P_t_full_comp_range = cell(length(sigma_s),length(sigma_si));
P_t_ml_comp_range = cell(length(sigma_s),length(sigma_si));
parameters_lookup = cell(length(sigma_s),length(sigma_si));


% Effective width of angular extent of coherent component (TUNING PARAMETER FOR LEADS)
[~,sigma_0_lead_surf] = lead_backscatter(lambda,sigma_sw,T_sw,S_sw,beta_c);
[~,sigma_0_mp_surf] = pond_backscatter(lambda,T_fw,beta_c,u_a);


% Synthetic topography
[PosTairsnow, surface_type] = gam_snow_topo_shell(gam_file_start,dx,L_w,L_h,D_off,f_p,L_ang);
[PosTsnowice,surface_type] = gam_ice_topo_shell(snowice_gam_file_start,dx,L_w,L_h,D_off,f_p,L_ang);


%initialising empty arrays:
sigma_0_ice_surfs={};



epsr_ds = 1.601871268;


for akai = 1:length(kappa_es)
    kappa_e = kappa_es(akai);
    
    sigma_0_snow_surfs = {};
    sigma_0_snow_vols = {};
    tau_snows = {};

    for akaj = 1:length(sigma_si)
        if akai==1
            disp('i==1');
            [~,sigma_0_ice_surf,~] = ice_backscatter(lambda,sigma_si(akaj),l_si,T_si,S_si,h_s,beta_c,epsr_ds);
            sigma_0_ice_surfs=[sigma_0_ice_surfs,sigma_0_ice_surf];
        else
            disp('i!=1');
            sigma_0_ice_surf = sigma_0_ice_surfs{akaj};
            %this may need to be si..(j) then hi = ...{1}
        end
        % Time domain
        t = (0.5/bandwidth)*((1:(1/t_sub):N_tb) - t_0);
        
        for akak = 1:length(sigma_s)
            if akaj==1
                [theta,sigma_0_snow_surf,sigma_0_snow_vol,~,tau_snow,c_s,~] = snow_backscatter(lambda,sigma_s(akak),l_s,T_s,rho_s,r_s,h_s,beta_c,kappa_e);
                sigma_0_snow_surfs = [sigma_0_snow_surfs,sigma_0_snow_surf];
                sigma_0_snow_vols = [sigma_0_snow_vols,sigma_0_snow_vol];
                tau_snows = [tau_snows,tau_snow];
            else
                sigma_0_snow_surf = sigma_0_snow_surfs{akak};
                sigma_0_snow_vol = sigma_0_snow_vols{akak};
                tau_snow = tau_snows{akak};
            end


            itN = 1; % Number of iterations
            P_t_full = zeros(length(t),N_b,itN);
            P_t_ml = zeros(length(t),itN);
            P_t_full_comp = zeros(length(t),N_b,4,itN);
            P_t_ml_comp = zeros(length(t),4,itN);
    
            [P_t_full,P_t_ml,P_t_full_comp,P_t_ml_comp] = Facet_Echo_Model_independent_interfaces(op_mode,lambda,bandwidth,P_T,h,v,pitch,roll,prf,beam_weighting,G_0,D_0,gamma1,gamma2,N_b,t,...
                                                                                                  PosTairsnow,PosTsnowice,surface_type,sigma_0_snow_surf,sigma_0_snow_vol,kappa_e,...
                                                                                                  tau_snow,c_s,h_s,sigma_0_ice_surf,sigma_0_lead_surf,sigma_0_mp_surf, ice_std_to_snow_std_ratio);

            fprintf(['Iteration ' num2str(1) '/' num2str(itN) ', Simulation ' num2str(sum(~cellfun(@isempty,P_t_ml_range(:)))+1) '/' ...
                     num2str(length(sigma_s)*length(sigma_si)*length(kappa_es)) '\n']);

            parameters_lookup{akai,akaj,akak} = [kappa_es(akai) sigma_si(akaj) sigma_s(akak)];
            P_t_full_range{akai,akaj,akak} = nanmean(P_t_full,3);
            P_t_ml_range{akai,akaj,akak} = nanmean(P_t_ml,2);
            P_t_full_comp_range{akai,akaj,akak} = nanmean(P_t_full_comp,4);
            P_t_ml_comp_range{akai,akaj,akak} = nanmean(P_t_ml_comp,3);
        end
    end
end

save(append('both/IndInters',fileno,"j",index ,"modFBEMBc",strrep(string(round(beta_c,2,"significant")),'.','p'),"NoOmegaAndExtraTwo4sOldGAMh_s",strrep(string(round(h_s,2,"significant")),'.','p')),...
      't','P_t_full_range','P_t_ml_range','P_t_full_comp_range','P_t_ml_comp_range','-append');

clearvars -except gam_file_no

end
% CLEAR WORKSPACE WHEN STARTING A NEW SET OF SIMULATIONS
end
