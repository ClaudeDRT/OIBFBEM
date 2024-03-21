for ice_std_to_snow_std_ratio = [1]
%gam_file = "mesh04032_j129ns170nt87lam0p01hdfr0.8117511572862858ldfr1.2471343884639885.txt";
%gam_file = "mesh04032_j110ns170nt87lam0p01hdfr1.0176839234066961ldfr1.5109305459790376.txt";
%gam_file = "mesh04032_j103ns170nt87lam0p01hdfr0.7800703052197226ldfr1.3965045529750146.txt";

%gam_file = "mesh04032_j121ns170nt87lam0p01hdfr0.8326115936070405ldfr1.3195621292510473.txt";
%gam_file = "mesh04032_j118ns170nt87lam0p01hdfr0.8477556409004329ldfr1.3172748729450026.txt";
%gam_file = "mesh04032_j109ns170nt87lam0p01hdfr0.812489140731803ldfr1.4849937829127684.txt";
%gam_file = "mesh04032_j107ns170nt87lam0p01hdfr0.8215526156195672ldfr1.408834665208368.txt";
%gam_file = "mesh04032_j105ns170nt87lam0p01hdfr0.7938545835332939ldfr1.3680221384390139.txt";
%gam_file = "mesh04032_j112ns170nt87lam0p01hdfr1.0905164967376173ldfr1.4249376388475454.txt";
%airsnow_gam_file = "mesh04032_j112ns170nt87lam0p01hdfr1.0905164967376173ldfr1.4249376388475454.txt";
airsnow_gam_file = "mesh04032_j113ns170nt87lam0p01hdfr1.0476540983751386ldfr1.4459628370947835.txt";
%snowice_gam_file = "icemesh04032_j112ns170nt87lam0p01.txt";
snowice_gam_file = "icemesh04032_j113ns170nt87lam0p01.txt";


%insitu_estimated_snowdepth = snowdepth_returner(gam_file);
insitu_estimated_snowdepth = 0.15540172222069407;


%for h_s = [insitu_estimated_snowdepth-0.02 insitu_estimated_snowdepth insitu_estimated_snowdepth+0.02]
%for h_s = [0.19]

for ratio_of_beta_c_to_epsilon_b = [1 0.1 0.05 0.01 0.005 0.001 0.0005]
%for ratio_of_beta_c_to_epsilon_b = [0.1]
h_s = 0.19

%ratio_of_beta_c_to_epsilon_b = 0.005;

%for iterator = 1:4
%    if iterator==1
%        ratio_of_beta_c_to_epsilon_b = 0.0025
%    elseif iterator==2
%        ratio_of_beta_c_to_epsilon_b = 0.001
%    elseif iterator==3
%        ratio_of_beta_c_to_epsilon_b = 0.0005
%    elseif iterator==4
%        ratio_of_beta_c_to_epsilon_b = 0.0001
%    elseif iterator==5
%        ratio_of_beta_c_to_epsilon_b = 0.1
%    elseif iterator==6
%        ratio_of_beta_c_to_epsilon_b = 0.05
%    elseif iterator==7
%        ratio_of_beta_c_to_epsilon_b = 0.025
%    elseif iterator==8
%        ratio_of_beta_c_to_epsilon_b = 0.01
%    elseif iterator==9
%        ratio_of_beta_c_to_epsilon_b = 0.0075
%    elseif iterator==10
%        ratio_of_beta_c_to_epsilon_b = 0.005
%    elseif iterator==11
%        ratio_of_beta_c_to_epsilon_b = 0.0025
%    else 
%end
         
%Shell Script for using GAM surfaces in FBEM derived from OIB ATM laser point data
%Claude de Rijke-Thomas
%Uses Jack Landy's Facet-Based Echo Model ((c) J.C. Landy, University of
%Bristol, 2018)
%4th May 2023
%gam_file = "mesh04032_j100ns220nt100lam0p03hdfr1.0346281027463298ldfr1.3885290943764401.txt";
%gam_file = "mesh04032_j101ns220nt100lam0p0005hdfr1.0076837745730272ldfr1.3513073481757105.txt";
%gam_file = "mesh04032_j102ns220nt100lam0p0003hdfr1.0130267017629315ldfr1.3551172596421481.txt";
%gam_file = "mesh04032_j103ns220nt100lam0p0001hdfr1.0120952173718487ldfr1.4316082950159146.txt";
%gam_file = "mesh04032_j113ns180nt87lam0p0002hdfr1.2519008019219942ldfr1.5040877805898538.txt";
%gam_file = "mesh04032_j114ns180nt87lam0p0002hdfr1.2416901086714447ldfr1.5310229632264918.txt";
%gam_file = "mesh04032_j115ns180nt87lam0p0002hdfr0.9930749932414124ldfr1.3306031323196597.txt";
%gam_file = "mesh04032_j116ns180nt87lam0p00015hdfr1.0349498331246616ldfr1.3616701568335219.txt";
%gam_file = "mesh04032_j117ns180nt87lam7e-05hdfr0.9329123318688036ldfr1.0494990447990729.txt";
%gam_file = "mesh04032_j129ns170nt87lam0p01hdfr0.8117511572862858ldfr1.2471343884639885.txt";
%gam_file = "mesh04032_j102ns170nt87lam0p01hdfr0.8073556126979353ldfr1.3437508604898019.txt";


%airsnow_gam_file = "mesh04032_j113ns170nt87lam0p01hdfr1.0476540983751386ldfr1.4459628370947835.txt";
%snowice_gam_file = "icemesh04032_j113ns170nt87lam0p01.txt";
airsnow_gam_file = "mesh04032_j110ns170nt87lam0p01hdfr1.0176839234066961ldfr1.5109305459790376.txt";
snowice_gam_file = "icemesh04032_j110ns170nt87lam0p01.txt";

%gam_file = "mesh04032_j100ns170nt87lam0p01hdfr1.1519669253775573ldfr1.3628073363797166.txt";
%gam_file = "mesh04032_j101ns170nt87lam0p01hdfr0.8113131124157394ldfr1.3476972699334568.txt";
%gam_file = "mesh04032_j103ns170nt87lam0p01hdfr0.7800703052197226ldfr1.3965045529750146.txt";
%gam_file = "mesh04032_j110ns170nt87lam0p01hdfr1.0176839234066961ldfr1.5109305459790376.txt";
%gam_file = "mesh04032_j112ns170nt87lam0p01hdfr1.0905164967376173ldfr1.4249376388475454.txt";
%gam_file = "mesh04032_j113ns170nt87lam0p01hdfr1.0476540983751386ldfr1.4459628370947835.txt";
%gam_file = "mesh04032_j114ns170nt87lam0p01hdfr1.0397050634646212ldfr1.4816831919861924.txt";
%gam_file = "mesh04032_j121ns170nt87lam0p01hdfr0.8326115936070405ldfr1.3195621292510473.txt";
%gam_file = "mesh04032_j129ns170nt87lam0p01hdfr0.8117511572862858ldfr1.2471343884639885.txt";
%gam_file = "mesh04032_j137ns170nt87lam0p01hdfr0.8356071685902801ldfr1.4210284279174248.txt";
%gam_file = "mesh04032_j147ns170nt87lam0p01hdfr0.8381953232449977ldfr1.4092615408004556.txt";

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
sigma_s = [0.0005 0.001 0.00125 0.0015 0.00175 0.002 0.003 0.0045]; % snow rms height (default = 0.001 m)
%sigma_s = [0.0015 0.002 0.003 0.0045];
%sigma_s = [0.001 0.002]
l_s = 0.0148; % snow correlation length (default = 0.04 m)
T_s = -22.060638297872337; % snow bulk temperature (default = -20 C)
rho_s = 328.50909090909086; % snow bulk density (default = 350 kg/m^3)
r_s = 0.001; % snow grain size (normal range from 0.0001 to 0.004 m, default 1 mm)



sigma_si = [0.0005 0.00075 0.001 0.00125 0.0015 0.00175 0.002 0.004]; % sea ice rms height (default = 0.002 m)
%sigma_si = [0.0015 0.0017 0.002]
l_si = 0.0148; % sea ice correlation length (default = 0.02 m)
% phi_pr = 1*(pi/180); % polar response angle for Giles et al., 2007 simplified surface scattering function
T_si = -17; % sea ice bulk temperature (default = -15 C)
S_si = 17.21111111111111; % sea ice bulk salinity (default = 6 ppt)
sigma_sw = 0.00001; % lead rms height (default = 0.00001 m)
T_sw = 0; % temperature of seawater (default = 0 C)
S_sw = 34; % salinity of seawater (default = 34 ppt)


kappa_es = [1.2141 1.7 2 3 5.67];
%kappa_es = [1.2141 2]
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


epsilon_b = lambda/(2*N_b*v*(1/prf)); % angular resolution of beams from full look crescent (beam separation angle)
beta_c = epsilon_b*ratio_of_beta_c_to_epsilon_b; % no wider than synthetic beam angle epsilon_b, rads

PARAMETERS = whos('*');
parts = strsplit(airsnow_gam_file, '_j');
firstsubparts = strsplit(parts{1}, 'mesh');
fileno = string(firstsubparts{2});
subparts = strsplit(string(parts(2)), 'ns');
index = string(str2num(subparts(1)));
%save(append('FEM_SimulationsOIB',fileno,"j",index ,"modFBEMkesBc0p007PostTice",strrep(string(round(ice_std_to_snow_std_ratio,2,"significant")), '.', 'p')));
save(append('FInterfaces_',fileno,"j",index ,"modFBEMsnowdepth",strrep(string(round(h_s,2,"significant")), '.', 'p'),"Bc",strrep(string(round(beta_c,3,"significant")),'.','p'), "PostTice",strrep(string(round(ice_std_to_snow_std_ratio,2,"significant")), '.', 'p')));

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
%[PosT,surface_type] = gam_topo_shell(gam_file,dx,L_w,L_h,D_off,f_p,L_ang);
% Synthetic topography
[PosTairsnow,surface_type] = gam_topo_shell(airsnow_gam_file,dx,L_w,L_h,D_off,f_p,L_ang);
[PosTsnowice,surface_type] = gam_topo_shell(snowice_gam_file,dx,L_w,L_h,D_off,f_p,L_ang);

%disp('the class of PosT(:,2) is: ');
%class(PosT(:,2))
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


            %%[theta,sigma_0_snow_surf,sigma_0_snow_vol,kappa_e,tau_snow,c_s,epsr_ds] = snow_backscatter(lambda,sigma_s(akak),l_s,T_s,rho_s,r_s,h_s,beta_c,kappa_e);
            itN = 1; % Number of iterations
            P_t_full = zeros(length(t),N_b,itN);
            P_t_ml = zeros(length(t),itN);
            P_t_full_comp = zeros(length(t),N_b,4,itN);
            P_t_ml_comp = zeros(length(t),4,itN);
    
            %there are a lot of ones in this line because that's the current iteration number (there's only one iteration per simulation)
            %[P_t_full(:,:,1),P_t_ml(:,1),P_t_full_comp(:,:,:,1),P_t_ml_comp(:,:,1)] = Facet_Echo_Model(lambda,bandwidth,P_T,h,v,pitch,roll,...
            %                                                                                           prf,G_0,D_0,N_b,t,PosT,surface_type,sigma_0_snow_surf,...
            %                                                                                           sigma_0_snow_vol,kappa_e,tau_snow,c_s,h_s,sigma_0_ice_surf,sigma_0_lead_surf,...
            %                                                                                           sigma_0_mp_surf);
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

%save(append('FEM_SimulationsOIB',fileno,"j",index,"modFBEMkesBc0p007PostTice",strrep(string(round(ice_std_to_snow_std_ratio,2,"significant")), '.','p')),'t','P_t_full_range','P_t_ml_range','P_t_full_comp_range','P_t_ml_comp_range','-append');

save(append('FInterfaces_',fileno,"j",index ,"modFBEMsnowdepth",strrep(string(round(h_s,2,"significant")),'.','p'),"Bc",strrep(string(round(beta_c,3,"significant")),'.','p'),"PostTice",strrep(string(round(ice_std_to_snow_std_ratio,2,"significant")),'.','p')),...
      't','P_t_full_range','P_t_ml_range','P_t_full_comp_range','P_t_ml_comp_range','-append');

clearvars -except ice_std_to_snow_std_ratio

end
% CLEAR WORKSPACE WHEN STARTING A NEW SET OF SIMULATIONS
end
