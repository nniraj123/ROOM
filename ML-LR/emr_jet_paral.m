clear all; close all
rng('default');
%nn = 10000;  % No. of time steps
%nn1 = round(nn/2);
nn = 5000;
nn1 = nn;
ii = 513; % Grid size for each time step
day_diff_save = 10; %difference between two consequitive time-frames in the data

fileID = fopen('psi1_DG.dat_0_50K'); %read in data for the first layer
%fileID1 = fopen('psi1_DG.dat_50K_100K'); %read in data for first layer
jj = 64;
kk = floor(ii/jj);
kk1 = [10, 50]; % choose a specific region in the flow
sz_temp = [size(1:kk:ii-1,2), size(kk1(1)*kk:kk:kk1(2)*kk,2)];
zeta1 = zeros(nn,sz_temp(1)*sz_temp(2)); %initialise data vector
for k = 1:nn1
    zeta(:,:) = fread(fileID,[ii ii],'single');
    zeta0 = zeta(1:kk:ii-1,kk1(1)*kk:kk:kk1(2)*kk);
    zeta0 = reshape(zeta0,1,[]);
    zeta1(k,:) = zeta0;
end
%{
for k = nn1+1:nn
    zeta(:,:) = fread(fileID1,[ii ii],'single');
    zeta0 = zeta(1:kk:ii-1,kk1(1)*kk:kk:kk1(2)*kk);
    zeta0 = reshape(zeta0,1,[]);
    zeta1(k,:) = zeta0;
end
%zeta1(end+1:2*end,:) = zeta1;
%
%}
% new extended data frames to capture LFV, could be obtained by averaging
% or skipped from the original data
%zeta1 = repmat(zeta1,4,1); % to capture the LFV, we need 4 time more data if using M = 150
k_time = 1;
day_diff_save_used = day_diff_save * k_time; 
zeta1 = zeta1(1:k_time:end,:);
[coeff,score,~,~,expl,mu] = pca(zeta1,'Economy',true);
%{
%score(:,1:ndim)*coeff(:,1:ndim)
%[residuals,reconstructed] = pcares(zeta1,pc_n);

W=200;
figure
%for pc_n = 2:2:60
pc_n = 30;
%x = coeff(:,1:pc_n)*score(:,1:pc_n)';
%x = coeff(:,1:pc_n);
x = score(:,1:pc_n);
X = center(x);
%}
% embedding time-window - all the periodicities to reveal are aasumed to be
% within the time-window (the number of time frames) 
% to capture LFV one need to have a window with the corresponding smallest
% frequency less than 1/17 years (to be safe 1/20 years). W_dim should be 
% ~7300 days. And the highest frequency is 0.5*365/day_diff_save_used
W = 50*round(2/k_time);
W_dim = W*day_diff_save_used;
freq_dim = [365/W_dim 0.5*365/day_diff_save_used]; % freq domain in years

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use compressed data composed by the pc_n number of PCs. To recover the
% original compressed temporal - spatial field 
% data[i x j] = score[i x pc_n] * coeff[j x pc_n]'
% score - PCs, coeff - EOFs
pc_n = 30; % number of PCs
pc_n0 = 1; % from which PC to start 
X = center(score(:,pc_n0:pc_n0+pc_n-1));

% DAHD decomposition, V - eigenvalues stacked into frequency bins over the
% interval [0 0.5] with W values. 

[E,V,fE1,EP,VP,fE2] = DAHD(X,W,0);

MM = 2*size(EP,3)-1;
MM2 = (MM-1)/2;

[~, i0] = max(max(VP)); % frequency number (the largest eigenvalue)
%%%%%%%Power Spectrum %%%%%%%%%
figure
set(gca,'FontSize',16);
semilogy(fE1*2*freq_dim(2),V,'ro','MarkerSize',4,'MarkerFaceColor','r');
hold on
semilogy(fE2(i0)*2*freq_dim(2), abs(VP(1:2*pc_n,i0)),'bo','MarkerSize',4,'MarkerFaceColor','b')
xlim([0 freq_dim(2)]);
ylim([0.1E5 10.0E9]);
xlabel('Freq')
filename = strcat(sprintf('Power spectrum for %d principal component starting from %d for [%d-%d]', pc_n, pc_n0, kk1*kk), ' time window ', num2str(W*day_diff_save_used),' days', ' num PCs ', num2str(pc_n));
title(filename);
grid on
print(filename,'-dpng');

K = 2*pc_n; % should be even and <= 2*pc_n
% In EP everything is arranged by pairs (the number of pairs is less or equal the number of retained PCs)

% Brownian noise to feed into the emulators, same for every frequency
tlength = size(zeta1,1);
dW = randn(tlength,K);
mod_freq = 1:W; % max = W
xx = zeros(tlength,K,size(mod_freq,2));
diverge = false(size(mod_freq,2),1);
K_modes = 1:K; %K can be changed to a smaller number

% Constructing harmonic components (DAHD continuation)
for i0 = mod_freq
    tmp = squeeze(EP(:,K_modes,i0));
    A0 = dahc(X,tmp);
    data0(:,:,i0) = A0(:,K_modes);
end
nlevels = 2; % always >= 2 (including the base level)
parfor i0 = mod_freq        
    disp(['Fit EMR Model for ', num2str(i0), '-th frequency out of ', num2str(W)]);        
    data = data0(:,:,i0);
    %%%%%%%%%%%%%%%%%%%%%%%%% Multi-Layer Stuart-Landau emulators    
    if i0 == 1
        % unpaired zero frequency dahc
        % moddeled by a linear regression
        [xx1,diverge1] = fit_Linear_MultiLayer(tlength,data(:,1:0.5*K),nlevels,dW(:,1:0.5*K),0,0);
        xx1 = [xx1 zeros(tlength,0.5*K)];
    else
        [xx1,diverge1] = fit_MLSL_MultiLayer(tlength,data,nlevels,dW,0,0);
        if diverge1 % if the nonlinear model diverges            
            [xx1,~] = fit_Linear_MultiLayer(tlength,data,nlevels,dW,0,0);                
        end
    end
    xx(:,:,i0) = xx1; diverge(i0) = diverge1;
end
%%%%%PROCESSING OUTPUT AND COMPARE WITH ORIGINAL DATA %%%%%%%
% Reconstruction of the physical space with the EMR modelled DAHCs
RECFLOW = 0;
ORFLOW = 0;
for i0 = mod_freq
    tmp = squeeze(EP(:,:,i0));    
    R0 = hrc(data0(:,:,i0),tmp,size(X,2));
    R1 = hrc(xx(:,:,i0),tmp,size(X,2));    
    % ORFLOW = ORFLOW + R0*coeff(:,1:pc_n)';
    % RECFLOW = RECFLOW + R1*coeff(:,1:pc_n)';
end

time_sub = 10;    
figure    
for i=1:time_sub
    subplot(5,floor(time_sub/5)+1,i)
    psi = reshape(RECFLOW(100*i,:),64,[])';
    surf(psi);
    view(2)
    shading interp
    colorbar
    if i == 1, lim1 = 3*std2(psi); end
    caxis([-lim1 lim1])
    xlim([0 jj]);
    ylim([0 size(zeta0,2)/jj]);
    title(i)
end

% plot mean ACF
tt = 1:min(size(data0,1),size(xx,1));
MM_ = size(tt,2)-1;
figure
xd = xcorr(center(mean(ORFLOW,2)),MM_,'coeff');
plot(tt,xd(1+MM_:end),'LineWidth',2);
hold on    
xd = xcorr(center(mean(RECFLOW,2)),MM_,'coeff');    
plot(tt,xd(1+MM_:end),'r','LineWidth',2);    
legend('Data','EMR');
title('ACF of the mean data')
    
% plot PDF
figure
[f,xi] = ksdensity(center(mean(ORFLOW,2)));
plot(xi,f,'r','LineWidth',2);
hold on
[f,xi] = ksdensity(center(mean(RECFLOW,2)));
plot(xi,f,'b','LineWidth',2);
legend('Data','EMR');
title('PDF of the mean data')

% plot ACF
k_sz = size(data0,2);
sz1 = floor(sqrt(k_sz));
figure
for i=1:k_sz
    subplot(sz1,floor(k_sz/sz1)+1,i)
    xd = xcorr(center(ORFLOW(:,i)),MM_,'coeff');
    plot(tt,xd(1+MM_:end),'LineWidth',2);
    hold on    
    xd = xcorr(center(RECFLOW(:,i)),MM_,'coeff');    
    plot(tt,xd(1+MM_:end),'r','LineWidth',2);    
    %legend('Data','EMR');
    title(i)
end
% plot PDF
figure
for i=1:k_sz
    subplot(sz1,floor(k_sz/sz1)+1,i)
    [f,xi] = ksdensity(center(ORFLOW(tt,i)));
    plot(xi,f,'r','LineWidth',2);
    hold on
    [f,xi] = ksdensity(center(RECFLOW(tt,i)));
    plot(xi,f,'b','LineWidth',2);
    %legend('Data','EMR');
    title(i)
end