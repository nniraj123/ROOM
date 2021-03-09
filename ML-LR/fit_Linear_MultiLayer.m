function [xx,diverge]= fit_Linear_MultiLayer(tlength,data,nlevels,dW,inorm,iplot)
%{
data = data0(:,1:end);
nlevels = 3;
inorm = 0;
iplot = 0;
tlength = round(size(data0,1));
dW = randn(tlength,npc);
%}
% LINEAR MULTILAYER REGRESSION
% DAHD data comes in pairs. The subroutine models simultaneously all the pairs of
% DAHCs for a given frequency. 
% INPUT : data - multivariate array [N,L], N - length of the time
% series, L = 2*npairs - number of dahcs,
%                each channel is centered prior to EMR fitting;
%         nlevel - total number of levels in the EMR model including main and base levels;
%         tlength - length of the simulated time series by EMR
%		  inorm - controls data normalization:
%                   inorm = 0 - default, normalizes all channels by std(data(:,1));%                  
%                   inorm = 1 - fit non-normalized data%
%         iplot = 1 - plot variance convergence rate
%--------------------------------------------------------------------------------
%---------------------------------------------------------------------------------
% OUTPUT: xx - multivariate array [tlength,2*npairs], successfully simulated data by EMR
%         diverge = true if the modelled solution diverges
diverge = false;

% DAHCs come in pairs npc is always even
[length, npc] = size(data);
%%%%%%%%%% x - MAIN ARRAY FOR MULTI-LEVEL EMR VARIABLES%%%%%%
aa_number = zeros(nlevels-1,1);
aa_number(1) = npc;
aa_number(2:end) = (2:nlevels-1);

aa = zeros(max(aa_number),nlevels,npc);% regression coeffs
mnd = zeros(nlevels,npc); % mean at each layer 
std_r = ones(nlevels,npc); % std at each layer
varr = zeros(nlevels,npc); % variance error
x = zeros(length,nlevels+1,npc);

% make data standard
mnd(1,:) = mean(data);
data = data - mnd(1,:); 
%%%%%% Normalize if specified %%%%%%%%%%%%%%%%
std_r(1,:) = std(data);
if inorm == 0
    data = data ./ std_r(1,:);
end
%%%%%%%%%%
options = optimoptions('lsqlin','Algorithm','interior-point','Display','off');

for np = 1:npc    
    x(:,1,np) = data(:,np);    
    %idx = 1:npc; idx(np) = [];
    %x_data = data(:,idx); 
    %%%% GRAND LOOP OVER LEVELS, STARTING WITH MAIN nl=1%%%%
    for nl = 1:nlevels-1 % without the lowest level
        %%%%%%%%%%%%%CONSTRUCT predictors (TENDENCIES BY EULER TIME-DIFFERENCING)%%%%%%%%%%%%%%%%%%%%
        xt = squeeze(diff(x(1:end-nl+1,nl,np)));
        %%%%%%%%%%%%%% fitting multivariate regression        
        if nl == 1
            % block matrix of predictors
            %CC = [squeeze(x(1:end-nl,nl,np)), x_data(1:end-nl,:)];
            CC = data(1:end-nl,:);
            %%%%%%%%%%%%%%%%%%%
            [param_MLSL,~,residual,~,~,~] = lsqlin(CC,xt,[],[],[],[],[],[],[],options);
            residual = -residual;
        else                        
            CC = x(1:end-nl,1:nl,np);            
            [param_MLSL,~,residual,~,~,~] = lsqlin(CC,xt,[],[],[],[],[],[],[],options);            
            residual = -residual;
        end
        % variance error estimate
        sst = sum(xt.^2);
        sse = sum(residual.^2);        
        varr(nl,np)=1-sse/sst;
        %%%%%%%%%%%%%%%%%%%%%%%%%                
        aa(1:aa_number(nl),nl,np) = param_MLSL;        
        
        mnd(nl+1,np) = mean(residual);
        residual = residual - mnd(nl+1,np);
        %%%%%% Normalize if specified %%%%%%%%%%%%%%%%
        if inorm == 0
            std_r(nl+1,np) = std(residual);
            residual = residual ./ std_r(nl+1,np);
        end
        x(1:end-nl,nl+1,np) = residual;
    end %END OF GRAND LOOP OVER LEVELS    
end % end of the loop over the DAHC pairs

% convergence as the number of layers increases
if iplot == 1
    figure
    set(gca,'FontSize',12);
    plot([1 nlevels],[0.5 0.5],'r','LineWidth',2);
    hold on
    legend('Optimal');
    plot(mean(varr(1:end-1,:),2),'b.-','LineWidth',2,'MarkerFaceColor','b','MarkerSize',20);
    title('Convergence')
    xlabel('Number of levels');
    set(gca,'FontSize',12);
end

covn = corrcoef(squeeze(x(:,nlevels,:)));
stdr = std_r(nlevels,:);
[rr,p] = chol(covn);
if p ~= 0
    data = data .* std_r(1,:) + mnd(1,:); nlevels = nlevels + 1;    
    [xx,diverge] = fit_Linear_MultiLayer(tlength,data,nlevels,dW,inorm,iplot);    
    return
end
%{
%%%%CHOLESKY DECOMPOSITION OF CORRELATION MATRIX OF RESIDUAL NOISE AT THE LAST LEVEL
disp(['CHOLESKY DECOMPOSITION OF CORRELATION MATRIX OF RESIDUAL NOISE']);
disp([num2str(rr)]);
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
disp(['PERFORM EMR MODEL SIMULATIONs ' num2str(tlength) ' steps']);

%%%%EMR MODEL SIMULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eps = zeros(tlength,nlevels+1,npc);
eps_tmp = zeros(tlength,nlevels+1,npc);

xinit = squeeze(x(end,1:nlevels,:));
% correlated red noise initialisation for every time-step
dW = dW*rr.*stdr;
for l = 1:tlength
    xinit_dif = zeros(nlevels,npc);
    for np = 1:npc        
        % collecting elements at the lowest level with Brownian motion input        
        eps_tmp(l,nlevels,np) = dW(l,np); %xinit(nl,pair_c);%  squeeze(x(l,nlevels,pair_c))' .* std_r(nlevels,pair_c) + mnd(nlevels,pair_c);
        %eps(l,nlevels,np) = squeeze(x(l,nlevels,np))' .* std_r(nlevels,np) + mnd(nlevels,np);
        %eps(l,nlevels,np) = 0;
        % collecting elements at the intermediate levels
        for nl = nlevels-1:-1:2
            x_low = xinit(1:nl,np)';
            zet = (x_low * aa(1:aa_number(nl),nl,np)) + xinit(nl,np);
            eps(l,nl,np) = squeeze(eps_tmp(l,nl+1,np))' + zet;
            eps_tmp(l,nl,np) = squeeze(eps(l,nl,np))'.* std_r(nl,np) + mnd(nl,np); % return mean and std;
        end
        % collecting elements at the main level
        nl = 1;
        eps(l,nl,np) = squeeze(xinit(nl,:)) * aa(1:aa_number(nl),nl,np) + squeeze(eps_tmp(l,nl+1,np)) + xinit(nl,np);                
        xinit_dif(1:nlevels,np) = squeeze(eps(l,1:nlevels,np));
    end
    %check whether the solution diverges
    if any(squeeze(isnan(eps(l,1,:))))
        diverge = true;
        eps(l-100:end,:,:) = []; 
        break
    end
    xinit = xinit_dif;
end
xx = squeeze(eps(:,1,:)) .* std_r(1,:) + mnd(1,:); % return std and mean at the main (nl=1) level