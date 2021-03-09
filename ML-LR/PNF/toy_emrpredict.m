%%% DO EMR FIT AND PREDICTION WITH CROSS-VALIDATION  %%%%%%%%%%%

load pnas_data.mat; %% use PNAS paper data

%% pnf_toy_generate %% OR generate new data

data0 = data(1:end,2);%% use x_2 only

NT = size(data0,1);

T = 40;      %%period of extrenal forcing
ipls = 0;
inorm = 0;
niter = 200; %% ensemble forecast size
lead = 100;  %% maximum prediction lead time 

N1 = 19000; %% end of the model training interval as in [1 N]; 
N2 = 19000;  %% start of the cross-validation interval as in [N2 NE];
NE = NT-lead; %% end of the cross-validation interval  as in [N2 NE];


%%% 2-LEVEL QUBIC MODEL

nelin = 3;
nlevel = 1;

[fcst,true,rms,anc,modstr,xt_res,varr] = fcstemrplsext(data0,T,nelin,nlevel,niter,inorm,ipls,lead,N1,N2,NE);


%%% 1-LEVEL LINEAR MODEL (LIM) 

nelin = 1;
nlevel = 1;

[fcst1,true,rms1,anc1,modstr1,xt_res1,varr1] = fcstemrplsext(data0,T,nelin,nlevel,niter,inorm,ipls,lead,N1,N2,NE);


tt=1:lead;
tt=tt*0.1;

figure

plot(tt,anc,'r','LineWidth',2)
hold on
plot(tt,rms,'b','LineWidth',2)
plot(tt,anc1,'m','LineWidth',2)
plot(tt,rms1,'c','LineWidth',2)
xlabel('Lead time')
ylabel('skill')
grid on
legend('EMR-Corr','EMR-RMSE','LIM-Corr','LIM-RMSE','Location','NorthWest');



