%%%% Need to run pnf_toy_generate.m with Nt=200000;%%%%%%%%%%%%%%
%% clear all;
%% 
pnf_toy_generate;
%%% FIT EMR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inorm = 2;
nlevel = 2;
nelin = 3;
ipls = 0;

NT=50; %% Number of EMR simulations

K=size(data,1); %% use all points for the fit; 

data0 = data(end-K+1:end,2);
N = size(data0,1);

%%% DO THE FIT WITHOUT PERIODIC FORCING
%%[xx,modstr,xt_res,varr]= fitemrplsext(N,[],data0,nelin,nlevel,NT,inorm,ipls);

%%% DO THE FIT WITH PERIODIC FORCING INCLUDED: GIVES BETTER RESULTS FOR ACFs
T = 40;
[xx,modstr,xt_res,varr]= fitemrplsext(N,T,data0,nelin,nlevel,NT,inorm,ipls);

%%% COMPUTE AND COMPARE ACFs FROM THE FIT
h = 0.1; %% time step
[cd,cs]=autocorremr(data0,xx,250,h);
title('ACF x_2');
xlabel('Time Lag');

