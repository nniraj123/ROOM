clear all

%%Nt=200000; %% USE THIS Nt for the EMR FIT Demo         

Nt=20000; %% USE THIS Nt for the PNF and EMR PREDICTION Demo 

load pnas_data.mat;%%%dataset with white noise sequence used in PNAS paper
%% or uncomment below to generate "default" noise sequence%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%rng('default');
%%noise =randn(1,Nt);

xp=zeros(Nt,1);
yp=zeros(Nt,1);

%%%RUN THE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xp(1)=.5;
yp(1)=.5;

for l=1:Nt-1
[xp(l+1),yp(l+1)] = pnftoy(l,xp(l),yp(l),noise(l));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data=zeros(Nt,2);

data(:,1)=xp(1:Nt);
data(:,2)=yp(1:Nt);



