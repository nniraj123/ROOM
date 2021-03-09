%%TO RUN THE MODEL AND CREATE DATA OF PNAS ARTICLE UNCOMMENT BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%pnf_toy_generate
%%%%% OR LOAD DATA of PNAS ARTICLE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load pnas_data.mat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nt=20000;
N=Nt;		%% Total length of the simulated time series

modnoise=zeros(Nt,2);
modnoise(:,1)=noise';

%%COMPUTE SSA  %%%%%%%%%%%%%%

M=300;%%% SSA WINDOW %%%%%%%%

x=center(data(:,2));

%%setting up a trajectory matrix (time-delay embedding)

T=zeros(N-M+1,M);
for i=1:M
T(:,i)=x(i:N-M+i,1);
end

%%compute covariance matrix for SSA and find its eigenvectors

C=T'*T/(N-M+1);
[EV,EW]=eig(C);
EW=diag(EW);
[EW ind]=sort(EW,'descend');
EV=EV(:,ind);

%%%%%%%%finding T-PCs and perform SSA reconstruction

[A]=ssapc(x,EV);
[R]=ssarc(A,EV);

%%%reconstruction of the quasi-oscil. mode (leading pair of SSA components)

rc2=sum(R(:,1:2),2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lead=100;     %%maximum prediction time

NS=14000;     %% start of validation interval

NE=N-lead;    %% end of validation interval 

NT=NE-NS+1;   %% length of validation interval

NENS =100;    %% number of white noise realizations in standard (non-PNF) ensemble forecast 

%%%%%%%%%%%%% PNF-ONLY PARAMETERS %%%%%

eps=0.02;     %% initial tolerenace in similar phase (radians), further refined to get ~ "NPH" memebrs in PNF ensemble

NPH = 60;     %% maximum size of phase-based PNF ensemble

IIC = 0;      %% set to 1 to use i.c. to reduce phase-based PNF ensemble to ~"NLIM" memebrs 

NLIM = 30;    %% final size of PNF ensemble based on (phase + i.c's) (NLIM < = NPH) %%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%COMPUTE PNF PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['DO PNF PREDICTION']);
[fcst,true,rms,anc]=find_pnf(data,modnoise,[],NS,NE,center(rc2(1:N,1)),1,lead,eps,NPH,IIC,NLIM);

%%%%%%%%%%%COMPUTE ENSEMBLE PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['DO ENSEMBLE PREDICTION'])
[fcstens,true,rmsens,ancens]=find_pnf(data,modnoise,NENS,NS,NE,center(rc2(1:N,1)),0,lead,[],[],[],[]);

%%%%%%%%%PLOT PREDICTION SKILL %%%%%%%%
tt=1:100;
tt=tt*0.1;
figure
subplot(121)
plot(tt,anc(:,1),'r','LineWidth',2)
hold on
plot(tt,ancens(:,1),'r--','LineWidth',2)
plot(tt,anc(:,2),'b','LineWidth',2)
plot(tt,ancens(:,2),'b--','LineWidth',2)
legend('X1-PNF','X1-ENS','X2-PNF','X2-ENS');
grid on;
ylim([0.2 1.2]);
title('(a) Corr')
subplot(122)
plot(tt,rms(:,1),'r','LineWidth',2)
hold on
plot(tt,rmsens(:,1),'r--','LineWidth',2)
plot(tt,rms(:,2),'b','LineWidth',2)
plot(tt,rmsens(:,2),'b--','LineWidth',2)
legend('X1-PNF','X1-ENS','X2-PNF','X2-ENS','Location','SouthEast');
ylim([0.0 1.0]);
grid on;
title('(b) RMS')

L=80;%%%%%1<L<lead 
tt=1:NT;
tt=tt*0.1;
figure
subplot(211)
npc=1;
plot(tt,fcst(:,L,npc))
hold on
plot(tt,fcstens(:,L,npc),'r')
hold on
plot(tt,true(:,L,npc),'k')
ylim([0.35 0.85]);
legend('PNF','ENS','TRUE')
title('(a) PNF vs ENSEMBLE forecasting on X1')
subplot(212)
npc=2;
plot(tt,fcst(:,L,npc),'b')
hold on
plot(tt,fcstens(:,L,npc),'r')
hold on
plot(tt,true(:,L,npc),'k')
plot(tt,rc2(NS+L:NE+L)+mean(data(:,2)),'g','LineWidth',1);
ylim([0.15 0.33]);
legend('PNF','ENS','TRUE','SSA')
title('(b)  PNF vs ENSEMBLE forecasting on X2')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tt=1:20000;
tt=tt*0.1;
figure
plot(tt,x+mean(data(:,2)),'r')
hold on
plot(tt,rc2+mean(data(:,2)),'b')
xlim([tt(NS) tt(NE)]);
legend('data','SSA-RC');
xlabel('time')
title('LFV of X_2 in validation interval');

