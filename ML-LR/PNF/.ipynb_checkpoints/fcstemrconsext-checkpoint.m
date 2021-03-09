function [fcst,true,rms,anc,modstr,xt_res,varr]= fcstemrconsext(data0,per,nelin,nlevel,niter,inorm,iskew,lead,lstartt,lstart,lend)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2017, All Rights Reserved
% Code by Dmitri Kondrashov
%
%
% Email: dkondras@atmos.ucla.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [fcst,true,rms,anc,modstr,xt_res,varr]= fcstemrconsext(data0,per,nelin,nlevel,niter,inorm,iskew,lead,lstartt,lstart,lend)
% CONSTRUCT MULTILEVEL-LEVEL NONLINEAR-CONSERVING EMR MODEL AND PERFORM PRECICTION WITH CROSS-VALDIATION
% INPUT : data0 - multivariate array [N,L], N - length of the time series, L - number of channels,
%                each channel is centered prior to EMR fitting; 
%         nelin - order of EMR model, supports only nelin=2 (quadratic)
%         nlevel - total number of levels in the EMR model including main level;
%         niter - total number of simulations (stochastic realizations) by EMR model; 
%         tlength - length of the simulated time series by EMR
%         per    -  sets optional external periodic forcing, array of periodicities. 
%
%		  inorm   - controls data normalization: 
%                   inorm = 0 - default, normalizes all channels by std(data(:,1)); 
%                   inorm = 1 - normalizes all channels by its own std.dev
%                   inorm = 2 - fit non-normalized data
%
%         iskew   - skew-symmetry constraints on linear part of main level
%                   iskew = 0 - default, skew-symmetry is not imposed 
%                   iskew = 1   imposes skew-symmetry constraints
%         lead    - maximum prediction lead
%         lstartt - end point of model training interval [1 lstartt]
%         [lstart lend] - sets valdiation interval where at every point forecasts are made;
%                         lend is assumed to be <=N-lead;
%--------------------------------------------------------------------------------
%  INTERNAL:        ires  - controls stochastic forcing at last level of EMR,
%						ires = 0 - (default) spatially correlated white noise
%						ires = 1 - residual noise (xt_res)
%						ires = 2 - no forcing
%                       ires = 3 - randomly sampled residual noise (xt_res)
%					nout  - number of leading channels to store in output of EMR-simulation
%					lim   - threshold abs value, if exceeded in EMR simulation, it is restarted
%					itermax - maximum number of EMR simulations to try 
%                   iext   - specifies interaction with external periodic forcing 
%                          iext = 0 - additive (a*sin(w.t) + b*cos(w.t))
%                          iext = 1 - default, adds multiplicative interactions (a*x*sin(w.t) + b*x*cos(w.t))
%--------------------------------------------------------------------------% OUTPUT:
%		  xt_res - array of regression residuals at the last level, size [N,L];
%         modstr - structure with model information, see comments at the end of the program
%         varr   - EMR diagnostics, array [nlevel,L];
%         if varr(nlevel,:)=0.5, then optimum number of levels is nlevel-1!
%         fcst - multivariate array of size [lend-lstart+1,lead,nout] and contains EMR ensemble mean forceast
%               in valdiation interval;
%         true - multivariate array of size [lend-lstart+1,lead,nout] with data for verification of EMR forecasts.
%               in valdiation interval;
%         rms  - array of size [lead,nout] and contains normalized RMSE computed from FCST and TRUE arrays
%         anc  - array of size [lead,nout] and contains anomaly correlation computed from FCST and TRUE arrays
%
%SETTING UP INTERNALS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	[tlength npc] = size(data0);
	data = data0(1:lstartt,:);
	[length npc] = size(data);
%% time step, do not change it from 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	ndt = 1;
	dt= ndt;
	outb = [];
	xx = [];
    stddata = std(data);
	meand=mean(data);	
	encons=[];
	ires = 0;
	nout = npc;
	itermax = 200;
	iext = 1;
%% time step, do not change it from 1!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	ndt = 1;
%PREPROCESSING DATA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	outb = [];
	xx = [];
    modstr=[];
    varr=[];
    xt_res=[];
    fcst =[];
    true=[];
    rms = [];
    anc = [];
    if nelin ~=2
    disp(['Sorry, Make the Model QUADRATIC!']);
    return;
    end
    if inorm ==1
    disp(['Sorry, use different normalization (inorm)!']);
    return;
    end

	xd = data;
	dmean=mean(xd);

for n=1:npc
xd(:,n)=xd(:,n)-dmean(n);
end
stddata = std(xd);

%%%%%set limit of acceptable solution 

if inorm ==2

lim=10*max(stddata);

else 

lim = 10;

end

%%%%%%%%%% x - MAIN ARRAY FOR MULTI-LEVEL EMR VARIABLES%%%%%%

    x = zeros(npc,length,nlevel);
	nmax=npc;
	for n=1:nmax
	for l=1:length
	x(n,l,1)=data(l,n);
	end
	end

%%%%%%%%%%%%%%%%%%SETUP of PERIODIC FORCING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if tlength > length
month = 1:tlength;
else 
month = 1:length;
end

[tmp lperiod] = size(per);
data_ext = [];

for i=1:lperiod
data_ext = [data_ext sin(2*pi/per(i)*month') cos(2*pi/per(i)*month')];
end

if iext == 0
next1 = lperiod; %%%
end 

if iext ==1 
next1 = (nmax+1)*lperiod;
end

next2 = 0;

%%%%%%%SETTING UP REGRESSION PARAMETERS%%%%%%%%%%%%%%%%%%%

nn=1;
for n=1:nmax	
nn=nn*(nelin+n)/n;
end
nnmax=nn;
%	anmax = max([nlevel*nmax+1 nnmax]);
anmax = max([nlevel*nmax+1+2*next2 nnmax+2*next1]);
xa = zeros(anmax,nlevel);
yy = zeros(nmax,nlevel);
annmax = zeros(nlevel,1);

%%%%%%%NUMBER OF EMR COEFFCIENTS AT EACH LEVEL%%%%%%%%%

annmax(1)=nnmax+2*next1;
for nl=2:nlevel
annmax(nl)=nl*nmax+1+2*next2;
end	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	aa = zeros(nmax,anmax,nlevel);%%EMR coefficents
	std1=zeros(nmax,1);
    dgns = zeros(nmax,1);
	flow = zeros(nmax,1);
	x_mean=zeros(nmax,nlevel);
	std_r = zeros(nlevel,npc);
%%%% GRAND LOOP OVER LEVELS, STARTING WITH MAIN nl=1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for nl=1:nlevel
%%%% PREPROCESSING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	x0 = x(:,:,nl);
	dmean = mean(x0,2);
%%%%%REMOVE THE MEAN PRIOR TO FIITING %%%%%%%%%%%%%%%%%%
	for n=1:npc
	x(n,:,nl)=x(n,:,nl)-dmean(n);
	end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stdd = std(x0(:,1:end-nlevel,:),0,2);

for n=1:npc

if nl ==1 
if inorm == 0 
x(n,:,nl)=x(n,:,nl)/stdd(1);
end
end

if nl ==1
if inorm == 1 
x(n,:,nl)=x(n,:,nl)/stdd(n);
end
end

if nl > 1 

if inorm == 1 
x(n,:,nl)=x(n,:,nl)/stdd(n);
end
end


end
%%%%%%%%%%%%%CONSTRUCT PREDICTAND (TENDENCIES BY EULER TIME-DIFFERENCING)%%%%%%%%%%%%%%%%%%%%
	    xt = zeros(length,nmax);
	    xt_res = zeros(length,nmax);
		for n=1:nmax
		for l=1:length-nl
		xt(l,n)=x(n,l+1,nl)-x(n,l,nl);
		end
%        xt(length,n)=0;
%		xt(length,n)=x(n,length,nl)-x(n,length-1,nl);
		end
%%%%%%%%%%%%%%CONSTRUCT DESIGN MATRIX OF PREDICTORS%%%%%%%%%%%%%%%%%%%%%%%%
	   nc=annmax(nl);%%number of predictors at current level
	   msvd=zeros(length,nc);%%empty design matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%GO OVER TIME LOOP TO CONSTRUCT DESIGN MATRIX%%%%%%%%%%%%%%%%%%%%%%%%
      for l=1:length
%%%%%%%%MAIN LEVEL (nl=1), CAN BE NONLINEAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
      if nl==1 

      count=1;
      xa(count,nl)=1;                                  
      for n=1:nmax 
      count=count+1;
          xa(count,nl)=x(n,l,nl);
      end                                     
	if nelin > 1 
      for n=1:nmax
      for n_1=1:n
        count=count+1;
          xa(count,nl)=x(n,l,nl)*x(n_1,l,nl);
      end
      end                                    
	end
	if nelin > 2
      for n=1:nmax
      for n_1=1:n
      for n_2=1:n_1
        count=count+1;
          xa(count,1)=x(n,l,1)*x(n_1,l,1)*x(n_2,l,1);
      end
      end                                    
      end                                    
	end
	if nelin > 3 
      for n=1:nmax
      for n_1=1:n
      for n_2=1:n_1
      for n_3=1:n_2
        count=count+1;
          xa(count,1)=x(n,l,1)*x(n_1,l,1)*x(n_2,l,1)*x(n_3,l,1);
      end
      end                                    
      end                                    
      end                                    
	end

%%%%%%%EXTERNAL FORCING ON MAIN LEVEL
if next1 ~= 0  

for i=1:2*lperiod
count = count+1;
xa(count,1)= data_ext(l,i);

if next1 > 1*lperiod  

for n=1:nmax
count = count+1;
xa(count,1) = data_ext(l,i)*x(n,l,1);
end
end

end

end
%%%%%%%END OF EXTERNAL FORCING ON MAIN LEVEL


	else
%%%%%%%%OTHER LEVELS (nl>1) ASSUMES LINEAR MODEL %%%%%%%%%%%%%
      count=1;
      xa(count,nl)=1;
     
      for nl_2=1:nl
      for n=1:nmax
       count=count+1;
       xa(count,nl)=x(n,l,nl_2);
        end         
        end      


	end  
%%%%%%STORE DESIGN MATRIX FOR GIVEN LEVEL%%%%%%%%%%%%%%%%%%%
	for n_1=1:count
	msvd(l,n_1)=xa(n_1,nl);
	end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	end   % END OF TIME LOOP TO CONSTRUCT DESIGN MATRIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if nl > 1
    msvd = msvd(1:end-nl,:);
    end
%%% CHECK FOR COLLINEARITY PROBLEMS %%%%%%%%%%%%%%%%%%%%%%%%
xc = corrcoef(msvd(:,2:nc));
disp(['LEVEL ' num2str(nl)]);
for n=1:nc-1
for i=n+1:nc-1
if abs(xc(n,i)) > 0.99
disp(['DESIGN MATRIX COLUMNS ' num2str(n+1) ' AND ' num2str(i+1) ' ARE HIGHLY COLLINEAR: ' num2str(xc(n,i))]);
end
end
end
%%%%%%SVD OF DESIGN MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bb=zeros(nc,npc); 
	meansvd = mean(msvd);
	nsvd = center(msvd);
%	nsvd = zeros(length,nc);
%	for n=1:nc
%	nsvd(:,n) = msvd(:,n) - meansvd(n);
%	end
	[U,S,V] = svd(nsvd(:,2:nc),0);
	p = diag(S);
	nth = nc-1;
	fprintf('Condition Number of Design Matrix at level: %d %f \n',nl,p(1)/p(nth));
	W = diag(1./diag(S));
	ss = V(:,1:nth)*W(1:nth,1:nth);



%%%%%IMPOSE ENERGY CONSERVATION FOR MAIN LEVEL%%%%%%%%%%%%%%%%%%%%%%%
%	if nl ==1 & pls ==1 && nelin > 1
%    [bb(1:nc,n),imin(n)]=runmatpls(msvd(:,2:nc),xt(:,n)); 
%    disp(['PLS for ' num2str(n) '; optimal number - ' num2str(imin(n))]);

if nl==1 

gsvd = zeros(length*npc,nc*npc);
xg = zeros(length*npc,1);
bg = zeros(nc*npc,1);

count=1+nmax;

for i=1:npc
gsvd(1+(i-1)*length:i*length,1+(i-1)*nc:i*nc)=msvd;%% design matrix is the same!!! 
xg(1+(i-1)*length:i*length)=xt(:,i);
%% need to create here equality matrix

for n_1=1:i
count=count+1;
if n_1==i
cons(i)=count;%%indices of the coefficient X_n^2 that should be zeroed;
end
end


end
%%%%%finding how many nonlinear triplet conditions && anytsymmetric conditions for linear part%%%%%%%%%%%%%%%
INDTR=[];
count=1;
i=1;
ntr =0;
nasym =0;
for j=1:npc
for k=1:j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if i~=j && j~=k && i~=k
ntr = ntr+1;
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if j~=k
nasym=nasym+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
end

for j=1:npc
for k=1:npc
posasym(j,k)=(j-1)*nc+1+k; %%offset for the intercept
end
end
%posasym
%nasym
%cons
%pause

%disp(['triplet conditions: ' num2str(ntr)]);

%fprintf('ntr= %d \n',ntr);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fr = zeros(npc,npc-1);
fm = zeros(npc,npc-1);
indm = zeros(npc,npc-1);

for i=1:npc
count = npc+1;
mtr = 0;
mfr=0;
mix=0;
for j=1:npc
for k=1:j
count = count+1;

%%%     finding indices of "foreign" terms for each i: x_j^2, i~=j;
if i ~= j && i~=k && k==j
mfr=mfr+1;
fr(i,mfr)=count+(i-1)*nc;
%% can also include the shift of count dependent on i: +(i-1)*nc
end
%%%     finding indices "mixed" terms for each i:X_j*X_i;
%%%     mix runs from 1:npc-1
if i == j && k~=j
mix=mix+1;
fm(i,mix)=count+(i-1)*nc;
indm(i,mix)=k;
%% can also include the shift to i: (i-1)*nc
end

if i == k && k~=j
mix=mix+1;
fm(i,mix)=count+(i-1)*nc;
indm(i,mix)=j;%% index of the mixed term
%% can also include the shift of count to i: (i-1)*nc
end


if i~=j && j~= k && i~=k;
mtr = mtr+1;
%fprintf('%d %d %d \n', i, j, k);

indtr(i,mtr,1)=j;
indtr(i,mtr,2)=k;
postr(i,mtr)=count+(i-1)*nc;%% to do shift

end


end
end
end
%postr

condtr=[];
numtr=0;
for i=1:npc
for it=1:ntr

%toteq = toteq+1;

pos = postr(i,it);

j=indtr(i,it,1);
k=indtr(i,it,2);

%j=indtr(1,it,1);
%k=indtr(1,it,2);

for it1=1:ntr
if indtr(j,it1,1)==i && indtr(j,it1,2)==k
posj = postr(j,it1);
elseif indtr(j,it1,2)==i && indtr(j,it1,1)==k
posj=postr(j,it1);
end 
end

for it1=1:ntr
if indtr(k,it1,1)==i && indtr(k,it1,2)==j
posk = postr(k,it1);
elseif indtr(k,it1,2)==i && indtr(k,it1,1)==j
posk=postr(k,it1);
end 
end
%%

%fprintf('%d %d %d\n',pos,posj,posk);
itmp=sort([pos posj posk]);
nexist=0;

for nn=1:numtr
if size(find(eq(condtr(nn,:),itmp)),2) >0 nexist=1;
end
end

if nexist ==0 
numtr=numtr+1;
condtr(numtr,:)=itmp;
%toteq=toteq+1;
%Aeq(toteq,itmp)=1;
end

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toteq=0;
%% with the zeroed intercept
%beq = zeros(2*npc+npc*(npc-1)+ntr,1);
%Aeq = zeros(2*npc+npc*(npc-1)+ntr,nc*npc);

%%%%%%without linasim%%%%
if iskew == 0
beq = zeros(npc+npc*(npc-1)+numtr,1);
Aeq = zeros(npc+npc*(npc-1)+numtr,nc*npc);
else 
%%%%%%with liniear asim%%%%
beq = zeros(npc+npc*(npc-1)+numtr+nasym,1);
Aeq = zeros(npc+npc*(npc-1)+numtr+nasym,nc*npc);
end

%%numtr,nasym

Aneq=zeros(npc,nc*npc);
bneq=zeros(npc,1);


%%%     to zero out x_J^2: 
for n=1:npc
toteq=toteq+1;
index = (n-1)*nc + cons(n);
Aeq(n,index)=1;
end

%%%     to zero out intercept: 
%for n=1:npc
%toteq=toteq+1;
%index = (n-1)*nc+1;
%Aeq(toteq,index)=1;
%end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tot=0;
for i=1:npc
k=0;
for j=1:npc %% exclude all x_J^2
if i~=j
toteq = toteq+1;
%tot = tot+1;
k=k+1;
%%    i is the current index 
ind = j;%%!!! the index that we need to find a mixed term with i in fm(ind, :), X_j*X_i!!!
mix = find(indm(ind,:)==i);		
pos = fr(i,k);%% position (can be also accounted for the shift to i)!!!
Aeq(toteq,fr(i,k))=1;
Aeq(toteq,fm(ind,mix))=1;
%fprintf('%d %d %d %d\n',i,ind,fr(i,k),fm(ind,mix));
end

end
end


for ii=1:numtr
toteq=toteq+1;
itmp=condtr(ii,:);
Aeq(toteq,itmp)=1;
end

if iskew >0
%%%%%%%%%%%%%%%%%setting skew-symmetery and positive diagonal for the linear part
for i=1:npc
for j=1:i
if i==j
Aneq(i,posasym(i,j))=1;
end
if i~=j
toteq=toteq+1;
Aeq(toteq,posasym(i,j))=1;
Aeq(toteq,posasym(j,i))=1;
end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
disp('Energy-conserving formulation for this level');
disp(['Total Number of EMR coeffcients on main level: ' num2str(npc*annmax(1))]);
disp(['Total Number of Equality Constraints: ' num2str(toteq)]);
if iskew >0 disp(['Impose Skew-Symmetry for Linear part']);end;
beq = zeros(size(Aeq,1),1);
b0 = ones(nc*npc,1);
options = optimset('Display','iter','LargeScale','off');

%[bg,resnorm,residual,exitflag,output,lambda]=lsqlin(gsvd,xg,[],[],Aeq,beq,[],[],b0,options);

[bg,resnorm,residual,exitflag,output,lambda]=lsqlin(gsvd,xg,Aneq,bneq,Aeq,beq,[],[],b0,options);


for n=1:npc

msvd = gsvd(1+(n-1)*length:n*length,1+(n-1)*nc:n*nc);
bb(:,n) = bg(1+(n-1)*nc:n*nc);

%size(bb)

for n_1=1:annmax(nl)
aa(n,n_1,nl)=bb(n_1,n);
end

regr = msvd*bb(:,n);    
sst = sum(xt(:,n).^2);     
xt_res(:,n)=xt(:,n)-regr;
sse = sum(xt_res(:,n).^2);
ssr = sum(regr.^2);  
varr(nl,n)=1-sse/sst;

end


disp(['convergence of linear constraints: ' num2str(max(abs(Aeq*bg)))]);


else 
%%%%%PERFORMING REGRESSION FOR EACH COMPONENT IN THE LOOP USING 
%%%%%SAME DESIGN MATRIX AT A GIVEN LEVEL%%%%%%%%%%%%%%%%%%%%%%%%%%

	mxt = mean(xt);
	for n=1:npc	
	xtn(:,n) = xt(:,n)-mxt(n);
	sk = U'*xtn(1:end-nl,n);
%%%%%COMPUTING AND STORE EMR COEFFICIENTS %%%%%%%%%%%%%%%%%%%%%%%%%  
	bb(2:nc,n) = ss(:,1:nth)*sk(1:nth);
	bb(1,n) = mxt(n) - bb(2:nc,n)'*meansvd(2:nc)';
    for n_1=1:annmax(nl)
    aa(n,n_1,nl)=bb(n_1,n);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%COMPUTING REGRESSION RESDIUALS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	regr = msvd*bb(:,n);    
	sst = sum(xt(1:end-nl,n).^2);     
	xt_res(1:end-nl,n)=xt(1:end-nl,n)-regr;
	sse = sum(xt_res(1:end-nl,n).^2);
	ssr = sum(regr.^2);  
	varr(nl,n)=1-sse/sst;
	end
    end
%%%%%%%END OF COMPONENT LOOP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%SET REGRESSION RESIDUALS AS MODEL VARIABLES FOR NEXT LEVEL%%%%%
std_r(nl,:)=std(xt_res(1:end-nl,:));
rr1(nl)=trace(chol(corrcoef(xt_res(1:end-nl,:))));
E(nl,:)=sort(eig(corrcoef(xt_res(1:end-nl,:))),'descend');


	if nl ~= nlevel 

        for l=1:length-nl
        for n=1:nmax
        x(n,l,nl+1)=xt_res(l,n);
        end
        end

	end

 	end			
%%%%%END OF GRAND LOOP OVER LEVELS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('EMR Diagnostics per level and component of the state vector:');
for nl=1:nlevel
tmp=['Component '];
for i=1:nmax
tmp=[tmp ' i=' num2str(i) ': ' num2str(varr(nl,i)) '; '];
end
disp(['Level ' num2str(nl) ':']);
disp(tmp);
end
disp(['Optimal model: at last level (L=nlevel) & all components converges to 0.5,']);
%%%%CHOLESKY DECOMPOSITION OF CORRELATION MATRIX OF RESIDUAL NOISE AT THE LAST LEVEL
    rr=chol(corrcoef(xt_res))';
disp(['CHOLESKY DECOMPOSITION OF CORRELATION MATRIX OF RESIDUAL NOISE:']);
	disp([num2str(rr)]);
range=0;

ldat = lend - lstart +1;
ltot = lend+lead;

true = zeros(ldat,lead,nout);
fcst = zeros(ldat,lead,nout);


month = 1:tlength;

data_ext = [];

for i=1:lperiod
data_ext = [data_ext sin(2*pi/per(i)*month') cos(2*pi/per(i)*month')];
end

nof = nlevel-1;

fsurr = NaN*zeros(ldat,lead,nout,niter);

xx = zeros(npc,tlength,nlevel);
xx(:,1:length,:)=x;
x = xx;
xx=x(:,1:length,:);


%%%%%%DO IF VALIDATION INTERVAL DOES NOT FULLY OVERLAP WITH THE TRAINING INTERVAL%%%%%%%%%%%%%%%%%%%%

if lend > lstartt



for l=lstartt+1:tlength

if inorm == 0
for n=1:nmax
x(n,l,1)=(data0(l,n)-meand(n))/stddata(1);
end
end 

if inorm == 1
for n=1:nmax
x(n,l,1)=(data0(l,n)-meand(n))/stddata(n);
end
end 


if inorm == 2
for n=1:nmax
x(n,l,1)=data0(l,n)-meand(n);
end
end 


end

[x,xt,xt_res]= model_proj(x,xt_res,aa,annmax,nlevel,xa,nelin,next1,next2,length,lstartt-1,lperiod,data_ext);

end

armean = zeros(lead,nmax);
for ik=1:lead
armean(ik,:)=meand;
end

disp(['PERFORM EMR PREDICTION AND CROSS-VALIDATION']);

for ll=lstart-nof:lend-nof 

lc=ll+nof;
lpt = lc-lstart+1;

xs = zeros(lead,nmax);%%									

for nl=1:nlevel
for n=1:nmax
xinit(n,nl)=x(n,ll,nl);
end
end

it=0;
iter = 0;
clear fpcs;

while it ~= niter
iter = iter+1;


for nl=1:nlevel
for n=1:nmax
yy(n,nl)=xinit(n,nl);
end
end   


count=1;
xa(count,1)=1;                            

for n=1:nmax 
count=count+1;
xa(count,1)=yy(n,1);
end                                     

if nelin >1 

for n=1:nmax
for n_1=1:n
count=count+1;
xa(count,1)=yy(n,1)*yy(n_1,1);
end
end   

end

if nelin > 2
for n=1:nmax
for n_1=1:n
for n_2=1:n_1
count=count+1;
xa(count,1)=yy(n,1)*yy(n_1,1)*yy(n_2,1);
end
end                                    
end                                    
end


if next1 ~= 0  
for i=1:2*lperiod
count = count+1;
xa(count,1)= data_ext(ll,i);

if next1 > 1*lperiod  
for n=1:npc 
count = count+1;
xa(count,1) = data_ext(ll,i)*yy(n,1); 
end
end

end % end of extending
end

for nl=2:nlevel

count=1;
xa(count,nl)=1;

for nl_2=1:nl
for n=1:nmax
count=count+1;
xa(count,nl)=yy(n,nl_2);
end         
end      

end    %%% end of nl=2:nlevel loop
k=0;


for l=1:lead+nof

for n=1:nmax
x_mean(n,nlevel)=sqrt(ndt)*std_r(nlevel,n)*randn;
end 

spn = rr*x_mean(:,nlevel);	

for n=1:nmax  

for nl=1:nlevel-1 
if inorm==1
x_mean(n,nl)=std_r(nl,n)*yy(n,nl+1);
else
x_mean(n,nl)=yy(n,nl+1);
end
end


x_mean(n,nlevel)=spn(n); %do it with internal random stochastic forcing

end


for nl=1:nlevel

for n=1:nmax
std1(n)=0;
for n_1=1:annmax(nl)
std1(n)=std1(n)+aa(n,n_1,nl)*xa(n_1,nl);
end
std1(n)=(std1(n)+x_mean(n,nl))*dt;
yy(n,nl)=yy(n,nl)+std1(n);
end
end

if l == 1
%fprintf('%f %f\n',yy(1,1),data0(ll+1,1));
end

if isnan(yy(1,1))==1 
fprintf('%d %d \n',it,iter);									
range=1;
break; 
end;

[ymax imax] = max(abs(yy(:,1)));

if ymax > lim
%fprintf('Outside range= %f %d %d \n',ymax,it,imax);
range=1;
break;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if l> nof*ndt
k=k+1;
for n=1:nmax   
xs(k,n) = yy(n,1);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count=1;
xa(count,1)=1;                            

for n=1:nmax 
count=count+1;
xa(count,1)=yy(n,1);
end                                     

if nelin > 1 
for n=1:nmax
for n_1=1:n
count=count+1;
xa(count,1)=yy(n,1)*yy(n_1,1);
end
end                                    
end

if nelin > 2
for n=1:nmax
for n_1=1:n
for n_2=1:n_1
count=count+1;
xa(count,1)=yy(n,1)*yy(n_1,1)*yy(n_2,1);
end
end                                    
end                                    
end

if next1 ~= 0

for i=1:2*lperiod  

count = count+1;
dext = data_ext(ll+l,i);
xa(count,1)= dext;

if next1 >1*lperiod
for n=1:nmax
count = count+1;
xa(count,1) = dext*yy(n,1);
end
end

end

end

%%%%%%%%%%%%%%start the orther levels
for nl=2:nlevel
count=1;
xa(count,nl)=1;

for nl_2=1:nl
for n=1:nmax
count=count+1;
xa(count,nl)=yy(n,nl_2);
end         
end      

end  % end of nl=2:nlevel loop

end  %% END OF LENGTH SIMULATION
%armean
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if range==0 
it=it+1;

if inorm == 0
fsurr(lpt,:,:,it)=xs(:,1:nout)*stddata(1)+armean(:,1:nout);
end 

if inorm == 1
datan = ones(lead,1)*stddata;
fsurr(lpt,:,:,it)=xs(:,1:nout).*datan+armean(:,1:nout);
end 

if inorm == 2
fsurr(lpt,:,:,it)=xs(:,1:nout)+armean(:,1:nout);
end 

else 

%%if iter >=itermax

%%break;

%end

end %%% end if range

range=0;

end  %% end of while
if(mod(lc-1,10)==0) 
disp(['Current time step: ' num2str(lc) ' ; End:  ' num2str(lend) ' ; ENSEMBLE SIZE: ' num2str(it) ' ; Simulations: ' num2str(iter)]);
%fprintf('%d %d %d\n',lc,it,iter);
end
fpcs = squeeze(fsurr(lpt,:,:,:));
true(lpt,:,:)=data0(lc+1:lc+lead,1:nout);
if nout ==1
fcst(lpt,:,:)=nanmean(fpcs,2);%%here 2 is only for scalar time series
else 
fcst(lpt,:,:)=nanmean(fpcs,3);%%here 3 is  for multivariate time series
end

end  %% end of prediction cycle

anc = zeros(lead,nout);
rms = zeros(lead,nout);
stdt= squeeze(std(true));

if nout > 1

for j=1:nout
for i=1:lead
rms(i,j)=sqrt(sum((squeeze(fcst(:,i,j))-squeeze(true(:,i,j))).^2)/ldat)/stdt(i,j);
anc(i,j)=xcorr(center(squeeze(fcst(:,i,j))),center(squeeze(true(:,i,j))),0,'coeff');
end
end

  else 

for j=1:nout
for i=1:lead
rms(i,j)=sqrt(sum((fcst(:,i)-true(:,i)).^2)/ldat)/stdt(i);
anc(i,j)=xcorr(center(fcst(:,i)),center(true(:,i)),0,'coeff');
end
end

  end

figure
plot(anc,'r')
hold on
plot(rms,'b')
grid on
legend('Corr','RMSE');

%%%%%%%STORE INFORMATION ABOUT EMR MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
modstr.nmax   = nmax;  %% number of channels
modstr.nelin  = nelin;%% as in input parameter
modstr.nlevel = nlevel;% as in input parameter
modstr.stdr   = std_r;%standard deviation of the regression residuals at each level
modstr.aa     = aa;% the model coefficents
modstr.rr     = rr;% Cholesky decomposition of the noise covariance matrix
modstr.out    = outb; 
modstr.varr   = varr;% regression diagnostics, size [L,nlevel]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
set(gca,'FontSize',12);
plot([1 nlevel],[0.5 0.5],'r','LineWidth',2);
hold on
legend('Optimal');
plot(mean(modstr.varr,2),'b.-','LineWidth',2,'MarkerFaceColor','b','MarkerSize',20);
title('Convergence')
xlabel('Number of levels');
set(gca,'FontSize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	return
