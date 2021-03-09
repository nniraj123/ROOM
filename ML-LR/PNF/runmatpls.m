function [beta,imin]=runmatpls(xblock1,yblock1)
[nrow ncol]=size(xblock1);
tmp = min(size(xblock1));
if tmp <= 40 
ncmp = tmp;
else 
ncmp = 40;
end
opt = statset('UseParallel',true);
[xl,yl,xs,ys,beta,pctvar,mse] = plsregress(xblock1,yblock1,ncmp,'CV',10,'MCReps',8,'opt',opt);
%%search for optimal number of components %%%%%%%%%%%%%
[err,imin] = min(mse(2,2:end));
%fprintf('%d \n',imin);
%%% estimated "imin" accounts for the intercept %%%%%%% 
[xl,yl,xs,ys,beta] = plsregress(xblock1,yblock1,imin);
%clf
%mse(2,:)
%semilogy(mse(2,:))
%pause
%fprintf('%d \n',i);
return
