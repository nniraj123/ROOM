function [cd,c]=autocorremr(data,datas,lead,h);
figure
th=2;
[N0,L0]=size(data);
[N,K]=size(datas);
datas = reshape(datas,N,L0,K/L0);
j=0;
for i=1:L0
j=j+1;
%%%%%%%doing data
tmpd = data(:,i); 
cd = xcorr(center(tmpd),lead,'coeff');
c=zeros(K/L0,2*lead+1);
for k=1:K/L0
tmp = datas(:,i,k);
c(k,:) = xcorr(center(tmp),lead,'coeff');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(j>2)
j=1;
figure
end
tmpc = mean(c);
stdtmp = std(c);
tt = 1:lead+1;
tt = tt*h;
if L0 > 1 subplot(2,1,j); end;
plot(tt(1:lead+1),tmpc(lead+1:end),'LineWidth',th)
hold on
plot(tt(1:lead+1),cd(lead+1:end),'r','LineWidth',th)
plot(tt(1:lead+1),tmpc(lead+1:end)+ones(1,lead+1).*stdtmp(lead+1:end),'k','LineWidth',1)
legend('EMR','Data','std.dev')
plot(tt(1:lead+1),tmpc(lead+1:end)-ones(1,lead+1).*stdtmp(lead+1:end),'k','LineWidth',1)
ttitle = ['Comp-'  num2str(i)];
title(ttitle);
ylim([-0.5 1]);
xlim([tt(1) tt(lead+1)]);
end
