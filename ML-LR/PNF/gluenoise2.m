function [slice]=gluenoise2(noise,lag,nof); 
[N,L]=size(noise);
slice=zeros(lag+nof,L,N-lag-nof+1);
for i=1:N-lag-nof+1
slice(:,:,i)=noise(i:i+lag+nof-1,:);
end
return