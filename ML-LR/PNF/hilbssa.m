function [phi]=hilbssa(data);
smoo_2=center(data);
len=size(smoo_2,1);
x=-hilbert(smoo_2,len);
tmp=atan(imag(x)./smoo_2);
phi=zeros(len,1);
ind=find(imag(x)>0 & smoo_2 >0);
phi(ind)=tmp(ind);
ind=find(imag(x)>0 & smoo_2 < 0);
phi(ind)=pi+tmp(ind);
ind=find(imag(x)<0 & smoo_2 < 0);
phi(ind)=pi+tmp(ind);
ind=find(imag(x)<0 & smoo_2 > 0);
phi(ind)=2*pi+tmp(ind);
end

