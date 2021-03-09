function [xn,yn] = pnftoy(l,xp,yp,noise)

%%%%% INPUT:  xp,yp at time step l, and driving noise
%%%%% OUTPUT: xn,yn at step l+1

h=0.1;
sigma=0.3;
r=1;
c=1.5;
d=1;
alpha=0.3;
a=0.05;
w=.25;
b=0.0;
f=0;

t=l*h;

xn=xp+(r+sigma*noise/sqrt(h))*xp*(alpha+xp)*(1-xp)*h-c*xp*yp*h + a*sin(2*pi*w*t)*h;
%+b*sin(2*pi*f*t)*h;

yn=yp-d*alpha*yp*h+(c-d)*yp*xp*h;

end




