clear;
close all;

maxTimeSteps = 200;
dt = 1;
v = 50;
% 2 intruders
x1 = []; x1(1) = 0;
y1 = []; y1(1) = 300;
h1 = []; h1(1) = 0;
x2 = []; x2(1) = 0;
y2 = []; y2(1) = -500;
h2 = []; h2(1) = 0;
x3 = []; x3(1) = 0;
y3 = []; y3(1) = -1200;
h3 = []; h3(1) = 0;


%% Generate straight trajectories
syms x;
int1yref = diff(-1 / 2000 * x^2 + y1(1), x);
int2yref = diff(1 / 2000000 * x^3 - 1/5000 * x^2 - 1/1 * x + y2(1), x);
int3yref = diff(1 / 1300 * x^2 + y3(1), x);

for t = 2:dt:floor(maxTimeSteps/3)
    int1heading = atan(subs(int1yref, x1(t-1)));
    x1(t) = x1(t-1) + cos(int1heading) * v * dt;
    y1(t) = y1(t-1) + sin(int1heading) * v * dt;
    h1(t) = int1heading;
    int2heading = atan(subs(int2yref, x2(t-1)));
    x2(t) = x2(t-1) + cos(int2heading)  * v * dt;
    y2(t) = y2(t-1) + sin(int2heading)  * v * dt;
    h2(t) = int2heading;
    int3heading = atan(subs(int3yref, x3(t-1)));
    x3(t) = x3(t-1) + cos(int3heading)  * v * dt;
    y3(t) = y3(t-1) + sin(int3heading)  * v * dt;
    h3(t) = int3heading;
end

h1(1) = h1(2);
h2(1) = h2(2);
h3(1) = h3(2);

x1 = x1'; y1 = y1'; h1 = h1';
x2 = x2'; y2 = y2'; h2 = h2';
x3 = x3'; y3 = y3'; h3 = h3';

th = linspace(0,2*pi,100);
plot(1000 * cos(th), 1000 * sin(th));
hold on;
plot(x1, y1);
plot(x2, y2);
plot(x3, y3);
grid on;
axis equal;

traj = [x1 y1 h1 x2 y2 h2 x3 y3 h3];

%%
csvwrite("./coordsFiles/intTraj.csv", traj)