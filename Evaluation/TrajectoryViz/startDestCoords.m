clear;
close all;

angle = 60;
R = 2000;

x1_0 = -R; y1_0 = 0; x1_dest = R; y1_dest = 0;
x2_0 = R * cosd(180 - angle); y2_0 = R * sind(180 - angle); x2_dest = 500; y2_dest = -500;
x3_0 = -707; y3_0 =-707 ; x3_dest = 707; y3_dest = 707;
% x4_0 = -707; y4_0 = 707; x4_dest = 707; y4_dest = -707;

plot([x1_0 x1_dest], [y1_0 y1_dest]);
hold on;
plot([x2_0 x2_dest], [y2_0 y2_dest]);
plot([x3_0 x3_dest], [y3_0 y3_dest]);
% plot([x4_0 x4_dest], [y4_0 y4_dest]);
axis equal;

X = [x1_0 y1_0 x1_dest y1_dest;
     x2_0 y2_0 x2_dest y2_dest]';
 
X = [x1_0 y1_0 x1_dest y1_dest;
     x2_0 y2_0 x2_dest y2_dest;
     x3_0 y3_0 x3_dest y3_dest;]';
 
% X = [x1_0 y1_0 x1_dest y1_dest;
%      x2_0 y2_0 x2_dest y2_dest;
%      x3_0 y3_0 x3_dest y3_dest;
%      x4_0 y4_0 x4_dest y4_dest;]';

% csvwrite("./coordsFiles/startDestCoords3.csv", X)