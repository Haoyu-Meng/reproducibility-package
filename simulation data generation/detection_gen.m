close all;
clear all;
clc;
c = 3e8;

%% parameters
f0 = 1.9e9;
B = 0.512e9;
lambda = c/(f0+B/2);
d_an = 0.71;  % array aperture

image_x = -10:0.2:10;
image_y = 3:0.15:15;   % Set to remain consistent with the simulation_detect.m

image_x11 = -15:0.2:15;
image_y11 = 3:0.15:15; % Set slightly larger than the imaging range to prevent the detection bounding box from exceeding the range.

dx = image_x(2)-image_x(1);
dy = image_y(2)-image_y(1);

%% calculate ellipse parameter
[I_X,I_Y] = meshgrid(image_x,image_y);
R_detect = sqrt(I_X.^2+I_Y.^2);

delta_ran = c/2/B;
delta_azi = lambda/d_an * R_detect;
theta_ell = atan(-I_X./I_Y);
num = length(theta_ell);
detect_area = cell(size(I_X));
baohu_area = cell(size(I_X));
xunlian_area = cell(size(I_X));

for i = 1:length(image_x)
    for j = 1:length(image_y)
        x_0 = I_X(j,i);
        y_0 = I_Y(j,i);
        theta_0 = theta_ell(j,i);
        major = delta_azi(j,i);
        minor = delta_ran;
    
        x_ = (I_X-x_0) .* cos(theta_0) + (I_Y-y_0) .* sin(theta_0);
        y_ = -(I_X-x_0) .* sin(theta_0) + (I_Y-y_0) .* cos(theta_0);
    
        % The range is set slightly larger than the calculated ellipse, which yields better detection performance.
        area_baohu = (x_.^2 / (1.5*major)^2 + y_.^2 / (1.5*minor)^2) <= 1;
        area_xunlian = (x_.^2 / (2*major)^2 + y_.^2 / (2*minor)^2) <= 1;
        detect_area{j,i} = area_xunlian-area_baohu;
    end
end

save(['detect_area_x_' num2str(image_x(1)) '_' num2str(image_x(end)) '_y_' ...
      num2str(image_y(1)) '_' num2str(image_y(end)) '.mat'],...
      "detect_area","image_y","image_x","d_an")