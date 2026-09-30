if exist("baohu_area")
    clearvars -except baohu_area detect_area cm
else
    clearvars -except cm;
    load('detect_area*',"detect_area","baohu_area");
end

close all;
clc;
c = 3e8;

%% parameters
f0 = 1.9e9;
B = 0.512e9;  
lambda = c/(f0+B/2);
PRT = 1e-3;
K = B/PRT;
fs = 4e6;
t = 0:1/fs:PRT;
t = t(end-3940+1:end);

fft_number = 102400;
RL = linspace(0,c*fs*PRT/B,fft_number);
index_RL = find(RL<40);
RL = RL(index_RL);
SNR = -20;

% TranPosi=[ ];  The coordinates of the radar transmitters, arranged along the x-axis, y-axis, and z-axis.
% 
% RecePosi=[ ];  The coordinates of the radar receivers, arranged along the x-axis, y-axis, and z-axis.

an = zeros(2,3,100);  % 100 is number of transmitters × number of receivers
for i=1:100
    an(1,:,i) = TranPosi(ceil(i/10),:);    % 10 is number of receivers
    an(2,:,i) = RecePosi(i-10*(ceil(i/10)-1),:); % 10 is number of receivers
end

num_PRT = 300;
move_rand = 0.2;   % The left-right position offset caused by human body movement.
phase_rand = 160/180*pi; % The potential phase deviation that may occur among different channels.

[target1,flag111] = target_gen(PRT,length(TranPosi),num_PRT);
for i_tar = 1:length(target1)
    if flag111(i_tar)
        target1{i_tar} = target1{i_tar} + ...
                         (rand(2,num_PRT)-0.5)*move_rand*2; % The left-right position offset
    end
end

%% radar echo
echo = zeros(length(t),size(an,3),num_PRT);
PC = zeros(length(index_RL),size(an,3),num_PRT);

for i_prt = 1:num_PRT
for i_an = 1:size(an,3)
    for i_target = 1:length(target1)
        R_tx = sqrt( (an(1,1,i_an)-target1{i_target}(1,i_prt))^2 +...
                     (an(1,2,i_an)-target1{i_target}(2,i_prt))^2 +...
                     (an(1,3,i_an)-0)^2 );
        R_rx = sqrt( (an(2,1,i_an)-target1{i_target}(1,i_prt))^2 +...
                     (an(2,2,i_an)-target1{i_target}(2,i_prt))^2 +...
                     (an(2,3,i_an)-0)^2 );
        tau1 = (R_tx+R_rx)/c;
        echo(:,i_an,i_prt) = echo(:,i_an,i_prt)+exp(-2*1j*pi*f0*tau1 + 1j*pi*K*tau1^2 - 2*1j*pi*K*tau1*t)';
    end
    echo(:,i_an,i_prt) = echo(:,i_an,i_prt)+...
                         sqrt(10^(-SNR/10))*(randn(length(t),1)+1j*randn(length(t),1))/sqrt(2);
end
temp = fft(echo(:,:,i_prt).*hamming(3940),fft_number);
temp = temp(index_RL,:);
if rand<0.2
    temp = temp.*exp(1j*(rand(1,size(an,3))-0.5)*phase_rand*2); % The potential phase deviation
end

PC(:,:,i_prt) = temp(index_RL,:);
end

%% fast BP imaging
image_x = -10:0.2:10;  dx = image_x(2)-image_x(1);
image_y = 3:0.15:15;     dy = image_y(2)-image_y(1);
image_z = 0;   % Set to remain consistent with the detection_gen.m

range_bp = zeros(1,length(image_x)*length(image_y)*length(image_z)*size(an,3));
point_BP = zeros(1,length(image_x)*length(image_y)*length(image_z)*size(an,3));
grid_loc = zeros(3,length(image_x)*length(image_y)*length(image_z));

cn=1;
for i_x = 1:length(image_x)
    for i_y = 1:length(image_y)
        for i_z = 1:length(image_z)
            grid_loc(:,cn) = [image_x(i_x),image_y(i_y),image_z(i_z)];
            cn = cn+1;
        end
    end
end

for i_an = 1:size(an,3)
    R_tx = sqrt(sum((an(1,:,i_an)'-grid_loc).^2,1));
    R_rx = sqrt(sum((an(2,:,i_an)'-grid_loc).^2,1));
    inc_r = 0;
    temp = (i_an-1)*length(image_x)*length(image_y)+1:i_an*length(image_x)*length(image_y);
    range_bp(temp) = R_rx+R_tx+inc_r;
    index = round((R_tx+R_rx+inc_r)/(RL(2)-RL(1)))+1;
    point_BP(temp) = index+length(RL)*(i_an-1);
end
com_phase = exp(-1j*2*pi*f0*range_bp/c);


PCF = zeros(length(image_y),length(image_x),num_PRT);
detect = zeros(length(image_y),length(image_x),num_PRT);
target_num = zeros(1,num_PRT); target_loc = cell(30,num_PRT);

for i_prt = 1:num_PRT
tic
%%%%%%%%%%%%%%%%%%%%%% imgaing %%%%%%%%%%%%%%%%%%%%%%%
PC_temp = reshape(squeeze(PC(:,:,i_prt)),1,[]);
BP_image_tmp = (PC_temp(point_BP)).*com_phase;
BP_image_tmp = reshape(BP_image_tmp,length(image_y),length(image_x),size(an,3));
BP_image = sum((BP_image_tmp),3);
W_PCF = 1-std(exp(1j*angle(BP_image_tmp)),1,3);
PCF(:,:,i_prt) = BP_image.*W_PCF;

%%%%%%%%%%%%%%%%%%%%%%%%% detecion %%%%%%%%%%%%%%%%%%%%%%%%%%
detect_image = abs(squeeze(PCF(:,:,i_prt)));
detect_result = zeros(size(detect_image));
fa = 1e-10;
for i = 1:length(image_x)
    for j = 1:length(image_y)
        temp = detect_area{j,i};
        N_detect = sum(temp,'all');
        alpha = N_detect * (fa^(-1/N_detect) - 1);
        threshold = alpha * (sum(temp.*detect_image,'all') / N_detect);
        if detect_image(j,i) > threshold
            detect_result(j,i) = detect_image(j,i);
        end
    end
end
detect(:,:,i_prt) = detect_result;

[L,m] = bwlabel(detect_result, 8);
centroid = regionprops(L,'Centroid');

target_num(i_prt) = m;
for i = 1:target_num(i_prt)
    target_loc{i,i_prt}(1) = image_x(round(centroid(i,1).Centroid(1,1)));
    target_loc{i,i_prt}(2) = image_y(round(centroid(i,1).Centroid(1,2)));
end

pause(0.1)
toc
end


%% save result
filename = 'simulation_detect_result';
save(filename,"PCF","target_loc","target_num","image_x","image_y","target1");

%% functions
function [target1,flag_count] = target_gen(PRT,N_Tx,num_PRT)
% This function is used to generate the target motion trajectories.
% Inputs:
% PRT — Pulse repetition time
% N_Tx — Number of transmitting antennas (related to the transmit round-robin scheme)
% num_PRT — Number of slow-time samples
% Output:
% target1 — A tuple whose number of elements equals the number of targets.
% Each element is a 2×num_PRT array representing the two-dimensional position coordinates of a target at each slow-time instant.target_count = 0;
target1 = {};

while true

    prompt = {['if exit' newline '0:no' newline '1:yes']};
    dlgtitle = ['there are' num2str(target_count) 'targets'];
    dims = [1 50];
    answer = inputdlg(prompt,dlgtitle,dims);
    
    if str2double(answer{1}) == 100
        break;
    else
        prompt = {['maximum speed' newline 'Press e to return to the previous step.'],'minimum of x-axis','maximun of x-axis','minimum of y-axis','maximun of y-axis','delay'};
        dlgtitle = ['randomly walking'];
        dims = [1 50];
        definput = {'2','-7','7','4','14','0'};
        answer = inputdlg(prompt,dlgtitle,dims,definput);
    
        if answer{1} == 'e'
            continue;
        end

        v_max = str2double(answer{1});
        image_x = [str2double(answer{2}) str2double(answer{3})];
        image_y = [str2double(answer{4}) str2double(answer{5})];

        positions = move_round_human(num_PRT,PRT*N_Tx,v_max,image_x,image_y);
        % The function move_round_human generates random walking trajectories 
        % of a human target in a two-dimensional plane. The inputs include 
        % the sequence length num_frames, the time interval dt, the maximum speed v_max, 
        % and the motion boundary coordinate vectors image_x and image_y. 
        % The output is a 2×num_frames position array positions, where the first 
        % row contains the x-coordinates and the second row contains the y-coordinates. 
        % The function simulates human motion using a random walk model with 
        % turning inertia: at each frame, the moving direction is randomly changed within a small angle,
        % the speed magnitude is randomly scaled within 0.9~1.1 times and constrained by v_max, 
        % and a turning inertia coefficient is introduced to make the trajectory smoother. 
        % When the target reaches the boundary, a reflection mechanism is applied 
        % (the velocity reverses and loses its energy) to keep the target within the specified region.

        positions(:,str2double(answer{6})+1:end) = positions(:,1:end-str2double(answer{6}));
        positions(:,1:str2double(answer{6})) = 0;
        
        target1{length(target1)+1} = positions;
        target_count = target_count+1;
        flag_count(target_count) = 1;


    end

end

end