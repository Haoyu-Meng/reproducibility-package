close all;
clear all;
clc;
c = 3e8;
load(['simulation_detect_result.mat']);
flag_save = 1;

%% parameters
global f0 B lambda d_an xa ya mode
f0 = 1.9e9;
B = 0.512e9;
lambda = c/(f0+B/2);
d_an = 0.71;
xa = image_x;
ya = image_y;
mode = 1;   % 1: Each time, the image extracted according to the resolution is rotated and compressed.
            % 2: Only an image of the same size is extracted, and only rotation is performed.

%% tracking setting
num_PRT = size(target_loc,2);

% 初始化轨迹
ini_trace = struct('trace_id',0, ...  1. Identification：id
               'begin_prt',0, ...  1. Identification：begin
               'total_prt',0, ...  1. Identification：total
               'end_prt',0, ...  1. Identification：end
               'success_association_num',0, ...  2. Tracking quality：SA
               'consecutive_loss_num',0, ...  2. Tracking quality：CU
               'lifetime',0, ...  2. Tracking quality：lifetime
               'KCF_model',[], ...  3. Image：TA
               'KCF_alpha',[], ...  3. Image：FP
               'now_state','initial', ...  4. States：IS
               'relation_other',0, ... 4. States：IR
               'position',[]); % 4. States: position   

trace = ini_trace;

first_for = 2;
for i_prt = 1:num_PRT
first_for = first_for-1;   % used for making GIF

%% 1. detection result
if target_num(i_prt) == 0
    measure = [];
else
    measure = target_loc(1:target_num(i_prt),i_prt);
end
image = squeeze(PCF(:,:,i_prt)); image = abs(mat2gray(abs(image))-1)*255;

if flag_save ~= 1
figure(10000);
imagesc(image_x,image_y,abs(squeeze(PCF(:,:,i_prt))));
title(['prt = ' num2str(i_prt)]);
colormap(jet); axis normal; 
set(gca,'YDir','normal','fontsize',16,'fontname','Times New Roman');
hold on;
for i = 1:target_num(i_prt)
    plot(measure{i}(1),measure{i}(2),'w*');
    text(measure{i}(1),measure{i}(2),['measure:' num2str(i)],...
         'FontSize',8,'FontName','Times New Roman',...
         'HorizontalAlignment','left','VerticalAlignment','top',...
         'Color','w');
end
hold off;
xlabel('azimuth/m','fontsize',16,'fontname','Times New Roman'); 
ylabel('range/m','fontsize',16,'fontname','Times New Roman');
end

%% 2. data association
association_martix = data_association(measure,trace,image);
if trace(end).trace_id == 0
    index_trace = [];
else 
    index_trace = 1:length(trace);
end
if isempty(measure)
    index_measure = [];
else
    index_measure = 1:length(measure);
end

%% 3. intersection trajectory
for i_t = 1:length(trace)
    if ~strcmpi(trace(i_t).now_state,'IS')
        continue;
    end
    if ~ismember(i_t,index_trace)
        continue;
    end

    % find trajectory
    trace1 = trace(i_t);
    trace2 = trace(trace1.relation_other);

    % find measurement
    if ~isempty(measure)
        [~,target11] = find(association_martix(i_t,:));
        [~,target2] = find(association_martix(trace1.relation_other,:));
        target = union(target11,target2);
    else
        target = [];
    end

    index_trace = setdiff(index_trace,[i_t,trace1.relation_other]);
    index_measure = setdiff(index_measure,target);

    if length(target) <= 1
        % 3.1 The number of targets matched by the two trajectories is ≤ 1, indicating that the two trajectories are still intersection
        trace1 = KCF_track(trace1,image,measure(target));
        trace2 = KCF_track(trace2,image,measure(target));

        if trace1.total_prt >= trace2.total_prt
            trace1.total_prt = trace1.total_prt+1;
            trace1.success_association_num = trace1.success_association_num+1;
            trace1.consecutive_loss_num = 0;
            trace1.end_prt = i_prt;

            trace2.total_prt = trace2.total_prt+1;
            trace2.consecutive_loss_num = trace2.consecutive_loss_num+1;
            trace2.end_prt = i_prt;
            if trace2.consecutive_loss_num > 50
                trace2.now_state = 'END';
                trace2.position = trace2.position(:,1:end-50);
                trace1.now_state = 'RUN';
            end
        else
            trace2.total_prt = trace2.total_prt+1;
            trace2.success_association_num = trace2.success_association_num+1;
            trace2.consecutive_loss_num = 0;
            trace2.end_prt = i_prt;

            trace1.total_prt = trace1.total_prt+1;
            trace1.consecutive_loss_num = trace1.consecutive_loss_num+1;
            trace1.end_prt = i_prt;
            if trace1.consecutive_loss_num > 50
                trace1.now_state = 'END';
                trace1.position = trace1.position(:,1:end-50);
                trace2.now_state = 'RUN';
            end
        end

    else   
        % seperation
        [prediction_trace1,fit1] = bezier_prediction(trace1.position);
        [prediction_trace2,fit2] = bezier_prediction(trace2.position);

        juli = zeros(2,length(target));
        for i_tt = 1:length(target)
            juli(1,i_tt) = sqrt( (prediction_trace1(1)-measure{target(i_tt)}(1))^2 +...
                                 (prediction_trace1(2)-measure{target(i_tt)}(2))^2);
            juli(2,i_tt) = sqrt( (prediction_trace2(1)-measure{target(i_tt)}(1))^2 +...
                                 (prediction_trace2(2)-measure{target(i_tt)}(2))^2);
        end

        [trace1_match,trace2_match] = data_reassociation(juli);
        % The input is the distance association cost juli. The output is the 
        % measurement index matched to trajectory 1 and the measurement index 
        % matched to trajectory 2. The method formulates a minimum-cost maximum-flow
        % problem and solves it using the modified Dijkstra method.
        trace1 = KCF_track(trace1,image,measure(target(trace1_match)));
        trace2 = KCF_track(trace2,image,measure(target(trace2_match)));

        % update trajectory 1
        trace1.total_prt = trace1.total_prt+1;
        trace1.success_association_num = trace1.success_association_num+1;
        trace1.consecutive_loss_num = 0;
        trace1.end_prt = i_prt;
        if trace1.total_prt > 30
            temp = 30;
        else
            temp = trace(i_t).total_prt;
        end
        trace1.lifetime = (temp-trace1.consecutive_loss_num)/30;
        trace1.now_state = 'RUN';
        trace1.relation_other = 0;

        % update trajectory 2
        trace2.total_prt = trace2.total_prt+1;
        trace2.success_association_num = trace2.success_association_num+1;
        trace2.consecutive_loss_num = 0;
        trace2.end_prt = i_prt;
        if trace2.total_prt > 30
            temp = 30;
        else
            temp = trace(i_t).total_prt;
        end
        trace2.lifetime = (temp-trace2.consecutive_loss_num)/30;
        trace2.now_state = 'RUN';
        trace2.relation_other = 0;
    end

trace(trace(i_t).relation_other) = trace2;
trace(i_t) = trace1;

end

%% 4. normal trajectory
[t1,t2] = find(association_martix & (sum(association_martix,2) == 1) & (sum(association_martix,1) == 1));
for i_up = 1:length(t1)
    i_t = t1(i_up);  
    i_m = t2(i_up);  
    if ~ismember(i_t,index_trace) || ~ismember(i_m,index_measure)
        continue;
    end
    index_trace = setdiff(index_trace,i_t);
    index_measure = setdiff(index_measure,i_m);

    % updata
    trace(i_t).success_association_num = trace(i_t).success_association_num+1;
    trace(i_t).consecutive_loss_num = 0;
    trace(i_t).total_prt = trace(i_t).total_prt+1;
    trace(i_t).end_prt = i_prt;
    if trace(i_t).total_prt > 30
        temp = 30;
    else
        temp = trace(i_t).total_prt;
    end
    trace(i_t).lifetime = (temp-trace(i_t).consecutive_loss_num)/30;
    if trace(i_t).lifetime > 0.2
        trace(i_t).now_state = 'RUN';
    end

    trace(i_t) = KCF_track(trace(i_t),image,measure(i_m));

end

%% 5. intersection generate
inter_target = find( sum(association_martix,1) > 1 );
for i_im = 1:length(inter_target)
if ~ismember(inter_target(i_im),index_measure)
        continue;
end
index_measure = setdiff(index_measure,inter_target(i_im));
inter_trace = find(association_martix(:,inter_target(i_im)));

UB_trace = [];
for i_it = 1:length(inter_trace)
    if ~ismember(inter_trace(i_it),index_trace)
        inter_trace = setdiff(inter_trace,inter_trace(i_it));
        continue;
    end
    index_trace = setdiff(index_trace,inter_trace(i_it));
    if strcmpi(trace(inter_trace(i_it)).now_state,'UB')
        UB_trace = [UB_trace inter_trace(i_it)];
    end
end
if length(UB_trace) ~= length(inter_trace) && ~isempty(UB_trace)
    trace(UB_trace).now_state = 'END';
    inter_trace = setdiff(inter_trace,UB_trace);
elseif length(UB_trace) == length(inter_trace)
    index_temp = 1; lifetime = trace(UB_trace(1)).lifetime;
    for i_ub = 2:length(UB_trace)
        if lifetime < trace(UB_trace(i_ub)).lifetime
            trace(UB_trace(index_temp)).now_state = 'END';
            index_temp = i_ub; lifetime = trace(UB_trace(i_ub)).lifetime;
        else
            trace(UB_trace(i_ub)).now_state = 'END';
        end
    end
    inter_trace = UB_trace(index_temp);
end

if length(inter_trace) == 1
    trace(inter_trace) = KCF_track(trace(inter_trace),image,measure(inter_target(i_im)));
    trace(inter_trace).success_association_num = trace(inter_trace).success_association_num+1;
    trace(inter_trace).consecutive_loss_num = 0;
    trace(inter_trace).total_prt = trace(inter_trace).total_prt+1;
    trace(inter_trace).end_prt = i_prt;
    if trace(inter_trace).total_prt > 30
        temp = 30;
    else
        temp = trace(inter_trace).total_prt;
    end
    trace(inter_trace).lifetime = (temp-trace(inter_trace).consecutive_loss_num)/30;
elseif length(inter_trace) > 1
    for i_it = 1:length(inter_trace)
        index_inter_trace = inter_trace(i_it);
        trace(index_inter_trace) = KCF_track(trace(index_inter_trace),image,measure(inter_target(i_im)));
        trace(index_inter_trace).success_association_num = trace(index_inter_trace).success_association_num+1;
        trace(index_inter_trace).consecutive_loss_num = 0;
        trace(index_inter_trace).total_prt = trace(index_inter_trace).total_prt+1;
        trace(index_inter_trace).end_prt = i_prt;
        if trace(index_inter_trace).total_prt > 30
            temp = 30;
        else
            temp = trace(index_inter_trace).total_prt;
        end
        trace(index_inter_trace).lifetime = (temp-trace(index_inter_trace).consecutive_loss_num)/30;
        trace(index_inter_trace).now_state = 'IS';
        trace(index_inter_trace).relation_other = setdiff(inter_trace,index_inter_trace);
    end
end

end

%% 6. unassociated trajectory update
for i_dt = 1:length(index_trace)
    if strcmpi(trace(index_trace(i_dt)).now_state,'UB')
        trace(index_trace(i_dt)).now_state = 'END';
        continue;
    end
    if strcmpi(trace(index_trace(i_dt)).now_state,'END')
        continue;
    end
    
    trace(index_trace(i_dt)) = KCF_track(trace(index_trace(i_dt)),image,[]);

    trace(index_trace(i_dt)).consecutive_loss_num = ...
                trace(index_trace(i_dt)).consecutive_loss_num+1;
    trace(index_trace(i_dt)).total_prt = trace(index_trace(i_dt)).total_prt+1;
    trace(index_trace(i_dt)).end_prt = i_prt;
    if trace(index_trace(i_dt)).total_prt > 30
        temp = 30;
    else
        temp = trace(index_trace(i_dt)).total_prt;
    end
    trace(index_trace(i_dt)).lifetime = (temp-trace(index_trace(i_dt)).consecutive_loss_num)/30;

    if trace(index_trace(i_dt)).consecutive_loss_num > 20
        trace(index_trace(i_dt)).now_state = 'END';
        trace(index_trace(i_dt)).position = trace(index_trace(i_dt)).position(:,1:end-5);
    end
end

%% 7. new trajectory generated
for i_dm = 1:length(index_measure) 
    miss_measure = measure{index_measure(i_dm)};
    association_martix(:,index_measure(i_dm)) = 0;

    if trace(end).trace_id ~= 0
        trace(length(trace)+1) = ini_trace;
    end
    trace(end).trace_id = length(trace);
    trace(end).begin_prt = i_prt;
    trace(end).total_prt = 1;
    trace(end).end_prt = i_prt;

    trace(end).success_association_num = 1;
    trace(end).lifetime = 1/30;

    trace(end).now_state = 'UB';
    trace(end).relation_other = 0;
    trace(end).position = miss_measure';

    x = miss_measure(1); y = miss_measure(2);
    R_detect = sqrt(x^2+y^2);
    delta_ran = c/2/B;
    delta_azi = lambda/d_an * R_detect;
    rotate_angle = atand(-x/y);

    theta = linspace(0,2*pi,10000);
    x_ell = x + delta_azi * cos(theta) * cosd(rotate_angle) - ...
                delta_ran * sin(theta) * sind(rotate_angle);
    y_ell = y + delta_azi * cos(theta) * sind(rotate_angle) + ...
                delta_ran * sin(theta) * cosd(rotate_angle);
     
    xmin = floor((min(x_ell)-xa(1)) / (xa(2)-xa(1)))+1; xmin = max(1,xmin);
    ymin = floor((min(y_ell)-ya(1)) / (ya(2)-ya(1)))+1; ymin = max(1,ymin);
    xmax = ceil((max(x_ell)-xa(1)) / (xa(2)-xa(1)))+1; xmax = min(length(xa),xmax);
    ymax = ceil((max(y_ell)-ya(1)) / (ya(2)-ya(1)))+1; ymax = min(length(ya),ymax);
    
    area = image(ymin:ymax,xmin:xmax);

    model_x = xmax-xmin+1; model_y = ymax-ymin+1;
    output_sigma = sqrt(model_y*model_x) * 1/10;
    [rs, cs] = ndgrid((1:model_y) - floor(model_y/2), ...
                      (1:model_x) - floor(model_x/2));
    expect = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
    expect_fft = fft2(expect);
    cos_window = hann(model_y) * hann(model_x)';

    rotate_area = imrotate(area,rotate_angle,"bilinear","crop");
    rotate_area = double(rotate_area) / 255 - 0.5;
    rotate_area = cos_window .* rotate_area;

    area_fft = fft2(rotate_area);
    area_corr = rotate_area(:)'*rotate_area(:);
    area_area_fft = area_fft.*area_fft;
    area_area = real(circshift(ifft2(area_area_fft), floor(size(rotate_area)/2)));
    response = exp(-1 / 0.2^2 * max(0, (area_corr + area_corr - 2 * area_area) / numel(rotate_area)));

	trace(end).KCF_alpha = expect_fft ./ (fft2(response) + 1.4e-2);
    trace(end).KCF_model = rotate_area;
end

%% 8. figure
figure(20251117); 
set(gcf,'Name','track_result','Color','w');
imagesc(image_x,image_y,(abs(PCF(:,:,i_prt))));title(['prt = ' num2str(i_prt)]);
colormap(jet); axis normal; 
set(gca,'YDir','normal','fontsize',16,'fontname','Times New Roman');
xlabel('azimuth/m','fontsize',16,'fontname','Times New Roman'); 
ylabel('range/m','fontsize',16,'fontname','Times New Roman');

hold on;
for i_t = 1:length(trace)
    condition1 = strcmpi(trace(i_t).now_state,'RUN') && trace(i_t).lifetime > 0.66;
    condition2 = strcmpi(trace(i_t).now_state,'IS');
    if flag_save ~= 1
        condition = ~strcmpi(trace(i_t).now_state,'END');
        if condition
        text(trace(i_t).position(1,end),trace(i_t).position(2,end),...
             ['id:' num2str(trace(i_t).trace_id)],'FontSize',8,'FontName','Times New Roman',...
             'HorizontalAlignment','left','VerticalAlignment','top','Color','w');
        end
    else
        condition = condition1 || condition2;
    end
    if condition
        plot(trace(i_t).position(1,:),trace(i_t).position(2,:),...
             'LineWidth',1);
        plot(trace(i_t).position(1,end),trace(i_t).position(2,end),...
             'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red');
    end
end
hold off;

end

%% metrics calculated
gt_trajs = target1;
est_trajs = {};

for i = 1:length(trace)
    if trace(i).success_association_num > 15
        est_trajs{end+1} = trace(i).position;
    end
end

metrics = compute_tracking_metrics(gt_trajs, est_trajs, ...
    'Alpha', 0.5, ...
    'AlphaRange', 0.1:0.1:1.0, ...
    'OSPA_c', 1.5, ...
    'OSPA_p', 2);

if flag_save == 1
    save(['simulation_detect_result_track_result.mat'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   functions   %%%%%%%%%%%%%%%%%
% function association_martix = data_association(measure,trace,image)

% The function data_association implements measurement-to-trajectory data 
% association for through-the-wall radar multi-target tracking. 
% The inputs include the measurement data measure (a cell array, 
% where each element is a single measurement), the existing trajectory 
% (a struct array), and the current-frame radar image image. 
% The output is the association matrix association_martix 
% (trajectory count × measurement count, with associated entries set to 1).
% The function first computes a multiple dimension association cost between 
% each trajectory and each measurement, consisting of three components: 
% the distance cost and the KCF response difference (computed by applying the trajectory's stored KCF model to the measurement position). 
% Two components are combined through weight to form the final cost. 
% A minimum-cost maximum-flow network is then constructed using digraph,
% and the MCMF is solved using modified Dijkstra method.
% The final association matrix is generated and returned.

% end

%%
function response = KCF_response(point,image,model)

% Input
% point — The input coordinate point, around which a region is extracted to compute the response
% image — The raw radar image
% model — The KCF response
% Output
% respone — The response within the image region

global mode

[this_area,rotate_angle] = get_point_area(point,image,model,mode);

cos_window = hann(size(model,1)) * hann(size(model,2))';
rotate_area = imrotate(this_area,rotate_angle,"bilinear","crop");
rotate_area = double(rotate_area) / 255 - 0.5;
rotate_area = cos_window .* rotate_area;

model_fft = fft2(model);
model_corr = model(:)'*model(:);
area_fft = fft2(rotate_area);
area_corr = rotate_area(:)'*rotate_area(:);
model_area_fft = model_fft.*conj(area_fft);
model_area_corr = real(circshift(ifft2(model_area_fft), floor(size(model)/2)));

response = exp(-1/0.2^2 * max(0, (model_corr + area_corr - 2 * model_area_corr) / numel(model)));

end

%% 
function [area,rotate_angle] = get_point_area(point,image,model,mode)

% Input
% point — A 1×2 coordinate point
% image — The radar image of the current frame
% model — The KCF model, used to obtain the two-dimensional size
% mode — Same definition as the global variable
% Output
% area — The corresponding image region, with dimensions consistent with the model
% rotate_angle — The corresponding rotation angle

global B lambda d_an xa ya

x = point(1); y = point(2);
x_model = size(model,2); y_model = size(model,1);

R_detect = sqrt(x^2+y^2);
% delta_ran: range resolution
% delta_azi: azimuth resolution
% rotate_angle: rotate angle corresponse to target angle

if mode == 1

    % 得到对应的椭圆的参数
    theta = linspace(0,2*pi,10000);
    x_ell = x + delta_azi * cos(theta) * cosd(rotate_angle) - ...
                delta_ran * sin(theta) * sind(rotate_angle);
    y_ell = y + delta_azi * cos(theta) * sind(rotate_angle) + ...
                delta_ran * sin(theta) * cosd(rotate_angle);
    
    % 找到椭圆的外接矩形并提取出来
    xmin = floor((min(x_ell)-xa(1)) / (xa(2)-xa(1)))+1; xmin = max(1,xmin);
    ymin = floor((min(y_ell)-ya(1)) / (ya(2)-ya(1)))+1; ymin = max(1,ymin);
    xmax = ceil((max(x_ell)-xa(1)) / (xa(2)-xa(1)))+1; xmax = min(length(xa),xmax);
    ymax = ceil((max(y_ell)-ya(1)) / (ya(2)-ya(1)))+1; ymax = min(length(ya),ymax);
        
    area = image(ymin:ymax,xmin:xmax);
    area = imresize(area,size(model),"bilinear");

else

    x_point = floor((x-xa(1)) / (xa(2)-xa(1))) + (1:x_model) - floor(x_model/2);
    y_point = floor((y-ya(1)) / (ya(2)-ya(1))) + (1:y_model) - floor(y_model/2);
    x_point(x_point<1) = 1; x_point(x_point>size(xa)) = size(xa);
    y_point(y_point<1) = 1; y_point(y_point>size(ya)) = size(ya);
    area = image(y_point,x_point);

end


end

%% 
% function trace_new = KCF_track(trace_old,image,measure)
% 
% This function updates a trajectory's position and KCF model parameters at the current frame. 
% The inputs include the old trajectory trace_old, the current-frame radar image image, 
% and the set of measurements measure associated with this trajectory (a cell array, possibly empty). 
% The output is the updated new trajectory trace_new. 
% The function first uses the trajectory's last-frame position as the center to extract a KCF response region
% from the current radar image, then computes the response map using the KCF filter, 
% and takes the position of the maximum response as the KCF localization result. 
% An image region is then extracted at the KCF-localized position, and after rotation correction, 
% normalization, and cosine window application, the KCF model and filter coefficients are updated. 
% When no measurement is available, the tracking result depends on the previous-frame tracking result and the KCF result. 
% When measurements are available, the tracking result depends on the mean of the measurements and the KCF result.
% Finally, the position is clamped within the imaging range and appended to the trajectory's position sequence.
% 
% end







