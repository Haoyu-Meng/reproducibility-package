close all;
clear all;
clc;

load(['simulation_detect_result.mat']);

%% 
measure_loc = target_loc;
location_all = {};
history_location = {};
if_pause = {};
if_begin = {};

for i_prt = 1:size(measure_loc,2)

if i_prt == 1
    for i_t = 1:target_num(i_prt)
        location_all{length(location_all)+1}(1:3,1) = [1e5;1e5;1e5];
        location_all{length(location_all)}(1:2,2) = measure_loc{i_t,1};
        location_all{length(location_all)}(3,2) = i_prt;
        if_pause{length(location_all)} = 0;
        if_begin{length(location_all)} = 0;
    end
else
    assoc_prob = zeros(length(location_all), target_num(i_prt));
    for t = 1:length(location_all)
        if location_all{t}(1,1) == 1e6
            continue;
        end 
        for m = 1:target_num(i_prt)
            predicted_measurement = target_loc_pred{t};
            innovation(1) = measure_loc{m,i_prt}(1) - predicted_measurement(1);
            innovation(2) = measure_loc{m,i_prt}(2) - predicted_measurement(2);
    
            H = [1 0 0 0 0 0; 0 1 0 0 0 0];
            S = H * P_out{t}(:,:,end) * H' + [0.05 0; 0 0.05];
    
            likelihood = exp(-0.5 * innovation * inv(S) * innovation') / sqrt(det(2*pi*S)); 
            assoc_prob(t, m) = likelihood;
        end
    end
    
    assoc_prob = assoc_prob ./ (sum(assoc_prob, 2)+eps*ones(length(location_all),1));

    index_measure = 1:target_num(i_prt);
    for i_l = 1:length(location_all)
        if location_all{i_l}(1,1) == 1e6
            continue;
        end
        [temp_a,temp_b] = max(assoc_prob(i_l,:));
        if isempty(temp_a) || temp_a<0.9
            location_all{i_l}(1:2,end+1) = target_loc_pred{i_l}(1:2);
            location_all{i_l}(3,end) = i_prt;
            if_pause{i_l} = if_pause{i_l}+1;
        else
            location_all{i_l}(1:2,end+1) = measure_loc{temp_b,i_prt};
            location_all{i_l}(3,end) = i_prt;
            index_measure = setdiff(index_measure,intersect(temp_b,index_measure));
            if_pause{i_l} = 0;
            if_begin{i_l} = if_begin{i_l}+1;
        end
    end

    if ~isempty(index_measure)
        for i_t = 1:length(index_measure)
            location_all{length(location_all)+1}(1:3,1) = [1e5;1e5;1e5];
            location_all{length(location_all)}(1:2,2) = measure_loc{index_measure(i_t),i_prt};
            location_all{length(location_all)}(3,2) = i_prt;
            if_pause{length(location_all)} = 0;
            if_begin{length(location_all)} = 0;
        end
    end

end

for i_loc = 1:length(location_all)
    if location_all{i_loc}(1,1) == 1e6
        continue
    end
    if if_pause{i_loc} > 20
        location_all{i_loc}(1:3,1) = [1e6;1e6;1e6];
        continue;
    end
    if if_begin{i_loc} < 6 && if_pause{i_loc} > 0
        location_all{i_loc}(1:3,1) = [1e6;1e6;1e6];
        continue;
    end
    location{i_loc} = location_all{i_loc}(:,end);
end


[target_loc,target_loc_pred,history_location,P_out] = Tracking(location,location_all,history_location,1);
% Interacting Multiple Model (IMM) tracking method

figure(129);
imagesc(image_x,image_y,(abs(PCF(:,:,i_prt))));
axis normal; set(gca,'YDir','normal'); colormap("jet");
hold on;
for i_loc = 1:length(history_location) 
    if if_begin{i_loc} <= 5
        continue
    end
    plot(history_location{i_loc}(1,2:end),history_location{i_loc}(2,2:end),'w-','LineWidth',1);
end
hold off;

end

%% metrics calculated
gt_trajs = target1;
est_trajs = {};

for i = 1:length(trace)
    est_trajs{end+1} = trace(i).position;
end

metrics = compute_tracking_metrics(gt_trajs, est_trajs, ...
    'Alpha', 0.5, ...
    'AlphaRange', 0.1:0.1:1.0, ...
    'OSPA_c', 1.5, ...
    'OSPA_p', 2);

save(['simulation_detect_result_JPDA_IMM.mat'])

%% 
for i_loc = 1:length(history_location) 
    if size(history_location{i_loc},2)<10
        continue
    end
    plot(history_location{i_loc}(1,2:end),history_location{i_loc}(2,2:end),'LineWidth',1);
    hold on;
end
set(gca,'fontsize',16,'fontname','Times New Roman');
xlabel('Azimuth/m','fontsize',16,'fontname','Times New Roman');
ylabel('Range/m','fontsize',16,'fontname','Times New Roman');
xlim([-7 7]);ylim([3 20]);