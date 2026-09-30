close all; 
clear all; 
clc;

load(['simulation_detect_result.mat']);
measure_loc = target_loc;

%% ---------------- parameters ----------------
params.Ps = 0.99;
params.Pd = 0.95;
params.clutter_intensity = 1e-4;
params.w_birth = 0.6;
params.mergeThreshold = 4;
params.pruneThreshold = 1e-4;
params.maxComponents = 500;
params.extractWeightTh = 0.3;
params.dt = 8e-3;

F = [1 0 params.dt 0;
     0 1 0 params.dt;
     0 0 1 0;
     0 0 0 1];
Q = diag([0.01,0.01,0.05,0.05]);

H = [1 0 0 0; 0 1 0 0];
R = diag([0.05, 0.05]);

track_params.new_track_confirm_frames = 6;
track_params.max_missed_frames = 20;
track_params.init_P = diag([1,1,1,1]);
cm = 1;

extract_spatial_gate = 1.0;

display_on = true;

%% ---------------- init ----------------
gm_components = {};
nextTrackID = 1;

tracks = [];

nFrames = size(measure_loc,2);

history_location = cell(0);

for k = 1:nFrames

    Z = [];
    if exist('target_num','var') && ~isempty(target_num)
        Nm = target_num(k);
    else
        Nm = 0;
    end
    for m = 1:Nm
        if isempty(measure_loc{m,k}), continue; end
        z = measure_loc{m,k}(:)';
        Z = [Z; z];
    end

    for i = 1:length(gm_components)
        comp = gm_components{i};
        xpred = F * comp.x;
        Ppred = F * comp.P * F' + Q;
        wpred = params.Ps * comp.w;
        gm_components{i}.x = xpred;
        gm_components{i}.P = Ppred;
        gm_components{i}.w = wpred;
    end

    for zi = 1:size(Z,1)
        z = Z(zi,:);
        newComp.w = params.w_birth;
        newComp.x = [z(1); z(2); 0; 0];
        newComp.P = diag([1.0, 1.0, 2.0, 2.0]);
        gm_components{end+1} = newComp;
    end

    updated_components = {};
    for i = 1:length(gm_components)
        comp = gm_components{i};
        xpred = comp.x; Ppred = comp.P; wpred = comp.w;
        missComp.w = (1 - params.Pd) * wpred;
        missComp.x = xpred;
        missComp.P = Ppred;
        updated_components{end+1} = missComp;
        for zi = 1:size(Z,1)
            z = Z(zi,:)';
            S = H * Ppred * H' + R;
            Kk = Ppred * H' / S;
            xin = xpred + Kk * (z - H * xpred);
            Pin = (eye(size(Ppred)) - Kk * H) * Ppred;
            innov = z - H * xpred;
            likelihood = exp(-0.5 * (innov' / S) * innov) / sqrt((2*pi)^2 * det(S) + eps);
            w_upd = params.Pd * wpred * likelihood;
            upComp.w = w_upd;
            upComp.x = xin;
            upComp.P = (Pin + Pin')/2;
            updated_components{end+1} = upComp;
        end
    end

    gm_components = updated_components;

    keepIdx = [];
    for i = 1:length(gm_components)
        if gm_components{i}.w > params.pruneThreshold
            keepIdx(end+1) = i;
        end
    end
    gm_components = gm_components(keepIdx);

    merged = {};
    used = false(1, length(gm_components));
    for i = 1:length(gm_components)
        if used(i), continue; end
        w_i = gm_components{i}.w;
        x_i = gm_components{i}.x;
        P_i = gm_components{i}.P;
        S_idx = i;
        for j = i+1:length(gm_components)
            if used(j), continue; end
            x_j = gm_components{j}.x;
            P_j = gm_components{j}.P;
            diff = x_j - x_i;
            Scov = P_i(1:2,1:2);
            d2 = diff(1:2)' * (Scov \ diff(1:2));
            if d2 < params.mergeThreshold
                S_idx(end+1) = j;
            end
        end
        Ws = 0; xsum = zeros(4,1); Psum = zeros(4,4);
        for idx_s = S_idx
            Ws = Ws + gm_components{idx_s}.w;
        end
        for idx_s = S_idx
            wj = gm_components{idx_s}.w;
            xj = gm_components{idx_s}.x;
            Pj = gm_components{idx_s}.P;
            xsum = xsum + wj * xj;
        end
        xbar = xsum / (Ws + eps);
        for idx_s = S_idx
            wj = gm_components{idx_s}.w;
            xj = gm_components{idx_s}.x;
            Pj = gm_components{idx_s}.P;
            diff = xj - xbar;
            Psum = Psum + wj * (Pj + diff * diff');
        end
        Pbar = Psum / (Ws + eps);
        merged{end+1} = struct('w', Ws, 'x', xbar, 'P', (Pbar + Pbar')/2);
        used(S_idx) = true;
    end
    gm_components = merged;

    if length(gm_components) > params.maxComponents
        ws = arrayfun(@(c) c{1}.w, gm_components);
        ws = zeros(1,length(gm_components));
        for ii = 1:length(gm_components), ws(ii) = gm_components{ii}.w; end
        [~, ord] = sort(ws,'descend');
        keep = ord(1:params.maxComponents);
        gm_components = gm_components(keep);
    end

    candidates = [];
    for i = 1:length(gm_components)
        if gm_components{i}.w >= params.extractWeightTh
            candidates(end+1).w = gm_components{i}.w;
            candidates(end).x = gm_components{i}.x;
            candidates(end).P = gm_components{i}.P;
        end
    end
    selected = false(1, length(candidates));
    final_targets = struct('w',0,'x',zeros(4,1),'P',zeros(4,4));
    for i = 1:length(candidates)
        if selected(i), continue; end
        xi = candidates(i).x;
        final_targets(end+1).w = candidates(i).w;
        final_targets(end).x = candidates(i).x;
        final_targets(end).P = candidates(i).P;
        for j = i+1:length(candidates)
            if selected(j), continue; end
            xj = candidates(j).x;
            if norm(xi(1:2)-xj(1:2)) < 1.0
                selected(j) = true;
            end
        end
    end

    meas_extract = [];
    for i = 2:length(final_targets)
        meas_extract = [meas_extract; final_targets(i).x(1:2)'];
    end

    for ti = 1:length(tracks)
        tracks(ti).state = F * tracks(ti).state;
        tracks(ti).P = F * tracks(ti).P * F' + Q;
    end

    available_targets = 1:size(meas_extract,1);
    assigned_tracks = false(1, length(tracks));
    assigned_targets = false(1, size(meas_extract,1));
    if ~isempty(meas_extract) && ~isempty(tracks)
        D = zeros(length(tracks), size(meas_extract,1));
        for ti = 1:length(tracks)
            for zj = 1:size(meas_extract,1)
                D(ti,zj) = norm(tracks(ti).state(1:2) - meas_extract(zj,:)');
            end
        end
        while true
            [minVal, idx] = min(D(:));
            if isempty(minVal) || ~isfinite(minVal), break; end
            [ti, zj] = ind2sub(size(D), idx);
            if minVal > extract_spatial_gate
                break;
            end
            assigned_tracks(ti) = true;
            assigned_targets(zj) = true;
            z = meas_extract(zj,:)';
            S = H * tracks(ti).P * H' + R;
            Kk = tracks(ti).P * H' / S;
            xin = tracks(ti).state + Kk * (z - H * tracks(ti).state);
            Pin = (eye(size(tracks(ti).P)) - Kk * H) * tracks(ti).P;
            tracks(ti).state = xin;
            tracks(ti).P = (Pin + Pin')/2;
            if isempty(tracks(ti).history)
                tracks(ti).history = nan(2, nFrames);
            end
            tracks(ti).history(:,k) = z;
            tracks(ti).lastVisibleFrame = k;
            tracks(ti).consecutiveInvisible = 0;
            tracks(ti).totalVisibleCount = tracks(ti).totalVisibleCount + 1;
            if tracks(ti).totalVisibleCount >= track_params.new_track_confirm_frames
                tracks(ti).confirmed = true;
            end
            D(ti,:) = Inf;
            D(:,zj) = Inf;
        end
    end

    for ti = 1:length(tracks)
        if ~assigned_tracks(ti)
            if isempty(tracks(ti).history), tracks(ti).history = nan(2,nFrames); end
            pred_meas = H * tracks(ti).state;
            tracks(ti).history(:,k) = pred_meas;
            tracks(ti).consecutiveInvisible = tracks(ti).consecutiveInvisible + 1;
        end
    end

    for zj = 1:size(meas_extract,1)
        if assigned_targets(zj), continue; end
        z = meas_extract(zj,:)';
        newtr.id = nextTrackID; nextTrackID = nextTrackID + 1;
        newtr.state = [z; 0; 0];
        newtr.P = track_params.init_P;
        newtr.history = nan(2, nFrames);
        newtr.history(:,k) = z;
        newtr.lastVisibleFrame = k;
        newtr.consecutiveInvisible = 0;
        newtr.totalVisibleCount = 1;
        newtr.confirmed = (newtr.totalVisibleCount >= track_params.new_track_confirm_frames);
        if nextTrackID-1 == 1
            tracks = newtr;
        else
            tracks(end+1) = newtr;
        end
    end

    alive_idx = [];
    for ti = 1:length(tracks)
        tr = tracks(ti);
        if tr.consecutiveInvisible >= track_params.max_missed_frames
            continue;
        end
        if (tr.totalVisibleCount < track_params.new_track_confirm_frames) && (tr.consecutiveInvisible > 0)
            continue;
        end
        alive_idx(end+1) = ti;
    end
    tracks = tracks(alive_idx);

    if display_on
        figure(1); clf;
        if exist('PCF','var')
            imagesc(image_x, image_y, abs(PCF(:,:,min(k,end)))/max(max(abs(PCF(:,:,1))))); colormap('gray'); set(gca,'YDir','normal');
        else
            imagesc(zeros(100)); set(gca,'YDir','normal');
        end
        hold on;
        for i = 1:length(gm_components)
            plot(gm_components{i}.x(1), gm_components{i}.x(2), 'co', 'MarkerSize', 3+4*sqrt(gm_components{i}.w));
        end
        for i = 1:size(meas_extract,1)
            plot(meas_extract(i,1), meas_extract(i,2), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
        end
        colors = lines(max(1,length(tracks)));
        for ti = 1:length(tracks)
            hist = tracks(ti).history;
            if isempty(hist), continue; end
            valid = ~isnan(hist(1,:));
            plot(hist(1,valid), hist(2,valid), '-', 'Color', colors(mod(ti-1,size(colors,1))+1,:), 'LineWidth', 1.2);
            plot(tracks(ti).state(1), tracks(ti).state(2), 's', 'MarkerFaceColor', colors(mod(ti-1,size(colors,1))+1,:), 'MarkerEdgeColor','k');
            trace_save(cm).position = tracks(ti).history; cm = cm+1;
        end
        title(sprintf('GM-PHD 帧 %d', k));
        drawnow;
    end

end


%% dalete trace
seen = containers.Map('KeyType', 'double', 'ValueType', 'logical');
keep = true(size(trace_save));
for i = numel(trace_save):-1:1
    pos = trace_save(i).position;
    first_val = pos(find(~isnan(pos), 1, 'first'));
    
    if isempty(first_val) || isKey(seen, first_val)
        keep(i) = false;
    else
        seen(first_val) = true;
    end
end

trace = trace_save(keep);

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

save(['simulation_detect_result_gmphd.mat'])

figure;
if exist('PCF','var')
    imagesc(image_x,image_y,abs(PCF(:,:,1))/max(max(abs(PCF(:,:,1))))); colormap('gray');
    set(gca,'YDir','normal');
end
hold on;
colors = lines(max(1,length(trace)));
for i = 1:length(trace)
    pos = trace(i).position;
    plot(pos(1,:), pos(2,:), '-', 'LineWidth', 1.5);
end
xlabel('Azimuth/m'); ylabel('Range/m');
hold off;
