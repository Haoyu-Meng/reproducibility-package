close all; 
clear all; 
clc;
load(['simulation_detect_result.mat']);

measure_loc = target_loc;

%% parameters
params.maxHypotheses = 50;
params.kBest = 3;
params.gateThreshold = 9;
params.Pd = 0.95;
params.R = diag([0.05,0.05]);
params.F = eye(6);
params.Q = diag([0.01,0.01,0.01,0.01,0.01,0.01]);
params.P0_pos = 1.0;
params.P0_vel = 1.0;
dt = 8e-3;
cm = 1;

% ------------- trajectory management -------------
pause_threshold = 20;
confirm_threshold = 6;

% ------------- inital -------------
nFrames = size(measure_loc,2);
hypotheses = {};  % start with empty until frame1 init
nextTrackID = 1;
history_location = {}; % collect histories for plotting later
H = [1 0 0 0 0 0; 0 1 0 0 0 0];

% ---------------- main frame loop ----------------
for t = 1:nFrames
    measList = {};
    for m = 1:target_num(t)
        if isempty(measure_loc{m,t}), continue; end
        measList{end+1} = measure_loc{m,t}(:);
    end
    M = numel(measList);

    if t == 1
        hyp.tracks = [];
        hyp.logWeight = 0;
        for i = 1:M
            z = measList{i};
            tr.id = nextTrackID; nextTrackID = nextTrackID + 1;
            tr.x = zeros(6,1);
            tr.x(1:2) = z;
            tr.P = diag([params.P0_pos, params.P0_pos, params.P0_vel, params.P0_vel, 1, 1]);
            tr.history = nan(2, nFrames);
            tr.history(:,t) = z;
            tr.if_pause = 0;
            tr.if_begin = 1;
            tr.start_frame = t;
            tr.alive = true;
            hyp.tracks = [hyp.tracks; tr];
        end
        hypotheses = {hyp};
        continue;
    end

    newHyps = {};
    for hidx = 1:length(hypotheses)
        parent = hypotheses{hidx};
        tracks = parent.tracks;
        Tn = length(tracks);
        candidates = cell(Tn,1);
        for ti = 1:Tn
            tr = tracks(ti);
            F = eye(6); F(1,3)=dt; F(2,4)=dt;
            Q = params.Q;
            xpred = F * tr.x;
            Ppred = F * tr.P * F' + Q;
            tracks(ti).xpred = xpred;
            tracks(ti).Ppred = Ppred;
            zpred = H * xpred;
            S = H * Ppred * H' + params.R;
            Sinv = inv(S);
            cand = [];
            for mi = 1:M
                z = measList{mi};
                innov = z - zpred;
                md2 = innov' * Sinv * innov;
                if md2 <= params.gateThreshold
                    loglike = -0.5 * md2 - 0.5 * log(det(2*pi*S));
                    cand = [cand; struct('measIdx',mi,'loglike',loglike,'isMissed',false)];
                end
            end
            missLog = log(1 - params.Pd + eps);
            cand = [cand; struct('measIdx',0,'loglike',missLog,'isMissed',true)];
            [~, idxs] = sort([cand.loglike], 'descend');
            kkeep = min(params.kBest, numel(idxs));
            candidates{ti} = cand(idxs(1:kkeep));
        end

        partials = struct('assigned', {}, 'assignments', {}, 'logprob', {});
        partials(1).assigned = false(1,M);
        partials(1).assignments = zeros(1,Tn);
        partials(1).logprob = parent.logWeight;
        beamWidth = params.maxHypotheses;
        for ti = 1:Tn
            newPartials = [];
            for p = 1:length(partials)
                base = partials(p);
                for c = 1:length(candidates{ti})
                    cand = candidates{ti}(c);
                    if cand.isMissed
                        np = base;
                        np.assignments(ti) = 0;
                        np.logprob = base.logprob + cand.loglike + log(0.01+eps);
                        np.assigned = base.assigned;
                        newPartials = [newPartials, np];
                    else
                        mi = cand.measIdx;
                        if ~base.assigned(mi)
                            np = base;
                            np.assignments(ti) = mi;
                            np.logprob = base.logprob + cand.loglike + log(params.Pd+eps);
                            np.assigned = base.assigned;
                            np.assigned(mi) = true;
                            newPartials = [newPartials, np];
                        end
                    end
                end
            end
            if isempty(newPartials)
                partials = [];
                break;
            end
            [~, ord] = sort([newPartials.logprob], 'descend');
            keep = min(beamWidth, numel(ord));
            partials = newPartials(ord(1:keep));
        end

        for p = 1:length(partials)
            assign = partials(p).assignments;
            newHyp.tracks = [];
            newHyp.logWeight = partials(p).logprob;
            assignedMeas = false(1,M);
            for ti = 1:Tn
                tr = tracks(ti);
                xpred = tr.xpred; Ppred = tr.Ppred;
                measIdx = assign(ti);
                if measIdx == 0
                    tr.x = xpred;
                    tr.P = Ppred;
                    tr.if_pause = tr.if_pause + 1;
                    tr.history(:,t) = H * xpred;
                else
                    z = measList{measIdx};
                    S = H * Ppred * H' + params.R;
                    K = Ppred * H' / S;
                    xin = xpred + K * (z - H*xpred);
                    Pout = (eye(size(Ppred)) - K*H) * Ppred;
                    tr.x = xin; tr.P = (Pout + Pout')/2;
                    tr.if_pause = 0;
                    tr.if_begin = tr.if_begin + 1;
                    tr.history(:,t) = z;
                    assignedMeas(measIdx) = true;
                end
                tr.alive = true;
                newHyp.tracks = [newHyp.tracks; tr];
            end

            for mi = 1:M
                if ~assignedMeas(mi)
                    z = measList{mi};
                    newtr.id = nextTrackID; nextTrackID = nextTrackID + 1;
                    newtr.x = zeros(6,1); newtr.x(1:2) = z;
                    newtr.P = diag([params.P0_pos, params.P0_pos, params.P0_vel, params.P0_vel, 1, 1]);
                    newtr.history = nan(2, nFrames);
                    newtr.history(:,t) = z;
                    newtr.if_pause = 0;
                    newtr.if_begin = 1;
                    newtr.start_frame = t;
                    newtr.alive = true;
                    newtr.xpred = zeros(6,1); newtr.x(1:2) = z;
                    newtr.Ppred = diag([params.P0_pos, params.P0_pos, params.P0_vel, params.P0_vel, 1, 1]);
                    newHyp.tracks = [newHyp.tracks; newtr];
                end
            end

            survivors = [];
            for tr_i = 1:length(newHyp.tracks)
                tr = newHyp.tracks(tr_i);
                if tr.if_pause > pause_threshold
                    continue;
                end
                if (tr.if_begin < confirm_threshold) && (tr.if_pause > 0)
                    continue;
                end
                survivors = [survivors; tr];
            end
            newHyp.tracks = survivors;

            newHyps{end+1} = newHyp;
        end 
    end

    if isempty(newHyps)
        newHyps = hypotheses;
    end

    weights = cellfun(@(hh) hh.logWeight, newHyps);
    [~, ord] = sort(weights, 'descend');
    keepN = min(params.maxHypotheses, numel(newHyps));
    hypotheses = newHyps(ord(1:keepN));

    [~, bestIdx] = max(cellfun(@(hh) hh.logWeight, hypotheses));
    bestHyp = hypotheses{bestIdx};

    location = {};
    target_loc_pred = {};
    P_out = {};
    history_location = {};
    for ti = 1:length(bestHyp.tracks)
        tr = bestHyp.tracks(ti);
        location{ti} = tr.history(:, find(~isnan(tr.history(1,:)), 1, 'last')); % last non-NaN
        if isempty(location{ti})
            location{ti} = [1e6;1e6;1e6];
        end
        F = eye(6); F(1,3)=dt; F(2,4)=dt;
        xpred = F * tr.x;
        Ppred = F * tr.P * F' + params.Q;
        predMeas = H * xpred;
        target_loc_pred{ti} = predMeas;
        if isempty(tr.history)
            P_out{ti} = tr.P;
        else
            P_out{ti} = tr.P;
        end
        history_location{ti} = tr.history;
    end

    figure(129);
    imagesc(image_x,image_y,(abs(PCF(:,:,t))));
    axis normal; set(gca,'YDir','normal'); colormap("jet");
    hold on;
    for i_loc = 1:length(history_location) 
        if length(find(~isnan(history_location{i_loc}(1,:)))) < 10
            continue
        end
        plot(history_location{i_loc}(1,2:end),history_location{i_loc}(2,2:end),'w-','LineWidth',1);
        trace(cm).position = history_location{i_loc}(1:2,:); cm = cm+1;
    end
    hold off;

end

%% delete trace
seen = containers.Map('KeyType', 'double', 'ValueType', 'logical');
keep = true(size(trace));
for i = numel(trace):-1:1
    pos = trace(i).position;
    first_val = pos(find(~isnan(pos), 1, 'first'));
    
    if isempty(first_val) || isKey(seen, first_val)
        keep(i) = false;
    else
        seen(first_val) = true;
    end
end

trace = trace(keep);

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
    'OSPA_p');

save(['simulation_detect_result_MHT_KM.mat'])


%% ------------- plot similar to original -------------
figure;
imagesc(image_x, image_y, abs(PCF(:,:,t))); axis normal; set(gca,'YDir','normal'); colormap("jet"); hold on;
for i_loc = 1:length(trace)
    plot(trace(i_loc).position(1, :), trace(i_loc).position(2, :), '-', 'LineWidth',1);
end
hold off;
set(gca,'fontsize',16,'fontname','Times New Roman');
xlabel('Azimuth/m','fontsize',16,'fontname','Times New Roman');
ylabel('Range/m','fontsize',16,'fontname','Times New Roman');
xlim([-5 6]); ylim([1 10]);