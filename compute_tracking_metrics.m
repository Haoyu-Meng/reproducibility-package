function metrics = compute_tracking_metrics(gt_trajs, est_trajs, varargin)

% Input:
% gt_trajs - Cell array, each element is a [2×T] ground-truth trajectory matrix (x,y)
% est_trajs - Cell array, each element is a [2×T] estimated trajectory matrix (x,y)
% Optional parameters:
% 'Alpha' - Matching threshold (meters), default 0.5
% 'AlphaRange' - HOTA multi-threshold sweep range, default 0.1:0.1:1.0
% 'OSPA_c' - OSPA cutoff distance, default 1.5
% 'OSPA_p' - OSPA order, default 2
%
% Output:
% metrics - Structure containing the following fields: HOTA, OSPA_mean

%% parameter
p = inputParser;
addParameter(p, 'Alpha', 0.5);
addParameter(p, 'AlphaRange', 0.1:0.1:1.0);
addParameter(p, 'OSPA_c', 1.5);
addParameter(p, 'OSPA_p', 2);
parse(p, varargin{:});

Alpha = p.Results.Alpha;
AlphaRange = p.Results.AlphaRange;
OSPA_c = p.Results.OSPA_c;
OSPA_p = p.Results.OSPA_p;

% 支持struct输入（自动提取position字段）
if ~iscell(gt_trajs) && isstruct(gt_trajs)
    gt_trajs = extractPositionField(gt_trajs);
end
if ~iscell(est_trajs) && isstruct(est_trajs)
    est_trajs = extractPositionField(est_trajs);
end

numGT = length(gt_trajs);
numEst = length(est_trajs);

%% 
allFrames = [];
for i = 1:numGT
    if gt_trajs{i}(2,1) < 1
        index = find(gt_trajs{i}(2,:)>1);
        gt_trajs{i} = gt_trajs{i}(:,index(1):end);
    end
    allFrames = [allFrames, size(gt_trajs{i}, 2)];
end
for i = 1:numEst
    allFrames = [allFrames, size(est_trajs{i}, 2)];
end
if isempty(allFrames)
    totalFrames = 0;
else
    totalFrames = max(allFrames);
end

%% 
gt_matrix = false(numGT, totalFrames);
est_matrix = false(numEst, totalFrames);
gt_positions = nan(numGT, totalFrames, 2);
est_positions = nan(numEst, totalFrames, 2);

for i = 1:numGT
    T = size(gt_trajs{i}, 2);
    gt_matrix(i, 1:T) = true;
    gt_positions(i, 1:T, :) = gt_trajs{i}';
end

for i = 1:numEst
    T = size(est_trajs{i}, 2);
    est_matrix(i, 1:T) = true;
    est_positions(i, 1:T, :) = est_trajs{i}';
end

%% HOTA
hota_values = zeros(length(AlphaRange), 1);
for ai = 1:length(AlphaRange)
    hota_values(ai) = computeHOTA_single(gt_matrix, est_matrix, gt_positions, ...
                                          est_positions, AlphaRange(ai));
end
metrics.HOTA = mean(hota_values);

%% OSPA
metrics.OSPA_mean = computeOSPA(gt_trajs, est_trajs, OSPA_c, OSPA_p);

end

%% ==================== function ====================

function trajs = extractPositionField(structArray)
trajs = {};
counter = 0;
for i = 1:length(structArray)
    if isfield(structArray(i), 'position') && ~isempty(structArray(i).position)
        counter = counter + 1;
        trajs{counter} = structArray(i).position;
    end
end
end

function hota = computeHOTA_single(gt_matrix, est_matrix, gt_pos, est_pos, alpha)

[~, totalFrames] = size(gt_matrix);

% A loop for the frame-level matching in the HOTA metric computation. 
% For each frame, it first identifies the ground-truth trajectories present 
% in that frame (activeGT) and the estimated trajectories present in that frame (activeEst);
% if either is empty, the frame is skipped. A cost matrix costMat is then constructed, 
% where each element represents the Euclidean distance between a ground-truth trajectory and 
% an estimated trajectory at that frame. The Hungarian algorithm is then 
% applied to solve the optimal one-to-one matching, yielding the assignment matrix. 
% Finally, all matched pairs are traversed, and only those with a successful match and 
% a distance below the threshold alpha are recorded in matches as 
% (ground-truth index, estimated index, frame number). After the loop completes, 
% matches contains all one-to-one frame-level matching pairs that satisfy
% the distance threshold.

TP = size(matches, 1);
FP = sum(est_matrix(:)) - TP;
FN = sum(gt_matrix(:)) - TP;

if TP + FP + FN > 0
    DetA = TP / (TP + FP + FN);
else
    DetA = 0;
end

if TP == 0
    AssA = 0;
else
    assocScores = zeros(TP, 1);
    for m = 1:TP
        gIdx = matches(m, 1);
        eIdx = matches(m, 2);
        
        samePair = matches(matches(:, 1) == gIdx & matches(:, 2) == eIdx, :);
        TPA = size(samePair, 1);
        
        gtTotal = sum(gt_matrix(gIdx, :));
        estTotal = sum(est_matrix(eIdx, :));
        
        FNA = gtTotal - TPA;
        FPA = estTotal - TPA;
        
        if TPA + FNA + FPA > 0
            assocScores(m) = TPA / (TPA + FNA + FPA);
        else
            assocScores(m) = 0;
        end
    end
    AssA = mean(assocScores);
end

hota = sqrt(DetA * AssA);
end

function ospa_mean = computeOSPA(gtTrajs, estTrajs, c, p)

    numGT  = length(gtTrajs);
    numEst = length(estTrajs);

    Tlist = [cellfun(@(x) size(x,2), gtTrajs), ...
             cellfun(@(x) size(x,2), estTrajs)];
    if isempty(Tlist)
        ospa_mean = 0;
        return;
    end
    totalFrames = max(Tlist);

    ospa_vals = [];

    for k = 1:totalFrames
        X = [];
        for i = 1:numGT
            if k <= size(gtTrajs{i},2)
                X = [X, gtTrajs{i}(:,k)];
            end
        end

        Y = [];
        for j = 1:numEst
            if k <= size(estTrajs{j},2)
                Y = [Y, estTrajs{j}(:,k)];
            end
        end

        m = size(X,2);
        n = size(Y,2);

        if m == 0 && n == 0
            continue;
        end

        N = max(m,n);

        if m>0 && n>0
            D = zeros(m,n);
            for i = 1:m
                for j = 1:n
                    d = norm(X(:,i)-Y(:,j));
                    D(i,j) = min(d,c)^p;
                end
            end
            % Hungarian algorithm like HOTA
            matchedCost = sum(sum(assign .* D));
        else
            matchedCost = 0;
        end

        cardPenalty = c^p * abs(n - m);

        ospa_k = ((matchedCost + cardPenalty) / N)^(1/p);
        ospa_vals(end+1) = ospa_k;
    end

    if isempty(ospa_vals)
        ospa_mean = 0;
    else
        ospa_mean = mean(ospa_vals);
    end
end