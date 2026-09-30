clc;
close all;
clear all;
load(['simulation_detect_result.mat']);

BP_image_12 = PCF;

% ------------------ parameters ------------------
params.padding = 1.6;
params.output_sigma_factor = 1/10;
params.sigma = 0.2;
params.lambda = 1.4e-2;
params.interp_factor = 0.0016;
params.area_sz = [100,100];
params.expand_attempts = 1;
params.visualize = true;

% ------------------ init ------------------
Resolution = image_x(2) - image_x(1);
frame = size(BP_image_12, 3);

ttt = zeros(target_num(1),2);
for ii = 1:target_num(1)
    ttt(ii,1) = round((target_loc{ii,1}(1)-image_x(1))/(image_x(2)-image_x(1)));
    ttt(ii,2) = round((target_loc{ii,1}(2)-image_y(1))/(image_y(2)-image_y(1)));
end
params.initial_positions = ttt;

Nt = size(params.initial_positions,1);
for tt = 1:Nt
    trace(tt).id = tt;
    trace(tt).position = nan(frame,2);
end




for tt = 1:Nt

positions = nan(frame, 2, 3);
rect_position_save = nan(frame, 4, 3);
theta_degree = nan(frame,1);
theta_rotation = nan(frame,1);

diagnostics.params = params;
diagnostics.positions = [];
diagnostics.rects = [];
diagnostics.theta_degree = [];
diagnostics.theta_rotation = [];

time = 0;
tic();

for K = 1:frame

    if K == 1
        row = params.initial_positions(tt,1); col = params.initial_positions(tt,2); % 使用与原脚本一致的默认
        if length(row) > 1
            row = (row(1) + row(2)) / 2;
            col = (col(1) + col(2)) / 2;
        end
        x_initial = row;
        y_initial = col;
        x_min = max(col - 10,1); 
        x_max = min(col + 10,length(image_x));
        y_min = max(row - 10,1); y_min = min(y_min,length(image_y));
        y_max = min(row + 10,length(image_y));
        BP_norma = abs(BP_image_12(:,:,1)) / max(max(abs(BP_image_12(:,:,1))));
        px_left = nansum(abs(BP_norma(y_min:y_max, x_min)));
        px_right = nansum(abs(BP_norma(y_min:y_max, x_max)));
        py_up = nansum(abs(BP_norma(y_min, x_min:x_max)));
        py_down = nansum(abs(BP_norma(y_max, x_min:x_max)));

        for i_expand = 1:params.expand_attempts
            if px_left > 0.015
                x_min = x_min - 1;
            end
            if px_right > 0.015
                x_max = x_max + 1;
            end
            if py_up > 0.01
                y_min = y_min - 1;
                if y_min == 0
                    y_min = y_min + 1;
                end
            end
            if py_down > 0.01
                y_max = y_max + 1;
                y_max = min(y_max,length(image_y));
            end
            px_left = nansum(abs(BP_norma(y_min:y_max, x_min)));
            px_right = nansum(abs(BP_norma(y_min:y_max, x_max)));
            py_up = nansum(abs(BP_norma(y_min, x_min:x_max)));
            py_down = nansum(abs(BP_norma(y_max, x_min:x_max)));
        end

        Nx = x_max - x_min;
        Ny = y_max - y_min;
        target_sz = [Nx, Ny];
        sz = floor(target_sz * (1 + params.padding));

        output_sigma = sqrt(prod(target_sz)) * params.output_sigma_factor;
        [rs, cs] = ndgrid((1:sz(2)) - floor(sz(2)/2), (1:sz(1)) - floor(sz(1)/2));
        y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
        yf = fft2(y);

        cos_window = hann(sz(2)) * hann(sz(1))';
        pos = [x_initial, y_initial];
    end
    area_sz = params.area_sz;
    row_min = max(1, round(pos(2) - area_sz(2)/2));
    row_max = min(length(image_y), round(pos(2) + area_sz(2)/2));
    col_min = max(1, round(pos(1) - area_sz(1)/2));
    col_max = min(length(image_x), round(pos(1) + area_sz(1)/2));

    target_area = BP_image_12(row_min:row_max, col_min:col_max, K);

    New_image = abs(mat2gray(abs(target_area)));
    thresh = graythresh(New_image);
    New_image1 = im2bw(New_image, thresh);

    M00 = Moment(New_image1, 0, 0);
    M01 = Moment(New_image1, 0, 1);
    M10 = Moment(New_image1, 1, 0);
    M11 = Moment(New_image1, 1, 1);
    M20 = Moment(New_image1, 2, 0);
    M02 = Moment(New_image1, 0, 2);

    Xcenter = [M10 / max(M00, eps), M01 / max(M00, eps)];

    U20 = M20 / max(M00, eps) - Xcenter(1)^2;
    U02 = M02 / max(M00, eps) - Xcenter(2)^2;
    U11 = M11 / max(M00, eps) - Xcenter(1) * Xcenter(2);
    C_ = [U20 U11; U11 U02];

    [Udirection, SV, ~] = svd(C_);
    SV = sqrt(SV);

    a = sqrt(SV(1) * M00 / pi / SV(4));
    b = sqrt(SV(4) * M00 / pi / SV(1));
    theta_degree(K) = rad2deg(atan(Udirection(2,1) / Udirection(1,1)));
    theta = deg2rad(theta_degree(K));

    im = abs(BP_image_12(:,:,K));
    im = abs(mat2gray(im) - 1) * 255;

    if K > 1
        [z, theta_rotation(K)] = get_subwindow_rotate(im, pos, sz, cos_window, theta_degree(K-1));
    else
        [z, theta_rotation(K)] = get_subwindow_rotate(im, pos, sz, cos_window, theta_degree(K));
    end

    if K > 1
        k = dense_gauss_kernel(params.sigma, z, model_x);
        response = real(ifft2(alphaf .* fft2(k)));
        [row_r, col_r] = find(response == max(response(:)), 1);
        pos = pos - floor(sz/2) + [col_r, row_r];
    end

    [x, ~] = get_subwindow_rotate(im, pos, sz, cos_window, theta_degree(K));

    k = dense_gauss_kernel(params.sigma, x);
    new_alphaf = yf ./ (fft2(k) + params.lambda);
    new_x = x;

    if K == 1
        alphaf = new_alphaf;
        model_x = x;
    else
        alphaf = (1 - params.interp_factor) * alphaf + params.interp_factor * new_alphaf;
        model_x = (1 - params.interp_factor) * model_x + params.interp_factor * new_x;
    end

    pos(1) = max(1, min(pos(1), length(image_x)));
    pos(2) = max(1, min(pos(2), length(image_y)));

    positions(K,:,2) = [image_x(pos(1)), image_y(pos(2))];

    rect_position = [(pos([1,2]) - target_sz([1,2])/2), target_sz([1,2])];
    rect_position(1) = max(1, rect_position(1));
    rect_position(2) = max(1, rect_position(2));
    rect_position_save(K,:,2) = [image_x(round(rect_position(1))), image_y(round(rect_position(2))), rect_position(3) * Resolution, rect_position(4) * Resolution];

    diagnostics.positions = [diagnostics.positions; positions(K,:,2)];
    diagnostics.rects = [diagnostics.rects; rect_position_save(K,:,2)];
    diagnostics.theta_degree = [diagnostics.theta_degree; theta_degree(K)];
    diagnostics.theta_rotation = [diagnostics.theta_rotation; theta_rotation(K)];

    if params.visualize
        figure(1);
        imagesc(image_x, image_y, abs(BP_image_12(:,:,K)) / max(max(abs(BP_image_12(:,:,K)))));
        hold on;
        rect_handle = rectangle('Position', rect_position_save(K,:,2), 'EdgeColor', 'r', 'LineWidth', 1);
        plot(positions(1:K,1,2), positions(1:K,2,2), 'r.', 'MarkerSize', 4);
        xlabel('X/m', 'FontSize', 16); ylabel('Y/m', 'FontSize', 16);
        set(gca, 'YDir', 'normal', 'FontSize', 16);
        txt = ['Frame ' num2str(K)];
        text(2, 17.5, txt, 'color', 'w', 'FontSize', 20);
        drawnow;
    end

end

time = time + toc();

trace(tt).position = positions(:,:,2)';
trace(tt).diagnostics = diagnostics;

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

save(['simulation_detect_result_KCF.mat'])

%% 
for i_loc = 1:length(trace) 
    plot(trace(i_loc).position(1,2:end),trace(i_loc).position(2,2:end),'LineWidth',1);
    hold on;
end
set(gca,'fontsize',16,'fontname','Times New Roman');
xlabel('Azimuth/m','fontsize',16,'fontname','Times New Roman');
ylabel('Range/m','fontsize',16,'fontname','Times New Roman');
xlim([-7 7]);ylim([3 20]);

%% =================== function ===================

% function M = Moment(image, mi, mj)
% % Compute image moments
% end

% function [out, theta_rotation] = get_subwindow_rotate(im, pos, sz, cos_window, theta)
% % Extract a sub-window of size sz centered at pos from the image, and rotate it by theta degrees.
% end

% function k = dense_gauss_kernel(sigma, x, y)
% % Compute a dense Gaussian kernel with bandwidth sigma
% end