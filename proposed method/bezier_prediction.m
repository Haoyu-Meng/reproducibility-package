function [prediction,fittrace] = bezier_prediction(trace)

    fittrace = bezier(trace);
    
    vx = diff(fittrace(1,:));
    vy = diff(fittrace(2,:));
    
    ax = diff(vx);
    ay = diff(vy);
    
    a_next_x = predict_loess(ax);
    a_next_y = predict_loess(ay);
    
    v_now_x = vx(end);
    v_now_y = vy(end);
    
    x_now = fittrace(1,end);
    y_now = fittrace(2,end);
    
    % s = s0 + v0*t + 0.5*a*t^2
    predicted_x = x_now + v_now_x + 0.5 * a_next_x;
    predicted_y = y_now + v_now_y + 0.5 * a_next_y;
    
    prediction = [predicted_x; predicted_y];

end


function predicted_value = predict_loess(x, span)

    if nargin < 2
        span = 0.2;
    end
    
    t = 1:length(x);
    
    window_size = min(30, round(length(x)*span));
    recent_t = t(end-window_size+1:end);
    recent_x = x(end-window_size+1:end);
    
    weights = exp(-0.5*((recent_t - t(end)) / (window_size/4)).^2);
    p = polyfit_weighted(recent_t, recent_x, 1, weights);
    predicted_value = polyval(p, length(x)+1);
end

function p = polyfit_weighted(x, y, n, w)
    W = diag(sqrt(w));
    A = zeros(length(x), n+1);
    for i = 0:n
        A(:,i+1) = x.^i;
    end
    p = (W*A) \ (W*y(:));
    p = p(end:-1:1)';
end


%% bezierfunction
% function fitted_curve = bezier(track_curve,method)

% Input
% track_curve — Historical tracking trajectory, 2 × number_of_frames
% method — Method used: 1 — directly fit a fixed-order Bezier curve; 2 — select an appropriate order
% Output
% fitted_curve — Fitted curve, 2 × number_of_frames

% end
