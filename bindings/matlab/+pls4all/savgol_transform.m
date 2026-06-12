function Xout = savgol_transform(X, varargin)
% pls4all.savgol_transform  Apply Savitzky-Golay filtering row-wise.
%
%   XOUT = pls4all.savgol_transform(X, 'window_length', 11, 'polyorder', 3)

opts = parse_options(varargin{:});
Xout = pls4all.n4m_preprocess_mex( ...
    'savgol', double(X), int32(opts.window_length), int32(opts.polyorder), ...
    int32(opts.deriv), double(opts.delta), opts.mode, double(opts.cval));
end

function opts = parse_options(varargin)
opts = struct( ...
    'window_length', 11, ...
    'polyorder', 3, ...
    'deriv', 0, ...
    'delta', 1.0, ...
    'mode', 'mirror', ...
    'cval', 0.0);
if mod(numel(varargin), 2) ~= 0
    error('pls4all:nargin', 'Options must be name/value pairs');
end
for idx = 1:2:numel(varargin)
    name = char(varargin{idx});
    if strcmp(name, 'window')
        name = 'window_length';
    end
    opts.(name) = varargin{idx + 1};
end
if isstring(opts.mode)
    opts.mode = char(opts.mode);
end
end
