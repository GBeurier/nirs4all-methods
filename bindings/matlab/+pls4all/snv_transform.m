function Xout = snv_transform(X, varargin)
% pls4all.snv_transform  Apply row-wise Standard Normal Variate.
%
%   XOUT = pls4all.snv_transform(X)
%   XOUT = pls4all.snv_transform(X, 'ddof', 0)

opts = parse_options(varargin{:});
Xout = pls4all.n4m_preprocess_mex( ...
    'snv', double(X), logical_scalar(opts.with_mean), logical_scalar(opts.with_std), int32(opts.ddof));
end

function opts = parse_options(varargin)
opts = struct('with_mean', true, 'with_std', true, 'ddof', 0);
if mod(numel(varargin), 2) ~= 0
    error('pls4all:nargin', 'Options must be name/value pairs');
end
for idx = 1:2:numel(varargin)
    name = char(varargin{idx});
    opts.(name) = varargin{idx + 1};
end
end

function value = logical_scalar(x)
value = double(logical(x));
end
