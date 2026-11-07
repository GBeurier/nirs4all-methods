function Xout = snv_transform(X, varargin)
% n4m.snv_transform  Apply row-wise Standard Normal Variate.
%
%   XOUT = n4m.snv_transform(X)
%   XOUT = n4m.snv_transform(X, 'ddof', 0)

opts = parse_options(varargin{:});
Xout = n4m.n4m_preprocess_mex( ...
    'snv', double(X), opts.with_mean, opts.with_std, opts.ddof);
end

function opts = parse_options(varargin)
opts = struct('with_mean', true, 'with_std', true, 'ddof', 0);
if mod(numel(varargin), 2) ~= 0
    error('n4m:nargin', 'Options must be name/value pairs');
end
for idx = 1:2:numel(varargin)
    name = char(varargin{idx});
    if ~isfield(opts, name)
        error('n4m:parameter', 'Unknown SNV option: %s', name);
    end
    opts.(name) = varargin{idx + 1};
end
end
