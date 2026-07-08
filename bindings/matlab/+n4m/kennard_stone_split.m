function split = kennard_stone_split(X, varargin)
% n4m.kennard_stone_split  Kennard-Stone train/test split.
%
%   SPLIT = n4m.kennard_stone_split(X)
%   SPLIT = n4m.kennard_stone_split(X, 'test_size', 0.25, 'zero_based', false)

opts = parse_options(varargin{:});
split = n4m.n4m_split_mex( ...
    'kennard_stone', double(X), double(opts.test_size), logical_scalar(opts.zero_based));
split.train = split.train(:);
split.test = split.test(:);
end

function opts = parse_options(varargin)
opts = struct('test_size', 0.25, 'zero_based', false);
if mod(numel(varargin), 2) ~= 0
    error('n4m:nargin', 'Options must be name/value pairs');
end
for idx = 1:2:numel(varargin)
    name = char(varargin{idx});
    opts.(name) = varargin{idx + 1};
end
end

function value = logical_scalar(x)
value = double(logical(x));
end
