function result = procedure(method_id, X, params, reference)
% PROCEDURE Run a native generic procedure discovered by its manifest ID.
% No algorithm registry or numerical calculations live in this adapter.
% Result row indices retain native zero-based indexing, and masks supplied as
% typed vectors must use native row-major flattening (reshape(mask.', [], 1)).
if nargin < 3
    params = struct();
end
if nargin < 4
    reference = [];
end
result = n4m.n4m_procedure_mex(method_id, X, params, reference);
end
