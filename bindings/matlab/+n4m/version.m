function v = version()
% n4m.version  Return the libn4m runtime version string.
%   "1.0.3+abi.2.0.0" for the snapshot that shipped this binding.
%
% The string is read by the n4m_version_mex shim which calls
% n4m_get_version_string() on the loaded libn4m.

v = n4m.n4m_version_mex();
end
