classdef SpectralEncoder < handle
    % Native fitted encoder. Construct through n4m.LVSE or n4m.GCU.
    properties (Access = private)
        Parameters
    end
    properties (Access = private, Transient)
        NativeHandle = []
    end
    methods
        function obj = SpectralEncoder(parameters)
            obj.Parameters = double(parameters);
        end
        function obj = fit(obj, X)
            next = n4m.n4m_spectral_mex('fit', double(X), obj.Parameters);
            if ~isempty(obj.NativeHandle)
                n4m.n4m_spectral_mex('destroy', obj.NativeHandle);
            end
            obj.NativeHandle = next;
        end
        function Z = transform(obj, X)
            if isempty(obj.NativeHandle), error('n4m:spectral', 'Encoder is not fitted'); end
            Z = n4m.n4m_spectral_mex('transform', obj.NativeHandle, double(X));
        end
        function Z = fit_transform(obj, X)
            obj.fit(X);
            Z = obj.transform(X);
        end
        function [A, offset] = export_linear_operator(obj)
            if isempty(obj.NativeHandle), error('n4m:spectral', 'Encoder is not fitted'); end
            result = n4m.n4m_spectral_mex('affine', obj.NativeHandle);
            A = result.operator;
            offset = result.offset;
        end
        function delete(obj)
            if ~isempty(obj.NativeHandle)
                n4m.n4m_spectral_mex('destroy', obj.NativeHandle);
                obj.NativeHandle = [];
            end
        end
    end
end
