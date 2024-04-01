function X = convNDfft(X,cols,sub_inds,ver)
    Ns = size(X);
    dim = length(Ns);
    for k=1:dim
        if ver==1
            col = cols{k}(:);
            n = length(col);
            col_fft = fft([zeros(Ns(k)-n,1);col]);
        else
            col_fft = cols{k}(:);
        end
        
        if ~isempty(col_fft)
            if dim ==1
                shift = [1 2];
                shift_back = shift;
            else
                shift = circshift(1:dim,1-k);
                shift_back=circshift(1:dim,k-1);
            end
        
            X = ifft(col_fft.*fft(permute(X,shift)));
            inds = cell(dim,1);
            inds{1} = sub_inds{k}; 
            inds(2:dim) = repmat({':'},dim-1,1);
            X = X(inds{:});
            X = permute(X,shift_back);
        end
                
    end
    X = real(X);
end
