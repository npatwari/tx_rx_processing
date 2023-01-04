
n_samples = 10400;
out_folder = '../signal_library/';

for M = [4]
    save_signal("OFDM", M, n_samples, out_folder, true);
end

%{
filter.oss = 1;
filter.filter_span = 10;

for rf = [0.2, 0.5]
    filter.roll_off = rf;
    
    for M = [2, 4, 8]
        save_signal('PSK', M, n_samples, out_folder, true, filter);
    end
    
    for M = [16, 64, 256]
        save_signal('QAM', M, n_samples, out_folder, true, filter);
    end
    
end
%}
function save_signal(mod, M, n_samples, out_folder, scale, filter)

    if nargin < 6
        apply_filter = false;
    else
        apply_filter = true;
    end
    mod
    in = randi([0, 1], n_samples, 1);
    if mod == "PSK"
        waveform = gen_psk_signal(M, in);
    elseif mod == "QAM"
        waveform = gen_psk_signal(M, in);
    elseif mod == "OFDM"
        in = randi([0 1], n_samples, 1);
        waveform = gen_ofdm_signal(M, in);
        apply_filter = false;
    else
        disp('Unsupported modulation');
        return
    end
    
    name = strcat(mod, num2str(M));
    
    if apply_filter
        rcFilter = comm.RaisedCosineTransmitFilter('Shape', 'Square root', ...
        'RolloffFactor', filter.roll_off, ...
        'OutputSamplesPerSymbol', filter.oss, ...
        'FilterSpanInSymbols', filter.filter_span);
        %https://www.mathworks.com/help/comm/ref/raisedcosinetransmitfilter.html
        waveform = rcFilter(waveform);
        name = name + "_" + num2str(filter.roll_off)
        name = name + "_" + num2str(filter.oss);
        name = name + "_" + num2str(filter.filter_span);
    end
    
    out_folder + name + '.iq';
    
    if scale
        waveform(1)
        waveform = 0.6 * waveform;
        waveform(1)
        write_complex_binary(waveform, out_folder + name + '_scaled.iq');
    else
        write_complex_binary(waveform, out_folder + name + '.iq');
    end
    size(waveform)
        
        

end

function waveform = gen_psk_signal(M, in) 
    % M = as a power-of-two scalar integer
    
    usedInput = in(1:log2(M)*(floor(length(in)/log2(M))));
    tmp = reshape(usedInput, log2(M), length(usedInput)/log2(M));
    symbolInput = bi2de(tmp', 'left-msb');
    
    phaseOffset = pi/M;
    waveform = pskmod(symbolInput, M, phaseOffset, 'bin');
end


function waveform = gen_qam_signal(M, in)
    % M = as a power-of-two scalar integer
    
    waveform = qammod(in, M, 'bin', 'InputType', 'bit', 'UnitAveragePower', true);
end


function waveform = gen_ofdm_signal(M, in)
    % M = as a power-of-two scalar integer
    
    ofdmMod = comm.OFDMModulator('FFTLength', 64, ...
    'NumGuardBandCarriers', [6;5], ...
    'InsertDCNull', true, ...
    'CyclicPrefixLength', [16], ...
    'Windowing', false, ...
    'NumSymbols', 100, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', false);

    usedInput = in(1:log2(M)*(floor(length(in)/log2(M))));
    tmp = reshape(usedInput, log2(M), length(usedInput)/log2(M));
    symbolInput = bi2de(tmp', 'left-msb');

    phaseOffset = pi/M;
    dataInput = pskmod(symbolInput, M, phaseOffset, 'gray');
    ofdmInfo = info(ofdmMod);
    ofdmSize = ofdmInfo.DataInputSize;
    dataInput = reshape(dataInput, ofdmSize);

    waveform = ofdmMod(dataInput);
end