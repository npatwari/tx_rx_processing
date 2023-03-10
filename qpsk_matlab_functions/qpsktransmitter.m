% QPSK transmitter
%
% ESE 471, Spring 2021
% Author: Neal Patwari
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Signal Generation
% INPUT:  none
% OUTPUT: binary data
%data      = [1 0 0 1];
%data = round(rand(1,49));
if 1,
    temp        = 'You Are Strong&Resilient'
    data_bits   = text2bits(temp)
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Signal Generation
    % INPUT:  binary data
    % OUTPUT: 4-ary data (0..3) values
    data   = binary2mary(data_bits, 4);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Modulation
    % INPUT: data
    % OUPUT: modulated values, x
    A = sqrt(9/2);
    inputVec   = [0   1   2   3];
    outputVecI = [A  -A   A  -A];
    outputVecQ = [A   A  -A  -A];
    xI          = lut(data, inputVec, outputVecI);
    xQ          = lut(data, inputVec, outputVecQ);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Upsample
    % INPUT: modulated values, x
    % OUTPUT: modulated values at sampling rate, x_s
    x_s_I       = oversample(xI,8);
    x_s_Q       = oversample(xQ,8);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Pulse-shape filter
    % INPUT: modulated values at sampling rate, x_s
    % OUTPUT: baseband transmit signal s
    pulse       = SRRC(0.5, 8, 6);
    s_0_I       = conv(x_s_I, pulse);
    s_0_Q       = conv(x_s_Q, pulse);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Up-convert
    % INPUT: s, baseband signal
    % OUTPUT: up_s, bandpass signal
    f_0         = 0.25;
    n           = 0:length(s_0_I)-1;
    s           = sqrt(2) .* s_0_I .* cos((2 * pi * f_0) .* n) ...
        - sqrt(2) .* s_0_Q .* sin((2 * pi * f_0) .* n);
    
    
    s = s + 0.2*randn(size(s));
    
end

% Load from one of M. Rice's mat files.
% load qpskdata_Rice.mat
% s = qpskdata(2,:);

% Plot for project handout
figure(1)
h = plot(s, '-o');
set(gca,'FontSize',20);
%set(gca,'ylim',[-1.5 1.5]);
set(h,'LineWidth',2);
xlabel('Sample')
ylabel('Value')
grid




% QPSK receiver

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Down-convert
% INPUT: up_s, bandpass signal
% OUTPUT: s, baseband signal
f_0   = 0.2499;
n         = 0:length(s)-1;
s_rx_I      = sqrt(2) .* s .* cos((2 * pi * f_0) .* n);
s_rx_Q      = -sqrt(2) .* s .* sin((2 * pi * f_0) .* n);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matched filter
% INPUT: baseband transmitted signal s
% OUTPUT: matched-filtered signal y
pulse       = SRRC(0.5, 8, 6);
yI          = filter(pulse, 1, s_rx_I);
yQ          = filter(pulse, 1, s_rx_Q);


% Plot for project handout
figure(2)
h = plot(yI, '-o');
set(gca,'FontSize',20)
set(h,'LineWidth',2);
xlabel('Sample')
ylabel('Value')
grid
figure(3)
h = plot(yQ, '-o');
set(gca,'FontSize',20)
set(h,'LineWidth',2);
xlabel('Sample')
ylabel('Value')
grid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time Synch
% Input: Matched Filter output
% OUTPUT: Synched MF output with samples at US_Rate, 2*US_Rate, ...
y_s_I       = yI(97:end);
y_s_Q       = yQ(97:end);

% Plot eye-diagram
figure(5);
hI = plot_eye_diagram(y_s_I, 8, 0);
figure(6);
hQ = plot_eye_diagram(y_s_Q, 8, 0);
%set(gca,'ylim',[-1.3, 1.3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample
% INPUT: Synched matched filter output
% OUTPUT: Symbol Samples (at n*T_sy)
US_Rate     = 8;
r_hat_I     = y_s_I(US_Rate: US_Rate: end);
r_hat_Q     = y_s_Q(US_Rate: US_Rate: end);
expected_symbols = 84;
if length(r_hat_I) > expected_symbols,
    r_hat_I = r_hat_I(1:expected_symbols);
    r_hat_Q = r_hat_Q(1:expected_symbols);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Symbol decisions
% INPUT: Symbol Samples
% OUTPUT: Bits
A = sqrt(9/2);
outputVecI = [A  -A   A  -A];
outputVecQ = [A   A  -A  -A];
r_hat       = [r_hat_I; r_hat_Q];
outputVec   = [outputVecI; outputVecQ];
symbols = size(r_hat,2);
symbols_out = zeros(1, symbols);
for i=1:symbols
   symbols_out(i) = findClosest(r_hat(:,i), outputVec);
end
%symbols_out = findClosestMary(r_hat, outputVec);

% Draw signal space constellation diagram.
figure(8)
plot(r_hat(1,:), r_hat(2,:),'.')
set(gca,'FontSize',20)
xlabel('x_0')
ylabel('x_1')
grid

% Draw signal space constellation diagram in 10 frames.
figure(9)
frames= 6;
num_pts = length(r_hat)/frames;
for k=1:frames,
    h = plot(r_hat(1,round((k-1)*num_pts)+1:round(k*num_pts)),...
        r_hat(2,round((k-1)*num_pts)+1:round(k*num_pts)),'ko');
    set(gca,'FontSize',20)
    set(h,'MarkerSize',14)
    set(h,'MarkerFaceColor','r')
    xlabel('x_0')
    ylabel('x_1')
    set(gca,'DataAspectRatio',[1 1 1])
    set(gca,'xlim',[-4 4])
    set(gca,'ylim',[-4 4])
    grid
    pause();
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binary Conversion
% INPUT: Symbol values
% OUTPUT: Bit values
data_out = mary2binary(symbols_out,4);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Translate to ascii text
% INPUT: Bits
% OUTPUT: Character vector, message_out
message_out = binvector2str(data_out);

