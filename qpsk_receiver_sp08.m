% QPSK transmitter
%
% ECE 5520, Spring 2008
% Author: Neal Patwari
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Signal Generation
% INPUT:  none
% OUTPUT: binary data
%data      = [1 0 0 1];
%data = round(rand(1,49));
preamble    = repmat([1 1 0 0], 1, 16);
sync        = [1 1 1 0 1 0 1 1 1 0 0 1 0 0 0 0];
if 0,
    temp        = 'I worked all semester on digital communications and all I got was this sequence of ones and zeros.';
    data_bits   = [preamble, sync, text2bits(temp)];
    
    
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
    s_0_I       = [conv(x_s_I, pulse) zeros(1,16)];
    s_0_Q       = [conv(x_s_Q, pulse) zeros(1,16)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Add fractional delay
    
    temp1 = fractionalDelay(s_0_I, -0.5);
    temp2 = fractionalDelay(s_0_Q, -0.5);
    
    figure(948)
    plot(1:length(s_0_I), temp1, 1:length(s_0_I), s_0_I)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Up-convert
    % INPUT: s, baseband signal
    % OUTPUT: up_s, bandpass signal
    f_0         = 0.25;
    n           = 0:length(s_0_I)-1;
    s           = sqrt(2) .* s_0_I .* cos((2 * pi * f_0) .* n) ...
        - sqrt(2) .* s_0_Q .* sin((2 * pi * f_0) .* n);
    
    
    s = s + 0.15*randn(size(s));
 
     save qpsk_timesync.mat s
     
else
    load qpsk_timesync.mat;  % s
end

%

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





% BPSK receiver

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Down-convert
% INPUT: up_s, bandpass signal
% OUTPUT: s, baseband signal
f_0   = 0.25;
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Down-sampling
% INPUT: 
% OUTPUT: 
dsI = yI(3:4:end);
dsQ = yQ(3:4:end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% System Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 2;             %number of samples/symbol
alpha = 0.5;       %excess bandwidth of SRRC pulse shape
Lp = 6;            %SRRC pulse shape parameter
%Ndata = 2000;      %number of QPSK data symbols

SNR = 20;          %Eb/N0 in dB (SNR >= 100 for no noise)
delay = 0.15;      %timing offset (relative to a symbol time)
BnTs = 0.05;      %normalized loop bandwidth cycles/symbol
zeta = 1;          %damping constant (I use a second-order loop)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the loop filter coefficients K1t and K2t
% INPUT:  BnTs, N, zeta
% OUTPUT: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Kp = 2*2.7;
K0 = -1;

th = BnTs/(zeta+0.25/zeta); 
th = th/N;
K1 = 4*zeta*th/(1 + 2*zeta*th + th*th);
K2 = K1*th/zeta;

K1t = K1/(K0*Kp);
K2t = K2/(K0*Kp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interpolation
% INPUT: 
% OUTPUT: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% do the loop!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TEDout = zeros(size(dsI));
strobeSave = zeros(size(dsI));
x_2_I = zeros(size(dsI));
x_2_Q = zeros(size(dsI));
v = zeros(size(dsI));
k = 1;
strobe = 0;
NCO_next = 0;
mu_next = 0;

for n=4:length(dsI),
    NCO = NCO_next;
    mu  = mu_next;    
    muSave(n) = mu;
    mu2 = mu*mu;
    
    % Quadratic (2nd order) Farrow filter
    %     alpha = 0.5;
    %     h_I  = [ alpha*mu^2 - alpha*mu,  ...
    %             -alpha*mu^2 + (1+alpha)*mu,  ...
    %             -alpha*mu^2 + (alpha-1)*mu + 1, ...
    %              alpha*mu^2 - alpha*mu];
    % Cubic (3nd order) Farrow filter
    h_I  = [ (1/6)*mu^3            - (1/6)*mu,  ...
            -0.5*  mu^3 + 0.5*mu^2 +       mu,  ...
            0.5*  mu^3 -     mu^2 -  0.5* mu    + 1,  ...
            -(1/6)*mu^3 + 0.5*mu^2 - (1/3)*mu];
    
    x_2_I(n) = h_I(1)*dsI(n) + h_I(2)*dsI(n-1) + h_I(3)*dsI(n-2) + h_I(4)*dsI(n-3);
    x_2_Q(n) = h_I(1)*dsQ(n) + h_I(2)*dsQ(n-1) + h_I(3)*dsQ(n-2) + h_I(4)*dsQ(n-3);
    
    if strobe == 1 
        % Timing Error Detector
        TEDout(n) = x_2_I(n-1)*(sign(x_2_I(n-2)) - sign(x_2_I(n))) ...
            + x_2_Q(n-1)*(sign(x_2_Q(n-2)) - sign(x_2_Q(n)));
    else
        TEDout(n) = 0;
    end
    
    % Loop Filter
    v(n) = v(n-1) + (K1t + K2t)*TEDout(n) - K1t*TEDout(n-1);
    %    gamma = 0.9985;
    %    v(n) =  v(n-1) + (1-gamma) * TEDout(n);
    
    if strobe == 1
        mmuu(k) = mu;
        r_hat_I(k) = x_2_I(n);
        r_hat_Q(k) = x_2_Q(n);
        k = k+1;
    end
    
    % Mod 1 counter, delay,
    W = 1/N + v(n);
    NCO_next = NCO - W;
    if NCO_next < 0
        NCO_next = 1 + NCO_next;
        strobe = 1;
        mu_next = NCO/W;
    else
        strobe = 0;
        mu_next = mu;
    end
    strobeSave(n) = strobe;
    
    
end

figure(11);
subplot(3,1,1)
plot(v)
subplot(3,1,2)
plot(muSave)
subplot(3,1,3)
plot(TEDout)

pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Symbol decisions
% INPUT: Symbol Samples
% OUTPUT: Bits
A = sqrt(9/2);
outputVecI = [A  -A   A  -A];
outputVecQ = [A   A  -A  -A];
r_hat       = [r_hat_I; r_hat_Q];
outputVec   = [outputVecI; outputVecQ];
symbols_out = findClosestMary(r_hat, outputVec);

% Draw signal space constellation diagram.
figure(8)
plot(r_hat(1,:), r_hat(2,:),'.')
set(gca,'FontSize',20)
xlabel('x_0')
ylabel('x_1')
grid


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binary Conversion
% INPUT: Symbol values
% OUTPUT: Bit values
data_out = mary2binary(symbols_out,4)

syncLen = length(sync);
for i=1:length(data_out)-syncLen+1,
    agreement = sum(data_out(i:i+syncLen-1) == sync);
    if agreement >= 0.9 * syncLen,
        break;
    end
end
databits_expected = 686;
packetdatabits = data_out((i + syncLen):(i + syncLen+databits_expected-1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Translate to ascii text
% INPUT: Bits
% OUTPUT: Character vector, message_out
message_out = binvector2str(packetdatabits)

