function main_opal_ann_control
% MAIN_OPAL_ANN_CONTROL  –  orchestrates UDP I/O and ANN inference
%   1. Receive   vpcc / ipcc from OPAL-RT
%   2. Predict   K_new  (hoặc [Lg,Rg]) bằng ANN trên host-PC
%   3. Transmit  K_new   trở lại OPAL-RT
%
%   Ctrl-C hoặc nhấn nút Stop để dừng an toàn.
%
%   Quang Mạnh Hoàng – 2025-06-19
% ===============================================================
clear all;
clc
%% ========= 1. CONFIGURATION ===================================
hostIP        = "192.168.5.15";   % địa chỉ máy tính Host (PC)
recvPort      = 5556;             % cổng nhận dữ liệu từ OPAL-RT
sendIP        = "192.168.5.89";   % địa chỉ target OPAL-RT
sendPort      = 6000;             % cổng target nhận controller gain
timeout_s     = 1;                % timeout UDP (s)
showEveryN    = 50;               % in log mỗi N gói
%% ========= 2. INITIALISE ======================================
udpObj = udpport("IPV4", ...
                 "LocalHost", hostIP, ...
                 "LocalPort", recvPort, ...
                 "ByteOrder","little-endian", ...
                 "Timeout",  timeout_s);

% % đóng socket an toàn khi hàm thoát
% c = onCleanup(@()clear udpObj);
% Load ANN (model đã lưu kèm mapminmax trong net)
annFile = "ann_model_paramsVI_Realtime.mat";   % <- dùng đúng file bạn save ở phần train
S       = load(annFile, "net");
net     = S.net;

% Dùng để lưu toàn bộ Lg và Rg
Lg_hist = [];
Rg_hist = [];

fprintf("[MAIN] UDP ready  (%s:%d)\n", hostIP, recvPort);


%% ========= 3. MAIN LOOP ======================================
msgID      = uint32(0);
pktCounter = 0;
t0         = tic;
% sau load ANN
receivedData = [];   % mỗi hàng: [vpcc; ipcc]′ = 1×200
K_new11 = [];
% --- các khai báo khác ---
initialK = -[339.437515294446	-37.1962550049616	-155.433559437714	37.2641713114898	3289.76434330323;
287.980448278270	35.5843781184771	-282.198010268474	58.4116660247888	-4058.33626972146];      % <-- giá trị khởi tạo tuỳ ý
K_current = [initialK(1,1) initialK(2,1) initialK(1,2) initialK(2,2) initialK(1,3) initialK(2,3) initialK(1,4) initialK(2,4) initialK(1, 5) initialK(2,5)];       % giá trị sẽ thực sự được gửi
lastSend  = tic;            % bộ đếm thời gian lần gửi gần nhất
t_start  = tic;             % mốc bắt đầu đếm 5 s
send_packet(udpObj, K_current, msgID, sendIP, sendPort);
try
    while 1
        % 3.1  –  polling: có khung dữ liệu?
        if udpObj.NumBytesAvailable < 8           % chưa đủ header
            pause(0.001);   continue
        end
        
        % ---- 3.2  Đọc 1 gói hoàn chỉnh --------------------------
        % ---- 3.2 Đọc 1 gói hoàn chỉnh --------------------------
        [vpcc, ipcc, ok] = read_packet(udpObj);
        if ~ok
            pause(0); continue
        end
    
        % ---- MỚI: gom dữ liệu vào receivedData ------------------
        dataRow = [vpcc(:);  ipcc(:)]; % 
        pktCounter = pktCounter + 1;      % tăng đếm gói
        if isempty(receivedData)
            receivedData = dataRow;         % khởi tạo
        else
            receivedData = [receivedData; dataRow];  % thêm hàng mới
        end
        % Đẩy receivedData ra Workspace để kiểm tra
        assignin('base','receivedData', receivedData);
        % ---- 3.3  ANN inference --------------------------------
        [Lg, Rg] = ANN_GIE(vpcc, ipcc, net);
        % Lưu vào history
        Lg_hist(end+1,1) = Lg;
        Rg_hist(end+1,1) = Rg;
        
        % Đẩy ra Workspace để debug/plot
        assignin('base','Lg_hist',Lg_hist);
        assignin('base','Rg_hist',Rg_hist);

        K_new = compute_gain_Lg_Rg((Lg-4.08)*1e-3, Rg-1);
        K_new_send = -[K_new(1,1) K_new(2,1) K_new(1,2) K_new(2,2) K_new(1,3) K_new(2,3) K_new(1,4) K_new(2,4) K_new(1,5) K_new(2,5)];
        if toc(lastSend) >= 1
            msgID = msgID + 1;
    
            if toc(t_start) <= 3
                % trong 5 s đầu, gửi giá trị khởi tạo
                K_to_send = K_current;
            else
                % sau 5 s, gửi gain mới từ ANN (có dấu âm nếu cần)
                K_to_send = K_new_send;
            end
    
            send_packet(udpObj, K_to_send, msgID, sendIP, sendPort);
            lastSend = tic;    % reset bộ đếm 0.5 s
        end
        % 
        % ---- 3.5  Log nhẹ nhàng --------------------------------
        pktCounter = pktCounter + 1;
        if mod(pktCounter, showEveryN)==0
            fprintf("[MAIN] %6d pkts |  K11 = %.4f\n", ...
                    pktCounter, K_to_send(1,1));     % in 1 phần tử đại diện
        end
    end
catch ME
    warning("Runtime error:\n%s", getReport(ME,'extended'));
end

fprintf("[MAIN] Stopped after %.1f s – %d packets processed.\n", ...
        toc(t0), pktCounter);
end
%% =============================================================
%% ==================== SUBFUNCTIONS ============================

function [vpcc, ipcc, ok] = read_packet(u)
% READ_PACKET – đọc 1 UDP frame từ OPAL-RT, tách 100 vpcc + 100 ipcc
%   Header: [dev_id(2) msg_id(4) msg_len(2)]
%   Payload: msg_len byte = 200×8 = 1600 byte
%   vpcc, ipcc: 100×1 (rỗng nếu lỗi)
%   ok: true nếu gói hợp lệ

    ok = false;
    vpcc = [];
    ipcc = [];

    % 1) Polling: có data chưa?
    if u.NumBytesAvailable <= 0
        return
    end

    % 2) Đọc dữ liệu raw
    rawBytes = read(u, u.NumBytesAvailable, 'uint8');
    
    % 3) Header ít nhất 8 byte?
    if numel(rawBytes) < 8
        disp('read_packet: Received packet too small, skipping...');
        return
    end

    % 4) Parse header
    %    dev_id = typecast(uint8(rawBytes(1:2)), 'int16');    % nếu cần
    %    msg_id = typecast(uint8(rawBytes(3:6)), 'int32');    % nếu cần
    msg_len = typecast(uint8(rawBytes(7:8)), 'int16');      % bytes of payload

    % 5) Kiểm tra toàn gói
    expectedTotal = 8 + double(msg_len);
    if numel(rawBytes) < expectedTotal
        disp(['read_packet: Incomplete packet (only ' num2str(numel(rawBytes)-8) ...
              ' of ' num2str(msg_len) ' payload bytes)']);
        return
    end
    if msg_len ~= 1600
        disp(['read_packet: msg_len=' num2str(msg_len) ' ≠ 1600 expected']);
        return
    end

    % 6) Quá trình extract payload thành double
    dataBytes = rawBytes(9:8+msg_len);
    data = typecast(uint8(dataBytes), 'double');  % should be 200×1

    if numel(data) < 200
        disp(['read_packet: Only ' num2str(numel(data)) ' doubles extracted, skipping']);
        return
    end

    % 7) Tách 100 vpcc / 100 ipcc
    vpcc = data(1:100);
    ipcc = data(101:200);

    ok = true;
end






% --------------------------------------------------------------
function [Lg, Rg] = ANN_GIE(vpcc, ipcc, net)
% ANN_GIE – Gom feature và để net tự áp dụng mapminmax nội bộ
%   vpcc, ipcc: 100×1 mỗi vector
%   net: feedforwardnet có inputs{1}.processFcns={'mapminmax'} và processSettings
%
%   Output: Lg, Rg ở thang nguyên gốc (net tự undo mapminmax cho output)

    % 1) Gom feature đúng thứ tự như khi train: [vpcc(1..100), ipcc(1..100)]
    feat = [vpcc(:); ipcc(:)];   % 200×1 (cột)

    % 2) Kiểm tra kích thước kỳ vọng của net để tránh nhầm thứ tự/độ dài
    nin = net.inputs{1}.size;
    if size(feat,1) ~= nin
        error("Feature length mismatch: expected %d, got %d", nin, size(feat,1));
    end

    % 3) Suy luận: net sẽ tự mapminmax(input) và tự undo mapminmax(output)
    Y  = net(feat);     % trả về 2×1
    Lg = Y(1);
    Rg = Y(2);
end

% --------------------------------------------------------------
function send_packet(u, K, msgID, ip, port)
    devID  = uint16(1);
    Kvec   = reshape(K.',1,[]);          % 1×10
    msgLen = uint16( numel(Kvec)*8 );
    payload = typecast(double(Kvec),'uint8');
    hdr    = [typecast(devID,'uint8'), ...
              typecast(uint32(msgID),'uint8'), ...
              typecast(msgLen,'uint8')];
    write(u, [hdr payload], ip, port);
end

%% --- Local function: tính K bằng pole-placement ---
function [K] = compute_gain_Lg_Rg(Lg, Rg)
% Unpack
f0   = 50;
Pref = 4.5e3;
Qref = 1.5e3;
Vn   = 110;
Vj   = 110;

% Tính Xg
Xg = 2*pi*f0*Lg;

% Giải E_ss và delta_ss
[Ess, delta_ss] = solve_E_delta(Pref, Qref, Vn, Rg, Lg, f0);
Vi0    = Ess;
theta0 = delta_ss;

% Tính KPQ
D = Rg^2 + Xg^2;
K11 = (3/D)*( Rg*Vi0*Vj*sin(theta0) + Xg*Vi0*Vj*cos(theta0) );
K12 = (3/D)*( Rg*(2*Vi0 - Vj*cos(theta0)) + Xg*Vj*sin(theta0) );
K21 = (3/D)*( Xg*Vi0*Vj*sin(theta0) - Rg*Vi0*Vj*cos(theta0) );
K22 = (3/D)*( Xg*(2*Vi0 - Vj*cos(theta0)) - Rg*Vj*sin(theta0) );
KPQ = [K11 K12; K21 K22];
% 3a) Conventional P-loop (fixed gains from your design)
Kp = 0.006383815167581;
Dp = 2.506338228783948e3;

% 3b) Conventional Q-loop (fixed gains)
Kq = 0.076670166944346;
Dq = 1.033099625954584;

% Ma trận hệ thống A, B cho pole-placement
A = [0 0 Dp*Kp          Kp*KPQ(1,2)      Kp*KPQ(1,1);
    0 0 0             Dq*Kq+Kq*KPQ(2,2) Kq*KPQ(2,1);
    0 0 0             0                0;
    0 0 0             0                0;
    0 0 1             0                0];
B = [1 0; 0 1; 1 0; 0 1; 0 0];

% Kiểm tra controllability
if rank(ctrb(A,B)) < size(A,1)
    warning("System not controllable: Lg=%.3e, Rg=%.3e", Lg, Rg);
    K = zeros(2,5);
    return;
end
% Đặc tả cực đặt cho pole-placement
zeta      = 1;
T_setl    = 1;
omega_n   = 4/(zeta*T_setl);
p1        = -zeta*omega_n + 1i*omega_n*sqrt(1-zeta^2);
poles     = [p1, conj(p1), -80, -90, -100];
% Tính K (2×5)
K = place(A, B, poles);
end

%% --- Local function: giải E_ss, delta_ss bằng fsolve ---
function [Ess, delta_ss] = solve_E_delta(Pref, Qref, Vn, Rg, Lg, f)
Xg = 2*pi*f*Lg;
D  = Rg^2 + Xg^2;
k  = 3 / D;
fun = @(x)[ ...
    Pref - k*x(1)*( Rg*(x(1) - Vn*cos(x(2))) + Vn*Xg*sin(x(2)) );
    Qref - k*x(1)*( Xg*(x(1) - Vn*cos(x(2))) - Vn*Rg*sin(x(2)) ) ];

% Giá trị đoán ban đầu
Pslope = Pref*D / (3*Vn^2);
delta0 = asin( max(min(Pslope,0.99),-0.99) );
x0     = [Vn, delta0];

opts = optimoptions('fsolve','Display','off', ...
    'FunctionTolerance',1e-12,'StepTolerance',1e-12);
sol  = fsolve(fun, x0, opts);
Ess      = sol(1);
delta_ss = sol(2);
end