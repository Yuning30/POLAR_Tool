%nn_rl;
Ts = 1;  % Sample Time
N = 3;    % Prediction horizon
Duration = 500; % Simulation horizon

pos_radius = 100;
ang_radius = 360;
rejoin_radius = 500;
rejoin_angle = 60; 
%degree

global simulation_result;
global disturb_range;

disturb_range = 0.1; % Disturbance range

formatSpec = '%f %f %f\n';

fileID = fopen('nn_rl_simulation.txt','w');




for m=1:1


    
x12 = ang_radius * rand(1);
x13 = ang_radius * rand(1);
x1 = 7500 + pos_radius*rand(1);
x2 = 7500 + pos_radius*rand(1);
x0 = sqrt(x1*x1 + x2*x2);
x4 = x1 + rejoin_radius* cosd(rejoin_angle + 180 + x12);
x5 = x2 + rejoin_radius* sind(rejoin_angle + 180 + x12);
x3 = sqrt(x4*x4 + x5*x5);
x6 = (300 + pos_radius*rand(1));
x7 = x6 * cosd(x12);
x8 = x6 * sind(x12);
x9 = (280 + pos_radius*rand(1));
x10 = x9 * cosd(x13);
x11 = x9 * sind(x13);

 

x = [x0;x1;x2;x3;x4;x5;x6;x7;x8;x9;x10;x11;x12;x13;];

simulation_result = x;

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,4);

u_max = 0;

% Apply the control input constraints
LB = -3*ones(N,4);
UB = 3*ones(N,4);

x_now = zeros(14,1);
x_next = zeros(14,1);
z = zeros(18,1);

x_now = x;

for ct = 1:(Duration/Ts)
    x_input = x_now(1:12, 1);
    x_input([1, 4]) = x_input([1, 4]) / 1000.0;
    x_input([7, 10]) = x_input([7, 10]) / 400.0;

    u = NN_output_rl(x_input,0,1,'rl_tanh256x256_mat');
    
    x_next = system_eq_dis(x_now, Ts, u);

    x = x_next;
    x_now = x_next;
end

plot(simulation_result(2,:),simulation_result(3,:), 'color', [223/255, 67/255, 69/255]);
% title('RL Rejoin Default', 'FontSize', 14)
% xlabel('x1', 'FontSize', 14);
% ylabel('x2', 'FontSize', 14);
set(gca,'FontSize',16)
hold on;

fprintf(fileID, formatSpec, simulation_result(1:3,:));

end
fclose(fileID);

% plot( [0.1, 0.4, 0.4, 0.1, 0.1], [0.15,0.15,0.45,0.45,0.15], 'color' , [72/255 130/255 197/255], 'LineWidth', 2.0);
% hold on;
% 
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% print(fig,'../rl/simulation','-dpdf')
% export_fig ../rl/simulation.pdf