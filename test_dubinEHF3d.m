clear
clc
close all

master_path = zeros(1000, 3, 10000);
for i = 1:10000
    % keep the init pos zero:
    x1 = 0; 
    y1 = 0;
    alt1 = 0;
    
    heading = rand_between(0, 360);
    psi1 = heading*pi/180; % psi1 = 20*pi/180; % initial heading, between [0, 2*pi]
    

    % a = -30;
    % b = 30;
    % r = a + (b-a)*rand(1);
    angle = rand_between(-30, 30);


    % change these variables to get different paths
    gamma = angle*pi/180; % gamma = -30*pi/180; % climb angle, keep in between [-30 deg, 30 deg]
    
    x2 = rand_between(-1000, 1000); % No specific ranges defined for these but figured I would cap them at something
    y2 = rand_between(-1000, 1000);
    % x2 = -100; 
    % y2 = -250;
    
    % keep these constant
    steplenght = 10; % trajectory discretization level
    r_min = 100; % vehicle turn radius.

    parameters = [x1, y1, alt1, psi1, gamma, x2, y2, steplenght, r_min];
    
    [path, psi_end, num_path_points] = dubinEHF3d(x1, y1, alt1, psi1, x2, y2, r_min, steplenght, gamma);

    psi_end = [psi_end, 0];

    % writematrix(path(1:num_path_points, :), "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path_" + num2str(i) + ".xls")
    
    % % writematrix(path, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path_" + num2str(i) + ".xls")

    % writematrix(parameters, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/parameters_" + num2str(i) + ".xls")
    master_path(:, :, i) = path;
    if i == 1
        % writematrix(path, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path_" + num2str(i) + ".xls")
        % writematrix(path, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path.csv")
        writematrix(parameters, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/parameters.xls")
        writematrix(psi_end, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/psi_end.xls")
    else
        % writematrix(path, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path.xls", 'WriteMode','append')
        % writematrix(path, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/dubin_path.csv",'WriteMode','append')
        writematrix(parameters, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/parameters.xls", 'WriteMode', 'append')
        writematrix(psi_end, "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/psi_end.xls", 'WriteMode', 'append')
    end

end
save('path.mat', 'master_path')

figure;
plot3(path(1:num_path_points,1),  path(1:num_path_points,2), path(1:num_path_points,3), 'b.-' ); 
hold on; grid on;
plot3(x1,y1,alt1, 'r*')
plot(x2, y2, 'm*')
axis equal
xlabel('x')
ylabel('y')
zlabel('alt')

function num = rand_between(low, high)
    num =  low + (high-low)*rand(1);
end
