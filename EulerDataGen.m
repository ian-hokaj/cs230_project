
% Gets solutions to 1D Riemann problem using exact solution of 
% Euler Equation from random initial conditions

clc; clear; close all;

%sample at grid size of 1024
x = linspace(-10,10, 1024);

% Baseline IC's from initial Problem 
% p_L = 10^5; %N/m2
% rho_L = 1; %kg/m3
% u_L = 100; %m/s
% p_R = 10^4; 
% rho_R = .125;
% u_R = 50;    


N = 1100; % num training+test samples

% set bounds for random initialization 
p_lb = 20000; p_ub = 200000;
rho_lb = .5; rho_ub = 2;
u_lb = 10; u_ub = 100;

i = 1;
% solve random IC's until we store all N samples 
while i <= N
    % set up random initial conditions    
    p_L = (p_ub-p_lb).*rand + p_lb;
    p_R = (p_ub-p_lb).*rand + p_lb;
    
    rho_L = (rho_ub-rho_lb).*rand + rho_lb;
    rho_R = (rho_ub-rho_lb).*rand + rho_lb;
    
    u_L = (u_ub-u_lb).*rand + u_lb;
    u_R = (u_ub-u_lb).*rand + u_lb;

    % Solve for the exact solution, if it exists from the IC's
    [W,exitflag] = Riem_exact(rho_R, u_R, p_R, rho_L, u_L, p_L, x, .01);
    % W==[rho;u;p] for the specified x points
    
    % only store solution and IC if solution converged
    if exitflag == 1 
        % data.a size: (N,gridsize,3) -- matrix of initial conditions
        data.a(i,:,1) = ones(length(x),1).*(rho_L.*(x<0) + rho_R.*(x>=0))';
        data.a(i,:,2) = ones(length(x),1).*(u_L.*(x<0) + u_R.*(x>=0))';
        data.a(i,:,3) = ones(length(x),1).*(p_L.*(x<0) + p_R.*(x>=0))';
        
        % data.u size: (N,gridsize,3) -- matrix of solutions
        data.u(i,:,1) = W(1,:)';
        data.u(i,:,2) = W(2,:)';
        data.u(i,:,3) = W(3,:)';
        i = i + 1;
    end


end

% data.x size: (1,gridsize)
data.x = x;

% figure(1)
% subplot(3,1,1)
% plot(x,ones(1,length(x)).*(rho_L.*(x<0) + rho_R.*(x>0)),'k--'); hold on;
% plot(x,W(1,:))
% ylabel('\rho (kg/m^3)')
% legend('Initial Condition','Solution at t=0.01 sec')
% title('Exact Riemann Solution')
% subplot(3,1,2)
% plot(x,ones(1,length(x)).*(u_L.*(x<0) + u_R.*(x>0)),'k--'); hold on;
% plot(x,W(2,:))
% ylabel('u (m/s)')
% subplot(3,1,3)
% plot(x,ones(1,length(x)).*(p_L.*(x<0) + p_R.*(x>0)),'k--'); hold on;
% plot(x,W(3,:))
% ylabel('P (N/m^3)')
% xlabel('X')



%% Functions


% computes exact riemann solution w/ nonlinear solve using self-similarity
function [W_xt,exitflag] = Riem_exact(rho1,u1,p1,rho4,u4,p4,x,t)
% t== t1 to solve for solution at
gam = 1.4;
W1 = [rho1; u1; p1];
W4 = [rho4; u4; p4];

c1 = sqrt(gam*p1/rho1);
c4 = sqrt(gam*p4/rho4);
zerofun = @(p2_p1) p4/p1 - p2_p1*(1 + (gam-1)/2/c4*(u4 - u1 - c1/gam*(p2_p1 - 1)/sqrt((gam+1)/2/gam*(p2_p1-1)+1)))^(-2*gam/(gam-1));
[p2_p1,~,exitflag,~] = fzero(zerofun,2);

p2 = p2_p1*p1;
u2 = u1 + c1/gam*(p2_p1-1)/sqrt((gam+1)/2/gam*(p2_p1-1)+1);
c2 = c1*sqrt(p2_p1*((gam+1)/(gam-1)+p2_p1)/(1+(gam+1)/(gam-1)*p2_p1));
rho2 = gam*p2/c2^2;
u3 = u2;
p3 = p2;
c3 = (gam-1)/2*(u4 - u3 + 2*c4/(gam-1));
rho3 = gam*p3/c3^2;
W2 = [rho2; u2; p2];
W3 = [rho3; u3; p3];

lam3_l = u4-c4;
lam3_r = u3 - c3;
lam2 = u2;
lam1 = u1 + c1*sqrt((gam+1)/2/gam*(p2_p1-1)+1);

W_xt = zeros(3,length(x));
for i = 1:length(x)
    x_t = x(i)/t;

    if x_t<lam3_l
        W_xt(:,i) = W4;
    elseif x_t<lam3_r && x_t>lam3_l
        u_exp = 2/(gam+1)*(x_t+(gam-1)/2*u4+c4);
        c_exp = 2/(gam+1)*(x_t +(gam-1)/2*u4+c4) - x_t;
        p_exp = p4*(c_exp/c4)^(2*gam/(gam-1));
        rho_exp = gam*p_exp/c_exp^2;
        W_xt(:,i) = [rho_exp,u_exp,p_exp];
    elseif x_t<lam2 && x_t>lam3_r
        W_xt(:,i) = W3;
    elseif x_t<lam1 && x_t> lam2
        W_xt(:,i) = W2;
    elseif x_t > lam1
        W_xt(:,i) = W1;
    end

end

end