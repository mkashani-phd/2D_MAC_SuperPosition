clear;
clc; 
clear fig;
close all




%gamma
function gamma = gamma(R, Eb, N0)
    gamma = R*Eb/N0;
end

function gamma = gamma_Super_tag(R, Eb, N0, alpha, P)
    gamma = R*Eb./((1-alpha)*P + N0);
end

function gamma = gamma_Super_msg(R, Eb, N0, alpha, P)
    gamma = R*Eb./(alpha*P + N0);
end


% Eb tag
function Energy_tag = Et(P,R)
    Energy_tag = P/R;
end

function Energy_super_tag = Et_Super(P,R, alpha)
    Energy_super_tag = (1-alpha)*P/R;
end



% Eb message
function Energy_msg_trad = Em_trad(t, m, R, P)
    Energy_msg_trad = P/R*(1+t/m);
end

function Energy_msg_1D = Em_1D(t, m, R, P, nr)
    Energy_msg_1D = P/R*(1+t/(nr*m));
end


function Energy_msg_2D = Em_2D(t, m, R, P, nr, nc)
    Energy_msg_2D = P/R*(1+(nr+nc)*t/(nr*nc*m));
end

function Energy_msg_2D_super = Em_2D_super(t, m, R, P, nr)
    Energy_msg_2D_super = P/R*(1+t/(nr*m));
end


% lemma 1

function Pe = error_rate(length, R, gamma)
    p = qfunc(sqrt(2*gamma));
    C = 1+p.*log2(p) + (1-p).*log2(1-p);
    Pe = qfunc(sqrt(length ./ (p .* (1 - p))) .* ((C - R) / log2((1 - p) ./ p)));
end


% Pe
function [Et_temp, Pe_t] = Pe_t_trad(t, P, R ,N0)
    Et_temp = Et(P,R);
    gam = gamma(R, Et_temp, N0);
    Pe_t = error_rate(t, R, gam);
end
function [Em_temp, Pe_m] = Pe_m_trad(t, m, R, P, N0)
    Em_temp = Em_trad(t, m, R, P);
    gam = gamma(R, Em_temp, N0);
    Pe_m = error_rate(m, R, gamma(R, gam, N0));
end

function [Et_temp, Pe_t_1D] = Pe_t_1D(t, P, R ,N0)
    Et_temp = Et(P,R);
    gam = gamma(R, Et_temp, N0);
    Pe_t_1D = error_rate(t, R, gam);
end
function [Em_temp, Pe_m_1D] = Pe_m_1D(t, m, R, P, N0, nr)
    Em_temp = Em_1D(t, m, R, P, nr);
    gam = gamma(R, Em_temp, N0);
    Pe_m_1D = error_rate(m, R, gam);
end


function [Et_temp, Pe_t_2D] = Pe_t_2D(t, P, R ,N0)
    Et_temp = Et(P,R);
    gam = gamma(R, Et_temp, N0);
    Pe_t_2D = error_rate(t, R, gam);
end
function [Em_temp, Pe_m_2D] = Pe_m_2D(t, m, R, P, N0, nr, nc)
    Em_temp = Em_2D(t, m, R, P, nr, nc);
    gam = gamma(R, Em_temp, N0);
    Pe_m_2D = error_rate(m, R, gam);
end


function [Et_temp, Pe_t_2D_super] = Pe_t_2D_super(t, P, R, N0, alpha)
    Et_temp = Et_Super(P,R, alpha);
    gam = gamma_Super_tag(R, Et_temp, N0, alpha, P);
    Pe_t_2D_super = error_rate(t, R, gam);
end


function [Em_temp, Pe_m_2D_super] = Pe_m_2D_super(t, m, R, P, N0, nr, alpha)
    Em_temp =  Em_2D_super(t, m, R, P, nr);
    gam = gamma_Super_msg(R, Em_temp, N0, alpha, P);
    Pe_m_2D_super = error_rate(m, R, gam);
end





% AUX AER for 2D
function pr_2D = pr_2D(Pem, Pet, nr, nc)
    pr_2D =  ((1-(1-Pem).^(nc-1)) .* (1-Pet)).*((1-(1-Pem).^(nr-1)) .* (1-Pet)) ;
end

function pr_2D_super = pr_2D_super(Pem, Pet, nr, nc, Pet_reg)
    pr_2D_super =  ((1-(1-Pem).^(nc-1)) .* (1-Pet)).*((1-(1-Pem).^(nr-1)) .* (1-Pet_reg)) ;
end


% AER 
function AER_trad = AER_trad (Pem, Pet)
    AER_trad = 1- ((1-Pem).*(1-Pet));
end

function AER_1D = AER_1D (Pem, Pet, nr)
    AER_1D = 1- (((1-Pem).^nr).*(1-Pet));
end

function AER_2D = AER_2D (Pem, Pet, nr, nc)
    AER_2D = pr_2D(Pem, Pet, nr, nc).*(1-Pem) + Pem;
end

function AER_2D_super = AER_2D_super (Pem, Pet, nr, nc, Pet_reg)
    AER_2D_super = pr_2D_super(Pem, Pet, nr, nc, Pet_reg).*(1-Pem) + Pem;
end


% AT 
function AT_trad = AT_trad (Pem, Pet, m, t)
    AT_trad = (m/(m+t)) * ((1-Pet).*(1-Pem));
end

function AT_1D = AT_1D (Pem, Pet, m, t, nr)
    AT_1D = (nr*m/(nr*m+t)) * ((1-Pet).*((1-Pem).^nr));
end

function AT_2D = AT_2D (Pem, Pet, m, t, nr, nc)
    AT_2D = (nr*nc*m/(nr*nc*m + nc*t)) * ((1-pr_2D(Pem, Pet, nr, nc)).*(1-Pem));
end

function AT_2D_super = AT_2D_super (Pem, Pet, m, t, nr, nc, Pet_reg)
    AT_2D_super = (nr*nc*m/(nr*nc*m + nc*t)) * ((1-pr_2D_super(Pem, Pet, nr, nc, Pet_reg)).*(1-Pem));
end



R = 4/5;
m = 256/R;
t = 256/R;

P = .1:.1:10;
N0 = 1;

nr = 5; % number of rows 
nc = 5; % number of columns

alpha = 0.13; 


%%%% trad %%%%
[Ebm_trad, PM_trad] = Pe_m_trad(t, m, R, P, N0);
[Ebt_trad,PT_trad] = Pe_t_trad(t, P, R ,N0);
error_rate_trad = AER_trad (PM_trad, PT_trad);
Throu_trad = AT_trad (PM_trad, PT_trad, m, t);

%%%% 1D %%%%
[Ebm_1D,PM_1D] =   Pe_m_1D(t, m, R, P, N0, nr);
[Ebt_1D,Pt_1D] =   Pe_t_1D(t, P, R ,N0);
error_rate_1D = AER_1D (PM_1D, Pt_1D, nr);
Throu_1D = AT_1D (PM_1D, Pt_1D, m, t, nr);


%%%% 2D %%%%%
[Ebm_2D,PM_2D] =   Pe_m_2D(t, m, R, P, N0, nr, nc);
[Ebt_2D,Pt_2D] =   Pe_t_2D(t, P, R ,N0);
error_rate_2D = AER_2D (PM_2D, Pt_2D, nr, nc);
Throu_2D = AT_2D (PM_2D, Pt_2D, m, t, nr, nc);


%%%% 2D_super %%%%%
[Ebt_2D_reg,PM_2D_reg] = Pe_t_2D(t, P, R ,N0);
[Ebm_2D_super,PM_2D_super] =  Pe_m_2D_super(t, m, R, P, N0, nr, alpha);
[Ebt_2D_super,Pt_2D_super] =  Pe_t_2D_super(t, P, R, N0, alpha);
error_rate_2D_super = AER_2D_super (PM_2D_super, Pt_2D_super, nr, nc, PM_2D_reg);
Throu_2D_super = AT_2D_super (PM_2D_super, Pt_2D_super, m, t, nr, nc, PM_2D_reg);


 
% Define colors
bgColor = [240, 240, 240] / 255; % Light gray for plot area
borderColor = [1, 1, 1];         % White for figure border

% New color palette for lines
lineColors = [
    0, 0.4470, 0.7410;   % blue
    0.8500, 0.3250, 0.0980; % orange
    0.9290, 0.6940, 0.1250; % yellow
    0.4940, 0.1840, 0.5560; % purple
];


% Plot for AER vs SNR

figure


semilogy(10*log10(R*Ebm_trad/N0), error_rate_trad, 'Color', lineColors(1, :), 'LineWidth', 5.5); hold on;
semilogy(10*log10(R*Ebm_1D/N0), error_rate_1D, 'Color', lineColors(2, :), 'LineWidth', 5.5);
semilogy(10*log10(R*Ebm_2D/N0), error_rate_2D, 'Color', lineColors(3, :), 'LineWidth', 5.5);
semilogy(10*log10(R*Ebm_2D_super/N0), error_rate_2D_super, 'Color', lineColors(4, :), 'LineWidth', 5.5);
hold off;
ylim([1e-10 1])
legend("Trad.", "1D MAC", "2D MAC", "2D MAC with SC", 'FontSize', 28, 'Location', 'northeast');
ylabel("AER", 'FontSize', 33, 'FontWeight', 'bold');
xlabel("Eb/N0 (dB)", 'FontSize', 33, 'FontWeight', 'bold');
% title("AER vs SNR", 'FontSize', 20, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 33, 'LineWidth',1.5, 'XColor', 'k', 'YColor', 'k');

% Plot for Auth Throughput vs SNR
figure


plot(10*log10(R*Ebm_trad/N0), Throu_trad, 'Color', lineColors(1, :), 'LineWidth', 5.5); hold on;
plot(10*log10(R*Ebm_1D/N0), Throu_1D, 'Color', lineColors(2, :), 'LineWidth', 5.5);
plot(10*log10(R*Ebm_2D/N0), Throu_2D, 'Color', lineColors(3, :), 'LineWidth', 5.5);
plot(10*log10(R*Ebm_2D_super/N0), Throu_2D_super, 'Color', lineColors(4, :), 'LineWidth', 5.5);
hold off;

legend("Trad. MAC", "1D MAC", "2D MAC", "2D MAC with SC", 'FontSize', 28, 'Location', 'northwest');
ylabel("Auth Throughput", 'FontSize', 33, 'FontWeight', 'bold');
xlabel("Eb/N0 (dB)", 'FontSize', 33, 'FontWeight', 'bold');

% title("Auth Throughput vs SNR", 'FontSize', 33, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 33, 'LineWidth', 1.5, 'XColor', 'k', 'YColor', 'k');



