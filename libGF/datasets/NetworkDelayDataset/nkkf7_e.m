%% NKKF for scamper data
% allows time-varying Gs
% also handles missing data
% but interpolated for testing
% model used
% y_s(t) = S(t)yl(t) + S(t)nu(t) + S(t)eps(t)
% yl(t) = yl(t-1) + eta(t)
% Ceta = G*Diag(sig(1),sig(2),...,sig(numnds))*G'
% Cnu = (L)^+
% Ceps = sig^2 I
% S(t) : known selection matrix, selects some paths randomly at time t

clear all
clc

%% load, interpolate missing data, remove outliers
% Data, 1: Internet2, 2: NZ-AMP
Data = 1;

if Data == 1
    load latmat
    lats = L(:,ap)';
end
if Data == 2
    load scicmpdata2
end
    
y = lats(:,1:end);
for i = 1:size(y,1)
    y(i,~isfinite(y(i,:))) = interp1(find(isfinite(y(i,:))),y(i,isfinite(y(i,:))),find(~isfinite(y(i,:))));
end
N = length(y);

% remove some paths
%loc = [15 33 54 67 68 79 81 99 110 120 121 128 130 132:139 162 167 174 175 184 186 2 3 4 5 12 26 56 65 87 88 94 119:129 151];
%y(loc,:) = [];
%npath(loc,:) = [];
%yorig = y;


%% Setup 
k = 10;                     % Number of paths measured per time slot
sig2 = 1;                   % Measurement noise

% estimate online covariances online according to [Myers-Tapley'76]
estimate_cov = 1; % 1: estimate, 0: do not estimate
T_training = 4500; % number of time slots employed for training
paths_training = 1; % 1: all paths; 2: k random paths; 3: k optimal paths


% optimize path selection
optimize_paths = 0; % 1: optimal paths, 0: random selection

maxiter = 10;                % Number of iterations 
T = 4500;                   % Number of time slots processed T in [1,N]
T_ini = 500;                % Do not consider the first T_ini slots in the main loop
T_initial = 2;

% simulate  compressed monitoring approach in [Coates-Pointurier-Rabbat'07]
compressed_tau1 = 1; % for tau = 1 
compressed_tau5 = 0; % for tau = 5 
wt = .5;

%% Construct matrix G
if Data == 2
    allnds = unique(npath(npath~=0));
    ndind = zeros(1:max(allnds),1);
    ndind(allnds) = 1:length(allnds);
    G = zeros(size(npath,1),length(allnds));
    for i = 1:size(npath,1)
        G(i,ndind(nonzeros(npath(i,:)))) = 1;
    end
end
numpath = size(G,1);
numnds = size(G,2);
z = zeros(numpath,1);         % zero vector
I = eye(numpath);             % I matrix
O = zeros(numpath);           % All zero matrix
sw = 2;                  % number of paths to swap per time slot


%% Set up parameters for the compressed monitoring approach in [Coates-Pointurier-Rabbat'07]

% modify delay data
% subtract propagation delay, which is assumed to be the minimum delay
prop_delay = min(y')';
yc = y - repmat(prop_delay,1,N);

% create wavelet diffusion matrix for delay prediction
% for tau = 1
W1 = zeros(numpath); 
deg = sum(G')';
for i = 1:numpath-1
    for j = i+1:numpath
        W1(i,j) = (G(i,:)*G(j,:)')/(deg(i) + deg(j) - G(i,:)*G(j,:)');
    end
end
W1 = W1+W1';
if Data == 1
   W1 = W1 + .01.*I;
end
[D1,temp_x,temp_y] = sinkhorn(W1,1e-10);
Tree = DWPTree (D1, numpath, 1e-8, struct('Wavelets',true));
if Data == 1
    dept_T = size(Tree,1) -2;
end
if Data == 2
    dept_T = size(Tree,1);
end
Wav1 =  Tree{dept_T,1}.ExtBasis;
for ii = 1:dept_T
    Wav1 = [Wav1 Tree{ii,2}.ExtBasis]; 
end


% for tau = 5 
if compressed_tau5 == 1
    W5 = [W1    wt*I O O O; wt*I  W1 wt*I  O O; O  wt*I W1 wt*I O; O O wt*I W1 wt*I; O O O wt*I W1];
    [D5,temp_x,temp_y] = sinkhorn(W5,1e-1);
    Tree5 = DWPTree (D5, numpath, 1e-4, struct('Wavelets',true));
    Wav5 =  Tree5{size(Tree5,1),1}.ExtBasis;
    for ii = 1:size(Tree5,1)
        Wav5 = [Wav5 Tree5{ii,2}.ExtBasis]; 
    end 
end



%% Set up Kriging and KKF parameters
% Measurement noise
Ceps = sig2*eye(numpath);

% Kriging Covariance matrix
A = zeros(numpath);
deg = sum(G')';
for i = 1:numpath-1
    for j = i+1:numpath
        A(i,j) = (G(i,:)*G(j,:)')/(deg(i) + deg(j) - G(i,:)*G(j,:)');
    end
end
A = A+A';
D = diag(sum(A'));
L = eye(numpath) - sqrt(inv(D))*A*sqrt(inv(D));
A = G*G';
A(logical(numpath)) = 0;
D = diag(sum(A));
L = D - A;
Cnuk = (G*G'+1e-6*eye(numpath));
Cnu = G*G';%pinv(L); 
Cnu_base = Cnu;

% Process noise, correlation for x
Ceta = cov(y'); 

% kriging paths - if required
[U,S,V] = svd(sqrt(Cnuk));
[Q,R,Pk] = qr(U(:,1:k)');
[pathk,in] = find(Pk(:,1:k));
bias = y(:,1)-Cnuk(:,pathk)*((Cnuk(pathk,pathk)+Ceps(pathk,pathk))\y(pathk,1));

% compressive network monitoring paths - if required
% for tau = 1
[Uc,Sc,Vc] = svd(sqrt(D1));
[Qc,Rc,Pkc] = qr(Uc(:,1:k)');
[pathc,inc] = find(Pkc(:,1:k));
load anrlc rlc;

%% estimation of the covariances
if estimate_cov == 1
    
    gp = 1:numpath;
    I = eye(numpath);
   
    ord = randperm(numpath);
    paths = 1:numpath;
    pran = paths;
    
    % try to estimate state transnition matrix (just a tentative)
    Cy2 = zeros(numpath);
    Cy1 = zeros(numpath);
    for nn = 3:T_training
        Cy2 = Cy2 + y(:,nn)*y(:,nn-2)';
        Cy1 = Cy1 + y(:,nn-1)*y(:,nn-2)';
    end
    Trans = Cy2*pinv(Cy1);
    
    rmean = 0;
    Rcov = Cnu;
    qmean = mean(y(1:T_training-1) - y(2:T_training)); 
    Qcov = Ceta;
    
    
    fprintf('Training phase. \n')
    
    % KKF
    
    chihat = 0.5*y(:,1);
    Mt = 10*eye(numpath);
    Cp = Cnu + Ceps;
    
    for t = 2:T_training
        chihat_1 = chihat;
        Mt_1 = Mt;
        
        
     
        if paths_training == 1
           paths_train = 1:numpath; 
        end
        if paths_training == 2
           ord = randperm(length(gp));
           paths = gp(sort(ord(1:k)));
        end
        if paths_training == 3
           [paths_train,sopt] = kkflazygreedy((Mt+Ceta+Cnu)/sig2,k);
        end
        
        
        obs = y(paths_train,t);
        
        Mt = Mt + Ceta;
        Kt = Mt(:,paths_train)/(Cp(paths_train,paths_train)+Mt(paths_train,paths_train));
        chihat = chihat + Kt*(obs-chihat(paths_train));
        Mt = Mt - Kt*Mt(paths_train,:);
        yhat = chihat + Cnu(:,paths_train)*(Cp(paths_train,paths_train)\(obs-chihat(paths_train)));
       
        % update covariance matrix Ceta
        qmean = ((t-1)/t).*qmean + (1/t).*(chihat);
        Qcov = ((t-1)/t).*Qcov + (1/t).*((chihat - qmean)*(chihat' - qmean')) + (1/t).*(Mt - Mt_1);
        Ceta = Qcov;
        
        % update covariance matrix Cnu
        innov = obs - chihat_1(paths_train);
        Rcov(paths_train,paths_train) = ((t-1)/t).*Rcov(paths_train,paths_train) + (1/t).*(innov*(innov')) + (1/t).*(Mt_1(paths_train,paths_train) + Ceta(paths_train,paths_train));
    
        Rcov2 = Rcov - Ceps;
        alpha = (sum(sum(Rcov2.*Cnu_base)))/(sum(sum(Cnu_base.^2)));
        Cnu = alpha.*Cnu_base;
        
        if(mod(t,500)==0)
            fprintf('.')
        end
        
    end
    
    
    fprintf('\n')
    
end

%% Main loop

tic
ekkf = zeros(maxiter,1); ekf = ekkf; ek = ekkf; ecnm = ekkf;
fprintf('Estimation phase. \n')
fprintf('iter \t\t KKF \t\t KF \t\t Kriging\t Comp. Mon.\n')

pathlist = zeros(k*maxiter,T);

gp = 1:numpath;

mspekkf = zeros(maxiter,T);
mspek = zeros(maxiter,T);
mspecnm = zeros(maxiter,T);
mspekf = zeros(maxiter,T);

for iter = 1:maxiter
    
    rlc = rand(1,numpath);
    rlc = rlc/mean(rlc);
    
    % KKF/Kriging for each time slot
    I = eye(numpath);
    chihat = 0.5*y(:,1);
    %Mt = 10*eye(numpath);
    
    % storage variables
    aa = sum(y); akkf = aa; akf = aa; ak = aa; acnm = aa; apred = zeros(N,1);
    ykkf = zeros(numpath,N); yk = ykkf; ykf = ykkf; ycnm = ykkf;
    ef = zeros(1,N);
    
    ord = randperm(numpath); 
    paths = 1:k;
    pran = paths;
    pfreq = zeros(T,numpath);
    
    kt = zeros(T,1);
    kf = zeros(T,1);
    krig = zeros(T,1);

    bias_cnm = 0;
    
    for t = T_initial:T
         
        
        if optimize_paths == 0
            % select k random paths
                ord = randperm(length(gp));
                paths = gp(sort(ord(1:k)));
                pathk = paths;
                pathc = paths;
        else
             [paths,sopt] = kkflazygreedy((Mt+Ceta+Cnu)/sig2,k);
        end
        
        pathlist(k*(iter-1)+1:k*iter,t) = paths;
        
        oths = setdiff(1:numpath,paths);
        othk = setdiff(1:numpath,pathk);
        othc = setdiff(1:numpath,pathc);

        
        St = I(paths,:);
        Cp = Cnu(paths,paths) + Ceps(paths,paths);
        
        % observations
        obs = y(paths,t);
           
        %------------------------------------------------------------
        % KKF
        %------------------------------------------------------------
        
        chihat_1 = chihat;
        Mt_1 = Mt;
     
        Mt = Mt + Ceta;
        Kt = Mt(:,paths)/(Cp+Mt(paths,paths));
        chihat = chihat + Kt*(obs-chihat(paths));
        Mt = Mt - Kt*Mt(paths,:);
        yhat = chihat + Cnu(:,paths)*(Cp\(obs-chihat(paths)));
       
        kf(t)  = norm(chihat);
        krig(t) = norm(Cnu(:,paths)*(Cp\(obs-chihat(paths))));
        
        %------------------------------------------------------------
        % Kriging only
        %------------------------------------------------------------
        ykhat = Cnuk(:,pathk)*((Cnuk(pathk,pathk)+Ceps(pathk,pathk))\y(pathk,t));
        
        
        % Kriging-only bias correction
        if(t<=T_ini)
            bias = ykhat-y(:,t);
            ykhat = y(:,t);
        else
            ykhat = ykhat-bias;
        end
        
        %------------------------------------------------------------
        % Compressed network monitoring
        %------------------------------------------------------------
        if compressed_tau1 == 1
            
            kk = 1:numpath;
            Wjj = diag((2*ones(1:numpath,1)).^kk);
            
            
            cvx_quiet(true)
            cvx_begin 
                variable b(size(Wav1,2),1)
                
                minimize  norm(b,1)
             
                subject to
                        yc(pathc,t) == Wav1(pathc,:)*Wjj*b;
                        Wav1*b >= 0; %#ok<VUNUS>
            cvx_end
              
            ycnmhat = Wav1*b + prop_delay;
        
            %for ii = 1:numpath
            %    ycnmhat(ii) = max(ycnmhat(ii),0);
            %end
            
            
            % Compressed network monitoring bias correction
            %if Data == 1 
                if(t<=T_ini)
                    %bias_cnm = bias_cnm + (ycnmhat-y(:,t))/T_ini;
                    bias_cnm = ycnmhat-y(:,t);
                    ycnmhat = y(:,t);
                else
                    ycnmhat = ycnmhat-bias_cnm;
                end
            %end
        
        else
            ycnmhat = zeros(numpath,1);
        end
        
        % store tracked variables
        ykkf(:,t) = yhat;
        yk(:,t) = ykhat;
        ykf(:,t) = chihat;
        
        ykkf(paths,t) = obs;
        yk(paths,t) = obs;
        
        ycnm(:,t) =  ycnmhat;
        
        % store total delay 
        akkf(t) = sum(yhat);
        akf(t) = sum(chihat);
        ak(t) = sum(ykhat);
        acnm(t) = sum(ycnmhat);
        
        if(t>T_ini)
            % store prediction errors
            mspekkf(iter,t) = sum((yhat(oths)-y(oths,t)).^2)/length(oths);
            mspek(iter,t) = sum((ykhat(othk)-y(othk,t)).^2)/length(othk);
            mspekf(iter,t) = sum((chihat(oths)-y(oths,t)).^2)/length(othk);
            mspecnm(iter,t) = sum((ycnmhat(othc)-y(othc,t)).^2)/length(othc);
        end
       
        apred(t) = sqrt(sum(sum(Mt)));
        
        if(mod(t,500)==0)
            fprintf('.')
        end
        
        
    end
    
    % bias 
%     bias_cnm = mean(y(othc,:)'-ycnm(othc,:)')';
%     ycnm(othc,1:T) = ycnm(othc,1:T) + repmat(bias_cnm,1,T);
%     for t=T_ini:T
%             % store prediction errors
%             mspecnm(iter,t) = sum((ycnm(othc,t)-y(othc,t)).^2)/length(othc);
%     end    
    
    fprintf('\n%g\t\t%g\t\t%g\t\t%g\t\t\t%g\t\t\g',iter,sum(mspekkf(iter,T_ini:T))/(T-T_ini),sum(mspekf(iter,T_ini:T))/(T-T_ini),sum(mspek(iter,T_ini:T))/(T-T_ini),sum(mspecnm(iter,T_ini:T))/(T-T_ini))
    
    
    %------------------------------------------------------------
    % Compressed network monitoring for tau = 5
    %------------------------------------------------------------
    ycnm5 = zeros(numpath,T);
    N_cnm5 = floor(T/5);
    mspecnm5 = zeros(maxiter,T);
    
    tau = 5;
    
    if compressed_tau5 == 1
        
        for n = 1:N_cnm5
        
            obs5 = yc(pathc,tau*(n-1)+1:tau*n);
            obs5 = reshape(obs5,k*tau,1);
            
            pathcv = [];
            othcv = [];
            for i = 1:tau
                pathcv = [pathcv (pathc+(numpath*(i-1)))];
                othcv = [othcv  (othc+(numpath*(i-1)))];
            end
            pathcv = reshape(pathcv,k*tau,1);
            othcv = reshape(othcv,(numpath-k)*tau,1);
            
            cvx_quiet(true)
            cvx_begin 
                variable b(size(Wav5,2),1)
            
                minimize sum((obs5 - Wav5(pathcv,:)*b).^2) + norm(b,1)
             
                subject to
                        obs5 == Wav5(pathcv,:)*b;         %#ok<*EQEFF>
                        Wav5*b >= 0; %#ok<VUNUS>
            cvx_end
       
            ycnm5hat =  Wav5*b;
            ycnm5hat = reshape(ycnm5hat,numpath,tau) + repmat(prop_delay,1,tau);
            
            % store values
            ycnm5(:,tau*(n-1)+1:tau*n) =  ycnm5hat;
            
            index = tau*(n-1)+1:tau*n;
            for ii = 1:tau
                mspecnm(iter,index(ii)) = sum((ycnm5(othc,index(ii))-y(othc,index(ii))).^2)/length(othc);       
            end
            
            
        end
    end
    
    
    
    %     plot(2:T,ef(2:T));
    
     plot(1:N,aa,'b',1:N,akkf,'r.',1:N,ak,'k.',1:N,acnm,'y.'); legend('True','KKF','Kriging','Compressive monitoring')

     
    %return

    fprintf('\n')
    
end
fprintf('\n')
%[mean(ekkf) mean(ekf) mean(ek)]





