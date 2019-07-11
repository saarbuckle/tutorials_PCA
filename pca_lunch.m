function varargout = pca_lunch(what,varargin)
%% ------------ function varargout = pca_lunch(what,varargin) -------------
% 
% A library of functions for doing some pca things for methods lunches.
% I've tried to add some explanations (hopefully most are correct).
% 
% How to call:
%   
%   outputs = pca_lunch('case_you_want_to_run', case-specific arguments)
%
% Case-specific arguments can be found in details for each case.
%
%   Example:
%   Simulate correlated data using the default model G, get the PCs of the
%   simulated data, and scatterplot the data and the principal components:
%
%           X     = pca_lunch('SIM:correlated');    % simulate data
%           [V,D] = pca_lunch('PCA:svd',X);         % do PCA using SVD
%           figure('Color',[1 1 1]);                % figure (white bkgrnd)
%           pca_lunch('PLOT:dataScatter',X);        % scatter data
%           pca_lunch('PLOT:drawlines',size(X,2));  % draw axis lines
%           pca_lunch('PLOT:pcLines',V,D);          % draw lines for PCs
%   
%
% saarbuckle@gmail.com 2019
%
% -------------------------------------------------------------------------

%% ------------ set defaults for simulated data ---------------------------
% set covariance matrix defaults
var      = [1,1];             % variances (diagonal values in G)
numC     = numel(var);        % number of variances dictates # variables
covar    = [0.8];             % covariances (off-diagonal values in G)
N        = 100;               % no. of desired measurement channels (neuron, voxel, etc.)
N        = N*numC;            % make N divisible by numC for sparse patterns
sigma    = 1;                 % default standard deviation of measurement channels (please don't change)
numIters = 1000;              % number of simulated datasets per model

%% ------------ set defaults for plot stylings ----------------------------
pcLineWidth = 2;
pc_clr      = {[1 0.6 0],[1 0 0],[0.7 0 0]}; % plotting colors for PCs
ax_clr      = [0 0 0 0.25];                  % set colour for lines drawn denoting x and y axes in figure space (the 4th value dictates transperancy)
ax_width    = 1;                             % width of the axis line
data_face   = [0.8 0.8 0.8];                 % marker face colours for data scatterplots
data_edge   = [0 0 0];                       % marker edge colours for data scatterplots
alpha_face  = 0.25;                          % transperancy of marker faces (1 = no transperancy)
alpha_edge  = 0.25;                          % transperancy of marker edges (1 = no transperancy)
tjLineWidth = 1.25;

%% ------------ brief error handling --------------------------------------
if sum(ismember(numC,[2,3])) < 1
    warning('Some cases in this code (plotting funcs) only support data sets with 2 or 3 conditions.');
end

%% ------------ function cases --------------------------------------------
switch what
    case '0' % ------------ wrapper functions for lunches -----------------
    case 'DO:corrEig'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Make one set of data where conditions are correlated.
        % Applies PCA via eig-decomposition.
        % Plots scatterplots of the dataset, the estimated PCs, and the
        % reconstructed PCs. 
        % You can change how many PCs are used in reconstruction by
        % changing k.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        k = 1; % how many PCs are used to reconstruct data?
        % X : correlated measurement channels
        X = pca_lunch('SIM:correlated');
        % do eigenvalue decomp for each dataset
        [V,D,Xproj,Xrecon] = pca_lunch('PCA:eig',X,k);
        % plot results        =
        pca_lunch('PLOT:doCorrPlots',X,V,D,Xproj,Xrecon);
    case 'DO:corrSvd'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Make one set of data where conditions are correlated.
        % Applies PCA via eig-decomposition.
        % Plots scatterplots of the dataset, the estimated PCs, and the
        % reconstructed PCs. 
        % You can change how many PCs are used in reconstruction by
        % changing k.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        k = 1; % how many PCs are used to reconstruct data?
        % X : correlated measurement channels
        X = pca_lunch('SIM:correlated');
        % do eigenvalue decomp for each dataset
        [V,D,Xproj,Xrecon] = pca_lunch('PCA:eig',X,k);
        % plot results        
        pca_lunch('PLOT:doCorrPlots',X,V,D,Xproj,Xrecon);
    case 'DO:allEig'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % make three sets of data:
        % Generates three sets of noiseless data, applies PCA via eig-decomposition.
        % Plots scatterplots of the datasets, the estimated PCs, and the
        % reconstructed PCs. 
        % You can change how many PCs are used in reconstruction by
        % changing k.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        k = 1; % how many PCs are used to reconstruct data?
        names = {'sparse','uncorrelated','correlated'};
        % X1 : sprasely tuned measurement channels (respond to only one condition)
        % X2 : not sprasely tuned measurement channels (can responsd to all conditions)
        % X3 : not sparsely tuned and correlated measurement channels
        X1 = pca_lunch('SIM:sparse');       X1 = pca_lunch('MISC:normalize',X1);
        X2 = pca_lunch('SIM:uncorrelated'); X2 = pca_lunch('MISC:normalize',X2);
        X3 = pca_lunch('SIM:correlated');   X3 = pca_lunch('MISC:normalize',X3);
        % do eigenvalue decomp for each dataset
        [V1,D1,X1proj,X1recon] = pca_lunch('PCA:eig',X1,k);
        [V2,D2,X2proj,X2recon] = pca_lunch('PCA:eig',X2,k);
        [V3,D3,X3proj,X3recon] = pca_lunch('PCA:eig',X3,k);
        
        % plot results
        pca_lunch('PLOT:doAllPlots',...
            {X1,X2,X3},...
            {V1,V2,V3},...
            {D1,D2,D3},...
            {X1proj,X2proj,X3proj},...
            {X1recon,X2recon,X3recon},...
            names);
    case 'DO:allSvd'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Generates three sets of noiseless data, applies PCA via SVD, and
        % plots scatterplots of the datasets, the estimated PCs, and the
        % reconstructed PCs. 
        % You can change how many PCs are used in reconstruction by
        % changing k.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        k = 1; % how many PCs are used to reconstruct data?
        names = {'sparse','uncorrelated','correlated'};
        % make three sets of data:
        % X1 : sprasely tuned measurement channels (respond to only one condition)
        % X2 : not sprasely tuned measurement channels (can responsd to all conditions)
        % X3 : not sparsely tuned and correlated measurement channels
        X1 = pca_lunch('SIM:sparse');
        X2 = pca_lunch('SIM:uncorrelated');
        X3 = pca_lunch('SIM:correlated');
        % do decomp via svd for each dataset
        
        [V1,D1,X1proj,X1recon] = pca_lunch('PCA:svd',X1,k);
        [V2,D2,X2proj,X2recon] = pca_lunch('PCA:svd',X2,k);
        [V3,D3,X3proj,X3recon] = pca_lunch('PCA:svd',X3,k);
        
         % plot results
        pca_lunch('PLOT:doAllPlots',...
            {X1,X2,X3},...
            {V1,V2,V3},...
            {D1,D2,D3},...
            {X1proj,X2proj,X3proj},...
            {X1recon,X2recon,X3recon},...
            names);
    case 'DO:pcDist'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Estimates the distribution of the first principal component 
        % estimated for multiple datasets WITHOUT ADDED NOISE.
        % Importantly, these datasets are all simulated from the same 
        % model (3 different models are used for three different distribution
        % maps- 'sparse','uncorrelated','correlated')
        % The case will plot three heatmaps. Each map corresponds to
        % one of the three models used to generate data.
        %
        % You can adjust how many simulated datasets are made for each
        % model by changing numIters.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        %type     = {'sparse','uncorrelated','correlated'};
        type     = {'correlated'};
        numType  = numel(type);
        figure('Color',[1 1 1]);
        for i = 1:numType
            subplot(1,numType,i); 
            [V,D] = pca_lunch('SIM:pcDist',type{i},numIters);
            pca_lunch('PLOT:pcDist',V,D);
            title(sprintf('1st PC density\n%d %s sims.', size(V,2), type{i}));
        end
    case 'DO:pcDistNoise'    
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Estimates the distribution of the first principal component 
        % estimated for multiple datasets WITH NOISE.
        % Importantly, these datasets are all simulated from the same 
        % model (3 different models are used for three different distribution
        % maps- 'sparse','uncorrelated','correlated')
        % The noise level changes on each simulation.
        % The case will plot several heatmaps. Each column corresponds to
        % one of the three models used to generate data, and each row
        % corresponds to different levels of noise added to the noiseless
        % data.
        %
        % You can adjust how many simulated datasets are made for each
        % model by changing numIters.
        %
        % You can adjust the added noise by changing noiseVar.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        noiseVar  = [0,0.001,0.01,0.1,1,10,100];
        signalVar = 1;
        normalize = 0;
        %type      = {'sparse','uncorrelated','correlated'};
        type      = {'correlated'};
        figure('Color',[1 1 1]);
        k = 1; % subplot ticker
        numNoise = numel(noiseVar);
        numType  = numel(type);
        for i = 1:numType
            for j = 1:numNoise
                subplot(numType,numNoise,k); 
                [V,D] = pca_lunch('SIM:pcDist',type{i},numIters,normalize,1,signalVar,noiseVar(j));
                pca_lunch('PLOT:pcDist',V,D);
                title(sprintf('%s\nno. sims: %d\nnoise: %1.3f', type{i}, size(V,2), noiseVar(j)));
                k = k+1;
            end
        end
    case 'DO:reconstructX'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Here, we reconstruct X from PCs, using variable numbers of PCs.
        % We plot the reconstruction error (sums of squared residuals), and
        % the cummulative variance explained by using each successive PC in
        % the reconstruction. 
        % The two lines should trend in opposite directions: more variance
        % explained = less reconstruction error.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numC     = 10;
        N        = numC*10;
        A        = rand(numC,N);
        G        = (A*A')./(N-1) + numC*diag(unifrnd(0,1,numC,1)); % positive semidefinite G
        X        = pca_lunch('SIM:mvnrndExact',G,N,sigma);
        [V,~,~,~,varExp] = pca_lunch('PCA:svd',X);
        % calculate error from reconstructions using eigenvectors
        Xmu = sum(X,1)./size(X,1); % column means
        Xc  = X - Xmu;             % remove column means
        ssres = nan(1,numC);
        for i = 1:numC
            Vi       = V(:,1:i);
            Xrecon   = Xc*Vi*Vi' + Xmu;
            res      = X(:) - Xrecon(:);
            ssres(i) = res'*res;
        end
        ssres     = ssres;
        cumVarExp = cumsum(varExp); % cummulative variance explained
        % plot sums-squared-residuals (normalized to range between 0 and
        % 1), and cummulative variance explained (also normalized) on same
        % plot.
        figure('Color',[1 1 1]);
        clr = {[0 0 0],[1 0 0]};
        yyaxis left
        plot(1:numC,cumVarExp,'Color',clr{1},'Linewidth',1.5);
        ylabel('cummulative variance explained (%)');
        yyaxis right
        plot(1:numC,ssres,'Color',clr{2},'LineWidth',1.5);
        ylabel('reconstruction error (a.u.)  ||(X-Xrecon)||2');
        % stylize
        h = get(gca);
        h.YAxis(1).Color = clr{1};
        h.YAxis(2).Color = clr{2};
        xlabel('# PCs');
        title('X reconstruction from PCs');    
        xlim([0.5 numC+0.5]);
    case 'DO:noStandardization'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Do pca where the standard deviation of each measurement channel
        % is not consistent.
        % Here, we expect LOUD channels to "pull" PCs towards them (b/c PCA
        % minimizes L2-norm).
        % Then, we will standardize the channels, redo PCA, and demonstrate
        % the importance of standardizing the channels.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % 1. Make non-standardized data with opposing correlations between
        % dimensions (means are identical - 0)
        G  = diag([10,1]);
        X  = pca_lunch('SIM:mvnrndExact',G,N,sigma);
        X  = pca_lunch('SIM:mvnrndNoise',X,5,1);
        % 2. Do PCA on non-standardized data
        [V1,D1] = pca_lunch('PCA:eig',X);
        % 3. Scatterplot the data, show the difference in the spreads, then
        % show the PCs
        figure('Color',[1 1 1]);
        subplot(2,2,1); % scatterplot data with PCs
        pca_lunch('PLOT:dataScatter',X);
        pca_lunch('PLOT:drawlines',numC,[0 0 0]);
        pca_lunch('PLOT:pcLines',V1,D1,pc_clr);
        title('NOT normalized');

        % 4. Standardize the data and redo PCA
        ranges  = range(X); % range of each channel per condition
        Xn      = bsxfun(@times,X,1./ranges); % scale all channels by their range
        [V2,D2] = pca_lunch('PCA:eig',Xn);
        % 5. Plot redone PC results
        subplot(2,2,3); % scatterplot data with PCs
        pca_lunch('PLOT:dataScatter',Xn);
        pca_lunch('PLOT:drawlines',numC,[0 0 0]);
        pca_lunch('PLOT:pcLines',V2,D2,pc_clr);
        title('normalized');
        
        % 6 Make heatmap of 1st PC in raw and standardized spaces:
        V  = nan(numC^2,numIters);  % pre-allocate arrays
        D  = nan(numC,numIters);
        Vn = V;
        Dn = D;
        for i = 1:numIters
            Xi = pca_lunch('SIM:mvnrndExact',G,N,sigma);
            %Xi = pca_lunch('SIM:mvnrndNoise',Xi,5,1);
            % PCA on raw patterns
            [v,d]  = pca_lunch('PCA:svd',Xi);
            V(:,i) = v(:);
            D(:,i) = d;
            % standardize the same patterns
            ranges = range(Xi); % range of each channel per condition
            Xin    = bsxfun(@times,Xi,1./ranges); % scale all channels by their range
            % PCA on standardized patterns
            [v,d]   = pca_lunch('PCA:svd',Xin);
            Vn(:,i) = v(:);
            Dn(:,i) = d;
        end
        subplot(2,2,2);
        pca_lunch('PLOT:pcDist',V,D);
        title('1st PC density- NOT normalized');
        subplot(2,2,4);
        pca_lunch('PLOT:pcDist',Vn,Dn);
        title('1st PC density- normalized');
    case 'DO:compareEigSvd'
        X = pca_lunch('SIM:correlated');
        [V1,D1,PC1] = pca_lunch('PCA:eig',X);
        [V2,D2,PC2] = pca_lunch('PCA:svd',X);
        
        % check covariances are the same
        disp('EIG covariance estimate:')
        disp(V1*diag(D1)*V1')
        disp('SVD covariance estimate:')
        disp(V2*diag(D2)*V2')
        
        % check eigenvalues are the same
        disp('EIG eigenvalues')
        disp(D1)
        disp('SVD eigenvalues')
        disp(D2)
        
        % check eigenvectors are the same
        disp('EIG eigenvectors')
        disp(V1)
        disp('SVD eigenvectors')
        disp(V2)
        
        % check correlation between PC projections (off-diagonals should be
        % zero in all cases)
        disp('Correlation matrix between PCs from EIG decomp (should be identity)')
        disp(corr(PC1))
        disp('Correlation matrix between PCs from SVD decomp (should be identity)')
        disp(corr(PC2))
        disp('Correlation matrix across PCs from both decomps (should be identity)')
        disp(corr(PC1,PC2))
    case 'DO:compareEigSvdImage'
        X = pca_lunch('IMG:baboon');
        [~,D1,PC1] = pca_lunch('PCA:eig',X);
        [~,D2,PC2] = pca_lunch('PCA:svd',X);
        
        % check eigenvalues are the same
        k = 5;
        disp('EIG eigenvalues')
        disp(D1(1:k))
        disp('SVD eigenvalues')
        disp(D2(1:k))
        
        % check correlation between PC projections (should be identity)
        figure('Color',[1 1 1]);
        subplot(1,3,1); imagesc(corr(PC1)); title('Corr b/t PCs (EIG)'); caxis([0 1]);
        subplot(1,3,2); imagesc(corr(PC2)); title('Corr b/t PCs (SVD)'); caxis([0 1]);
        subplot(1,3,3); imagesc(abs(corr(PC1,PC2))); title('Corr across SVD & EIG PCs'); caxis([0 1]);
                % abs of correlations because signs might change between
                % eig and svd (the signs are superfluous)
    
    case 'DO:exampleDynamics'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Here we will simulate temporal data for a system with dynamics
        % only. These systems are initialized by the first input U(:,:,1),
        % but then evolve over time using only A*X(:,:,t-1).
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numD    = 6;
        numCols = 3;
        numRows = ceil(numD/numCols);
        T       = 50;
        figure('Color',[1 1 1]);
        % get starting state (this will stay constant across datasets)
        U = pca_lunch('SIM:temporalDataPrep');
        for i = 1:numD
            % initialize new transformation A
            [~,A,B] = pca_lunch('SIM:temporalDataPrep');
            % generate temporal data
            X = pca_lunch('SIM:temporalData',U,A,B,1,0,T);
            % plot PC trajectories of this simulated data
            subplot(numRows,numCols,i);
            pca_lunch('PLOT:trajectory',X);
        end
    case 'DO:exampleInputs'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Here we will simulate temporal data for a system with dynamics
        % only. These systems are initialized by the first input U(:,:,1),
        % but then evolve over time using only A*X(:,:,t-1).
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numD    = 6;
        numCols = 3;
        numRows = ceil(numD/numCols);
        T       = 50;
        figure('Color',[1 1 1]);
        % get starting state (this will stay constant across datasets)
        U = pca_lunch('SIM:temporalDataPrep');
        for i = 1:numD
            % initialize new transformation A
            [~,A,B] = pca_lunch('SIM:temporalDataPrep');
            % generate temporal data
            X = pca_lunch('SIM:temporalData',U,A,B,0,1,T);
            % plot PC trajectories of this simulated data
            subplot(numRows,numCols,i);
            pca_lunch('PLOT:trajectory',X);
        end    
    case 'DO:variableInputWeights'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Here we will simulate temporal data for a system with dynamics
        % and input-driven components. 
        % These systems are initialized by the the same inputs and uses the
        % same A & B each time.
        % The degree the system is input-driven is varied.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numD    = 6;
        numCols = 3;
        numRows = ceil(numD/numCols);
        T       = 50;
        b       = [0 0.01 0.05 0.1 0.5 1];%linspace(0,1,numD);
        figure('Color',[1 1 1]);
        % get starting state (this will stay constant across datasets)
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        for i = 1:numD
            % generate temporal data
            X = pca_lunch('SIM:temporalData',U,A,B,1,b(i),T);
            % plot PC trajectories of this simulated data
            subplot(numRows,numCols,i);
            pca_lunch('PLOT:trajectory',X);
        end
    case 'DO:variableDynamicWeights'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Here we will simulate temporal data for a system with dynamics
        % and input-driven components. 
        % These systems are initialized by the the same inputs and uses the
        % same A & B each time.
        % The degree the system reflects dynamics is varied.
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numD    = 6;
        numCols = 3;
        numRows = ceil(numD/numCols);
        T       = 50;
        a       = linspace(0,1,numD);
        figure('Color',[1 1 1]);
        % get starting state (this will stay constant across datasets)
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        for i = 1:numD
            % generate temporal data
            X = pca_lunch('SIM:temporalData',U,A,B,a(i),1,T);
            % plot PC trajectories of this simulated data
            subplot(numRows,numCols,i);
            pca_lunch('PLOT:trajectory',X);
        end
    case 'DO:dynamicComponents'   
        % Split a dynamic system into the rotational and non-rotational
        % components, and plot resulting transforms.
        T = 50;                                         % timesteps
        C = 5;                                          % conditions
        N = C;                                          % measurement channels 
        U = pca_lunch('SIM:mvnrndExact',eye(C),C,1);    % starting input
        A = pca_lunch('SIM:transformA',N,N);
        Ar = (A-A')./2;         % rotational transform (imaginary eigenvalues)
        As = ((A+A')./2).*0.1;  % non-rotational transform (real eigenvalues)
        At = Ar+As;             % make the rotational component more strong so we can see it
        X{1} = pca_lunch('SIM:autoDynamics',Ar,U,T);
        X{2} = pca_lunch('SIM:autoDynamics',As,U,T);
        X{3} = pca_lunch('SIM:autoDynamics',At,U,T);
        figure('Color',[1 1 1]);
        subplot(1,3,1); pca_lunch('PLOT:trajectory',X{1}); title('1: rotational components');
        subplot(1,3,2); pca_lunch('PLOT:trajectory',X{2}); title('2: non-rotational components');
        subplot(1,3,3); pca_lunch('PLOT:trajectory',X{3}); title('3: all components');
    case 'DO:tensorAnalysis1'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Makes temporal dataset X (N x C x T), where X is characterized
        % (to some variable degree) by a dynamic transformation A and
        % time-variable inputs B*U:
        %
        %        x(:,:,t+1) = a*A*x(:,:,t) + b*B*u(:,:,t)
        %
        % where a and b are weights applied to the dynamical and
        % input-driven components of the data. 
        %
        % In this case, we generate temporal data with the same A, B, and
        % U, but we make 2 datasets:
        %   1. A dynamics only system
        %   2. An input-driven system
        %
        % With these datasets, we then apply PCA along the neuron-unfolding
        % and condition-unfolding of the data tensors (tensors are 3+
        % dimensional matrices where the dimensions are different units).
        % Unfolding the tensor X (NxCxT) along the neuron-mode results in a
        % reshaped matrix Xr (NxCT). Applying PCA via svd to the reshaped
        % matrix Xr will thus find "basis-neurons" (1xCT or CxT). These
        % neurons will demonstrate input-driven features of the data, and
        % importantly, will not capture any dynamic features in the data
        % that exist across conditions. 
        %
        % Alternatively, we can unfold X (NxCxT) along the condition-mode,
        % resulting in another rehaped matrix Xr (CxNT). Applying PCA via
        % svd to Xr now yeilds a set of "basis-conditions" (1xNT or NxT),
        % which characterizes a dynamical transform that exists across
        % neurons and times and is shared acorss a set of conditions. Multi
        % basis-conditions could also suggest that there are different
        % dynamics at play for subsets of conditions (e.g. perhaps
        % different categories of images initiate dynamical processes that
        % evolve with different A).
        %       * I think that last point is correct, but I haven't tested
        %       it.
        %
        % These unfolding analyses are from: 
        % Seely et al. (2016). https://doi.org/10.1371/journal.pcbi.1005164
        %
        % After finding the basis-conditions and basis-neurons for each
        % dataset, we ask how well these bases can reconstruct the data.
        % Reconstruction fit is equaluated as the mean R2 across conditions
        % (same results hold if we use mean R2 across neurons). For
        % practical purposes, I'm plotting only the fits using the first
        % top basis neuron or basis condition. I take the log of the ratio 
        % of these fits and plot them, such that a value of 0 indicates
        % similar reconstruction ability using either basis neurons or
        % conditions, values above 0 indicate the system is more dynamic,
        % and vice versa.
        %
        % Important to note, a and b range between 0 and 1. The ratio
        % between these two weights might not exactly relate to the degree
        % to which a system exhibits strong/weak dynamics. Larger values of
        % a will indeed mean there are more dynamics, but input-driven
        % effects on the data are stronger than dynamic effects even when
        % a==b. 
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % generate initial inputs to system
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        [~,C,T] = size(U);
        N       = size(A,1);
        % generate the datasets:
        X{1} = pca_lunch('SIM:temporalData',U,A,B,1,0,T);         % make fully dynamic system
        X{2} = pca_lunch('SIM:temporalData',U,A,B,0,1,T);         % make fully input-driven system
        numDatasets = numel(X);
        % Do reconstruction using neuron and condition modes, assess fits
        r2 = nan(min([N,C]),2,numDatasets); % preallocate array for reconstruction fits
        ratio = nan(numDatasets,1);
        for i = 1:numDatasets
            Xp = pca_lunch('TENSOR:prepData',X{i});                 % preprocess data
            % do the reconstruction along different tensor unfoldings of X
            condR2   = pca_lunch('TENSOR:conditionMode',Xp); % condition-mode reconstruction
            neuronR2 = pca_lunch('TENSOR:neuronMode',Xp);    % neuron-mode reconstruction
            % avg. reconstruction fits across conditions
            r2(:,1,i) = mean(condR2,2);
            r2(:,2,i) = mean(neuronR2,2);
            % take the ratio of the reconstruction with the first (best) PC
            % Here it is the log ratio such that a ratio of 1 (equal fits
            % using either basis-neurons or basis-conditions) will be zero.
            ratio(i)  = log(r2(1,1,i) / r2(1,2,i));
                % ratios >0 indicate better fit with basis-conditions (i.e.
                % system exhibits stronger dynamical properties)
        end
        
        % plot
        figure('Color',[1 1 1]);
        % plot trajectories through PC ("neural state") space
        subplot(2,2,1); pca_lunch('PLOT:trajectory',X{1}); title('1: dynamic system');
        subplot(2,2,3); pca_lunch('PLOT:trajectory',X{2}); title(sprintf('2: input-driven\nRANDOM input'));
        % plot a and b weights of each dataset:
        a = [1,0];
        b = [0,1];
        subplot(2,2,2); pca_lunch('PLOT:temporalWeights',a,b);
        % plot reconstruction fits
        subplot(2,2,4); 
        label = {sprintf('\tdynamic\n\tsystem'),...
            sprintf('\tinput-driven\n\trandom input')};
        pca_lunch('PLOT:tensorFitRatios',ratio,label); 
        ylim([-1 1]);
    case 'DO:tensorAnalysisAll'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Makes temporal dataset X (N x C x T), where X is characterized
        % (to some variable degree) by a dynamic transformation A and
        % time-variable inputs B*U:
        %
        %        x(:,:,t+1) = a*A*x(:,:,t) + b*B*u(:,:,t)
        %
        % where a and b are weights applied to the dynamical and
        % input-driven components of the data. 
        %
        % In this case, we generate temporal data with the same A, B, and
        % U, but we make 4 datasets:
        %   1. A dynamics only system
        %   2. An input-driven system
        %   3. An input-driven system where inputs don't change.
        %   4. An input-driven system where inputs change in predictable manner.
        %
        % With these datasets, we then apply PCA along the neuron-unfolding
        % and condition-unfolding of the data tensors (tensors are 3+
        % dimensional matrices where the dimensions are different units).
        % Unfolding the tensor X (NxCxT) along the neuron-mode results in a
        % reshaped matrix Xr (NxCT). Applying PCA via svd to the reshaped
        % matrix Xr will thus find "basis-neurons" (1xCT or CxT). These
        % neurons will demonstrate input-driven features of the data, and
        % importantly, will not capture any dynamic features in the data
        % that exist across conditions. 
        %
        % Alternatively, we can unfold X (NxCxT) along the condition-mode,
        % resulting in another rehaped matrix Xr (CxNT). Applying PCA via
        % svd to Xr now yeilds a set of "basis-conditions" (1xNT or NxT),
        % which characterizes a dynamical transform that exists across
        % neurons and times and is shared acorss a set of conditions. Multi
        % basis-conditions could also suggest that there are different
        % dynamics at play for subsets of conditions (e.g. perhaps
        % different categories of images initiate dynamical processes that
        % evolve with different A).
        %       * I think that last point is correct, but I haven't tested
        %       it.
        %
        % These unfolding analyses are from: 
        % Seely et al. (2016). https://doi.org/10.1371/journal.pcbi.1005164
        %
        % After finding the basis-conditions and basis-neurons for each
        % dataset, we ask how well these bases can reconstruct the data.
        % Reconstruction fit is equaluated as the mean R2 across conditions
        % (same results hold if we use mean R2 across neurons). For
        % practical purposes, I'm plotting only the fits using the first
        % top basis neuron or basis condition. I take the log of the ratio 
        % of these fits and plot them, such that a value of 0 indicates
        % similar reconstruction ability using either basis neurons or
        % conditions, values above 0 indicate the system is more dynamic,
        % and vice versa.
        %
        % Important to note, a and b range between 0 and 1. The ratio
        % between these two weights might not exactly relate to the degree
        % to which a system exhibits strong/weak dynamics. Larger values of
        % a will indeed mean there are more dynamics, but input-driven
        % effects on the data are stronger than dynamic effects even when
        % a==b. 
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % generate initial inputs to system
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        [~,C,T] = size(U);
        N       = size(A,1);
        % generate the datasets:
        X{1} = pca_lunch('SIM:temporalData',U,A,B,1,0,T);         % make fully dynamic system
        X{2} = pca_lunch('SIM:temporalData',U,A,B,0,1,T);         % make fully input-driven system
        X{3} = pca_lunch('SIM:temporalData',U(:,:,1),A,B,0,1,T);  % make fully input-driven where input doesn't change
        Ad   = pca_lunch('SIM:transformA',C,2);                   % make dynamic input where dynamics are rotational
        Ad   = (Ad-Ad')./2;
        Ud   = pca_lunch('SIM:autoDynamics',Ad,U(:,:,1),T);       % make fully input-driven where input changes in predictable manner
%         w = linspace(0,1,T/2);
%         w = [zeros(1,T/2) w];
%         w = kron(w,ones(C));
%         w = reshape(w,C,C,T);
%         Ud   = Ud.*flip(w,3) + U.*w;
        X{4} = pca_lunch('SIM:temporalData',Ud,A,B,0,1,T);
        numDatasets = numel(X);
        % Do reconstruction using neuron and condition modes, assess fits
        r2 = nan(min([N,C]),2,numDatasets); % preallocate array for reconstruction fits
        ratio = nan(numDatasets,1);
        for i = 1:numDatasets
            Xp = pca_lunch('TENSOR:prepData',X{i});                 % preprocess data
            % do the reconstruction along different tensor unfoldings of X
            condR2   = pca_lunch('TENSOR:conditionMode',Xp); % condition-mode reconstruction
            neuronR2 = pca_lunch('TENSOR:neuronMode',Xp);    % neuron-mode reconstruction
            % avg. reconstruction fits across conditions
            r2(:,1,i) = mean(condR2,2);
            r2(:,2,i) = mean(neuronR2,2);
            % take the ratio of the reconstruction with the first (best) PC
            % Here it is the log ratio such that a ratio of 1 (equal fits
            % using either basis-neurons or basis-conditions) will be zero.
            ratio(i)  = log(r2(1,1,i) / r2(1,2,i));
                % ratios >0 indicate better fit with basis-conditions (i.e.
                % system exhibits stronger dynamical properties)
        end
        
        % plot
        figure('Color',[1 1 1]);
        % plot trajectories through PC ("neural state") space
        subplot(2,3,1); pca_lunch('PLOT:trajectory',X{1}); title('1: dynamic system');
        subplot(2,3,2); pca_lunch('PLOT:trajectory',X{2}); title(sprintf('2: input-driven\nRANDOM input'));
        subplot(2,3,4); pca_lunch('PLOT:trajectory',X{3}); title(sprintf('3: input-driven\nSTABLE input'));
        subplot(2,3,5); pca_lunch('PLOT:trajectory',X{4}); title(sprintf('4: input-driven\nDYNAMIC input'));
        % plot a and b weights of each dataset:
        a = [1,0,0,0];
        b = [0,1,1,1];
        subplot(2,3,3); pca_lunch('PLOT:temporalWeights',a,b);
        % plot reconstruction fits
        subplot(2,3,6); 
        label = {sprintf('\tdynamic\n\tsystem'),...
            sprintf('\tinput-driven\n\trandom input'),...
            sprintf('\tinput-driven\n\tstable input'),...
            sprintf('\tinput-driven\n\tdynamic input')};
        pca_lunch('PLOT:tensorFitRatios',ratio,label); 
        ylim([-1 1]);
    case 'DO:tensorAnalysis2'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Makes temporal dataset X (N x C x T), where X is characterized
        % (to some variable degree) by a dynamic transformation A and
        % time-variable inputs B*U:
        %
        %        x(:,:,t+1) = a*A*x(:,:,t) + b*B*u(:,:,t)
        %
        % where a and b are weights applied to the dynamical and
        % input-driven components of the data. 
        %
        % In this case, we generate temporal data with the same A, B, and
        % U, but we make 2 datasets:
        %   3. An input-driven system where inputs don't change.
        %   4. An input-driven system where inputs change in predictable manner.
        %
        % Same description as tenosirAnalysis1
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % generate initial inputs to system
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        [~,C,T] = size(U);
        N       = size(A,1);
        % generate the datasets:
        X{1} = pca_lunch('SIM:temporalData',U(:,:,1),A,B,0,1,T);  % make fully input-driven where input doesn't change
        Ad   = pca_lunch('SIM:transformA',C,2);                   % make dynamic input where dynamics are rotational
        Ad   = (Ad-Ad')./2;
        Ud   = pca_lunch('SIM:autoDynamics',Ad,U(:,:,1),T);       % make fully input-driven where input changes in predictable manner
%         w = linspace(0,1,T/2);
%         w = [zeros(1,T/2) w];
%         w = kron(w,ones(C));
%         w = reshape(w,C,C,T);
%         Ud   = Ud.*flip(w,3) + U.*w;
        X{2} = pca_lunch('SIM:temporalData',Ud,A,B,0,1,T);
        numDatasets = numel(X);
        % Do reconstruction using neuron and condition modes, assess fits
        r2 = nan(min([N,C]),2,numDatasets); % preallocate array for reconstruction fits
        ratio = nan(numDatasets,1);
        for i = 1:numDatasets
            Xp = pca_lunch('TENSOR:prepData',X{i});                 % preprocess data
            % do the reconstruction along different tensor unfoldings of X
            condR2   = pca_lunch('TENSOR:conditionMode',Xp); % condition-mode reconstruction
            neuronR2 = pca_lunch('TENSOR:neuronMode',Xp);    % neuron-mode reconstruction
            % avg. reconstruction fits across conditions
            r2(:,1,i) = mean(condR2,2);
            r2(:,2,i) = mean(neuronR2,2);
            % take the ratio of the reconstruction with the first (best) PC
            % Here it is the log ratio such that a ratio of 1 (equal fits
            % using either basis-neurons or basis-conditions) will be zero.
            ratio(i)  = log(r2(1,1,i) / r2(1,2,i));
                % ratios >0 indicate better fit with basis-conditions (i.e.
                % system exhibits stronger dynamical properties)
        end
        
        % plot
        figure('Color',[1 1 1]);
        % plot trajectories through PC ("neural state") space
        subplot(2,2,1); pca_lunch('PLOT:trajectory',X{1}); title(sprintf('3: input-driven\nSTABLE input'));
        subplot(2,2,3); pca_lunch('PLOT:trajectory',X{2}); title(sprintf('4: input-driven\nDYNAMIC input'));
        % plot a and b weights of each dataset:
        a = [0,0];
        b = [1,1];
        subplot(2,2,2); pca_lunch('PLOT:temporalWeights',a,b);
        % plot reconstruction fits
        subplot(2,2,4); 
        label = {sprintf('\tinput-driven\n\tstable input'),...
            sprintf('\tinput-driven\n\tdynamic input')};
        pca_lunch('PLOT:tensorFitRatios',ratio,label); 
        ylim([-1 1]);
    case 'DO:tensorAnalysisMultiWeights'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % WRAPPER CASE TO DO A SET OF THINGS:
        % Makes temporal dataset X (N x C x T), where X is characterized
        % (to some variable degree) by a dynamic transformation A and
        % time-variable inputs B*U:
        %
        %        x(:,:,t+1) = a*A*x(:,:,t) + b*B*u(:,:,t)
        %
        % where a and b are weights applied to the dynamical and
        % input-driven components of the data. 
        %
        % In this case, we generate temporal data with the same A, B, and
        % U, but we vary the weights a and b.
        %
        % With these datasets, we then apply PCA along the neuron-unfolding
        % and condition-unfolding of the data tensors. 
        %
        % Important to note, a and b range between 0 and 1. The ratio
        % between these two weights might not exactly relate to the degree
        % to which a system exhibits strong/weak dynamics. Larger values of
        % a will indeed mean there are more dynamics, but input-driven
        % effects on the data are stronger than dynamic effects even when
        % a==b. 
        %
        % no inputs, no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % generate initial inputs to system
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        [~,C,T] = size(U);
        N       = size(A,1);
        % assign weights for how input driven & dynamical the system will be:
        a = [0:0.05:1];
        b = fliplr(a);
        % check a and b are same size
        numWeights = length(a);
        if numWeights~=length(b)
            error('a and b are not the same size')
        end
        % Do reconstruction using neuron-unfolding and condition-unfolding
        r2 = nan(min([N,C]),2,length(a)); % preallocate array for reconstruction fits
        ratio = nan(numWeights,1);
        label = {};
        for i = 1:numWeights
            % generate data with the same matrices and inputs, but change
            % the weights. This is important- we are only changing the
            % weights on each loop, NOT the input data, NOT the weight
            % matrix B, and NOT the discrete transformation matrix A.
            X = pca_lunch('SIM:temporalData',U,A,B,a(i),b(i),T); 
            X = pca_lunch('TENSOR:prepData',X);                 % preprocess data
            % do the reconstruction along different tensor unfoldings of X
            condR2   = pca_lunch('TENSOR:conditionMode',X); % condition-mode reconstruction
            neuronR2 = pca_lunch('TENSOR:neuronMode',X);    % neuron-mode reconstruction
            % avg. reconstruction fits across conditions
            r2(:,1,i) = mean(condR2,2);
            r2(:,2,i) = mean(neuronR2,2);
            % take the ratio of the reconstruction with the first (best) PC
            % Here it is the log ratio such that a ratio of 1 (equal fits
            % using either basis-neurons or basis-conditions) will be zero.
            ratio(i)  = log(r2(1,1,i) / r2(1,2,i));
                % ratios >0 indicate better fit with basis-conditions (i.e.
                % system exhibits stronger dynamical properties)
                
            % make some labels for plotting later
            label{end+1} = sprintf('a=%0.2f\n\nb=%0.2f',a(i),b(i));
        end
        
        % plot fits:
        figure('Color',[1 1 1]);
        % plot the weights we used to make the datasets 
        subplot(2,1,1);
        pca_lunch('PLOT:temporalWeights',a,b);
        % plot ratio of fits using first PC only across data made with
        % varying weights
        subplot(2,1,2); pca_lunch('PLOT:tensorFitRatios',ratio); 
        
    
    case '0' % ------------ cases to do pca -------------------------------
    case 'PCA:eig'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Does eignenvalue decomposition for data in matrix X [CxN].
        % Returns eigenvalues and vectors sorted by magnitude of eigenvalue.
        % Also returns PC projections for each point.
        %
        % Eigenvalue decomp. operates on a square matrix, meaning it has 
        % equal number rows and columns.  
        %
        % Eigenvectors are vectors where any transformation applied to them 
        % will result in a pure scaling along that vector. For example, 
        % in the case of a rotational transformation, the eigenvectors can
        % be considered the axes of rotation. The transformation is the
        % square matrix here- we want to find vectors that, when we take
        % our square matrix to and multiply it, just scale the vectors. The
        % amount the vectors scale are the eigenvalues.
        %
        % Eigenvectors are NOT garuanteed to be orthogonal (or independent)
        % from one another. For purposes of PCA, however, we do want
        % orthogonal basis vectors. 
        %
        % To guarnatee orthogonal eigenvectors, we need a square symmetric
        % matrix, meaning the upper-triangular (cells above and to the
        % right of the diagonal cells) and lower-triangular (vice-versa)
        % are mirror images of each other. 
        % See https://math.stackexchange.com/questions/82467/eigenvectors-of-real-symmetric-matrices-are-orthogonal
        % for many proofs of this characteristic (we don't need to know
        % why this is the case here).
        %
        % The covariance matrix of a dataset, calculated by
        % MISC:covariance, is a square symmetric matrix with the variances
        % of each dimension along the diagonal, and the covariances between
        % dimensions on the off-diagonal.
        %
        % Calculating the eigenvectors/values of the covariance matrix of a
        % data set is equivalent to finding the axes that account for the
        % most variance in the original data space. Why? The
        % covariance matrix summarizes the (in-)dependence of each variable 
        % in the data. Since we are attempting to find components that
        % maximize accounted variance, we can simply find the eigenvectors
        % of the covariance matrix, which will reflect where the data is
        % "most susceptible to change" (i.e. variable).
        %
        % For PCA with eig decomposition, the square matrix will be the
        % covariance matrix G.
        % Specifically, G = X*X'/(N-1)
        % - G is the covariance matrix of conditions (in this case) [CxC]
        % - X is our data matrix [CxN]- ** Condition means removed
        %
        % We can rewrite this as G = V*D*V'
        % - D is a diagonal matrix with eigenvalues (decreasing order) 
        % - V is a matrix of eigenvectors (or 'principal axes/components').
        %
        % SVD also performs dimensionality reduction, but it does so on the
        % data matrix directly, not the covariance matrix G. 
        %
        % inputs:
        %       varargin{1}  : X data matrix. PCA is applied across cols.
        %       varargin{2}  : k - scalar of how many PCs to use in
        %                       projection & reconstruction
        %
        % outputs:
        %       varargout{1} : V - right eigenvectors (PCs for col. features)
        %       varargout{2} : D - associated eigenvalues
        %       varargout{3} : Xproj (same size as X) data projected to PCs
        %       varargout{4} : Xrecon(same size as X) data reconstructed
        %                       from PC projections
        %       varargout{5} : varExp - variance explained by each PC
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        X = varargin{1};
        [N,k] = size(X); % k = no. PCs used for reconstruction (default = all)
        if numel(varargin)>1
            % Did user specify number of PCs to be used for reconstruction?
            k = varargin{2}; 
        end
        % --
        Xmu    = sum(X,1)./size(X,1);               % calculate column means (we add these back later to reconsturct data)
        Xc     = X - ones(N,1).*Xmu;                % remove column means
        G      = pca_lunch('MISC:covariance',Xc);   % get covariance of columns
        [V,D]  = eig(G);                            % do eigenvalue decomposition on covariance matrix G
        [D,si] = sort(abs(diag(D)),1,'descend');    % sort eigenvalues on size (larger values = more variance along this vector)
        V      = V(:,si);                           % apply same reordering to eigenvectors
        Xproj  = Xc*V;                              % calculate projection data (Xproj)- in PC space
        Xrecon = Xc*V(:,1:k)*V(:,1:k)' + Xmu;       % calculate reduced data (Xrecon)- in original data space (note V' = V^-1 for eigenvectors)
        varExp = 100*D/sum(D);                      % calculate % of total variance explained by each PC
        % --
        varargout = {V,D,Xproj,Xrecon,varExp}; 
        % check- difference should be very small
        %norm(G*V - V*D)
    case 'PCA:svd'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Does pca via singular value decomposition for columns data in matrix X.
        % Returns eigenvalues and vectors sorted by magnitude of eigenvalue.
        % Also returns PC projections (Xproj) and data reconstructed from
        % PCs (Xred).
        %
        % So what's going on in SVD? (see here for nice video:
        % https://www.youtube.com/watch?v=P5mlg91as1c)
        % With SVD, we try to decompose a rectangular matrix (not square)
        % into three separate matrices.
        %
        %   X = U * S * V'
        %   X : [C x N] data matrix
        %   U : [K x K] left singular vectors- eigenvectors of the rows
        %   S : [K x K] diagonal matrix of singular values (called sigma), sorted in decreasing order
        %   V : [N x K] right singular vectors- eigenvectors of the columns
        %
        % U and V are column orthonormal: U'*U = I = V'*V (i.e. they form a
        % basis)
        %
        % decompose the data matrix into a set of transformation matrices.
        % The intuition here is that we are just definining the data matrix
        % as a series of transformations of basis vectors (eigenvectors).
        % Specifically, X = USV'
        % - X is the data matrix [CxN]- ** Condition mean removed, such
        % that G = X*X'/(N-1)
        % - U is a unitary matrix, meaning U*U' = U'*U = I
        % - S are the singular values- a fancy way to say sqrt(eigenvalues)*(N-1)
        % - V are the eigenvectors
        %
        % So to demonstrate how these methods are the same, we can just 
        % rewrite some equations (assuming X is centred):
        %
        % second moment       G = V * D * V'
        %                       = X * X'
        %                       = (V*S*U' * U*S*V') / (N-1)
        %
        % eigenvectors - recall U is unitary (U'*U = U*U' = I),
        %                     D = S*U' * U*S /(N-1) 
        %                       = S*I*S / (N-1)
        %                       = S^2 / (N-1) (up to a reasonable precision)
        %  
        % So we can appreciate pca with svd or eig do the same thing.
        %
        % SVD approach is more common in many cases, and is the default in
        % the pca matlab function. I think the reasoning behind this is
        % that the SVD algorthim is more stable than that used in eig.
        % However, arguably the GREATEST strength of SVD for PCA is that
        % you get eigenvectors for both dimensions with SVD (U and V),
        % whereas you only get V with eig-decomposition.
        %
        % inputs:
        %       varargin{1}  : X data matrix. PCA is applied across cols.
        %       varargin{2}  : k - scalar of how many PCs to use in
        %                       projection & reconstruction
        %
        % outputs:
        %       varargout{1} : V - right eigenvectors (PCs for col. features)
        %       varargout{2} : D - associated eigenvalues
        %       varargout{3} : Xproj (same size as X) data projected to PCs
        %       varargout{4} : Xrecon(same size as X) data reconstructed
        %                       from PC projections
        %       varargout{5} : varExp - variance explained by each PC
        %       varargout{6} : U - left eigenvectors (PCs for row features)
        %       varargout{7} : S - singular values
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        X = varargin{1};
        [N,k] = size(X); % k = no. PCs used for reconstruction (default = all)
        k     = min([N,k]);
        if numel(varargin)>1
            k = varargin{2}; % number of PCs used for reconstruction
        end
        % --
        Xmu     = sum(X,1)./size(X,1);              % calculate column means (we add these back later to reconsturct data)
        Xc      = X - ones(N,1).*Xmu;               % remove column means
        [U,S,V] = svd(Xc,'econ');                   % do svd
        D       = S.^2 / (N-1);                     % calculate eigenvalues
        % sort the eigvenvals according to magnitude (this corresponds
        % with their significance- large eigenvals account for larger
        % data variance)
        [D,si] = sort(abs(diag(D)),1,'descend');    % sort eigenvalues on size (larger values = more variance along this vector)
        V      = V(:,si);                           % apply same reordering to eigenvectors
        U      = U(:,si);
        Xproj  = bsxfun(@times,U,diag(S)');         % calculate projection data (Xproj)- data projected to PCs. Equal to Xc*V;
        Xrecon = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'+Xmu; % calculate reduced data (Xrecon)- in original data space (note V' = V^-1 for eigenvectors)
        varExp = 100*D/sum(D);                      % calculate % of total variance explained by each PC
        % --
        varargout = {V,D,Xproj,Xrecon,varExp,U,S};
    
    case '0' % ------------ cases to simulate patterns, no temporal stuff -
    case 'SIM:mvnrndExact'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generates multivariate normal random data with 
        % a sample second moment matrix G=(U'*U)/N, which is EXACTLY G.
        % I.e. this case produces noiseless data from some model G.
        % Adapted form function JDierichsen's mvnrnd_exact.
        %
        % inputs:
        %       varargin{1} : G - [CxC] symmetric covariance matrix
        %       varargin{2} : N - number of measurement channels (neurons/voxels/etc.)
        %       varargin{3} : sigma - scalar specifying std of measurement
        %                               channels (one should likely use a
        %                               value of 1)
        %
        % outputs:
        %       varargout{1} : U - [NxC] data matrix, where N=channels and C = variables
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % get inputs
        verbose = 0;
        G     = varargin{1}; % desired second moment matrix of patterns we will generate
        N     = varargin{2}; % no. of measurement channels
        sigma = varargin{3}; % standard deviation of measurement channels
        
        K = size(G,1);          % no. conditions/variables
        U = normrnd(0,1,K,N);   % make K normal random patterns with mean=0 & std=1
        % Make some random patterns, U. These patterns will likely have some 
        % covariance. Since we want these patterns to have the (co)variance 
        % structure defined in G, we first need to make these patterns 
        % orthogonal (otherwise, the returned patterns will not be exactly G).
        % The orthogonalized patterns will be Z.
        E = (U*U');             % random pattern second moment (off-diag values are covariances b/t random patterns)
        Z = E^(-0.5)*U;         % force random patterns to be orthonormal by dividing out sqrt(covariance structure)
        Z = Z - (ones(1,N) .*(sum(Z,2)./N)); 
        
        % Now, we force these random, orthogonal patterns to have second
        % moment of exactly G.
        % First, we need to ensure G is not poorly defined (i.e. at the 
        % very least it is positive semidefinite so it can be safely
        % decomposed as G = A*A').
        % see here for more info: http://www2.gsu.edu/~mkteer/npdmatri.html
        if det(G)<0 || rcond(G)==0
            if verbose
                disp(G)
                disp('The above covariance matrix is not positive semidefinite (ill-conditioned).');
                disp('This means the covariance structure is likely not plausible.');
                disp('To counter this problem, a small regularation has been applied to the diagonal elements');
            end
            G = G + eye(K)*0.1;
            % double check this fixed the issue
            if det(G)<0 || rcond(G)==0
                error('Small regularization did not work. Try either increasing regularization, or re-considering this covariance matrix entirely.');
            end
        end
        
        % Now, decompose G (which is variance co-variances) into A*A' (so
        % we can map magnitude and correlations between condition/variable
        % patterns
        A = cholcov(G);         % safe decomposition of G into A*A'
        if (size(A,1)>N)
            error('Not enough columns to represent G'); 
        end
        if size(A,1)==1
            A = diag(A);
        end
        % Make patterns, U, that have a second moment that matches exactly 
        % G (or super close to G) 
        U = A'*Z(1:size(A,1),:)*sqrt(N).*sigma; 
        
        varargout = {U'}; % reorient U to be [NxC] so that things match with textbook definitions of methods for PCA
    case 'SIM:mvnrndNoise'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % simulate data with SIM:mvnrndExact, then add some white noise.
        % You can scale the signal-to-noise levels by using signalVar and
        % noiseVar.
        %
        % NOTE: the second moment here will not be equal in magnitude to
        % that used to first generate patterns. The variances will be
        % scaled by the signal and noise variance factors.
        %
        % NOTE: do not use this case when also changing the sigma of the
        % measurement channels- this will make moot the values of the
        % signalVar and noiseVar. Therefore, I do not allow you to specify
        % noise sigma.
        % inputs:
        %       varargin{1} : G - [CxC] symmetric covariance matrix
        %       varargin{2} : N - number of measurement channels (neurons/voxels/etc.)
        %       varargin{3} : sigma - scalar specifying std of measurement
        %                               channels (one should likely use a
        %                               value of 1)
        %
        % outputs:
        %       varargout{1} : U - [NxC] inputted data matrix with added noise
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        % get inputs
        U         = varargin{1}; % noiseless patterns we will add noise to
        signalVar = varargin{2}; % signal variance (1 scalar or K*N scalars)
        noiseVar  = varargin{3}; % noise variance (1 scalar or K*N scalars)
        U      = U';                               % reorient U to be [CxN] (just for this case)
        [K,N]  = size(U);                          % no. conditions and channels
        E      = normrnd(0,1,K,N);                 % make K normal random patterns with mean=0 & std=1
        Us     = bsxfun(@times,U,sqrt(signalVar)); % scale true activity by signal scaling factor(s)
        Es     = bsxfun(@times,E,sqrt(noiseVar));  % scale noise by noise scaling factor(s)
        X      = Us + Es;                          % Now add the random noise
        varargout = {X'};                          % revert data matrix to original orientations
    case 'SIM:sparse'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generate data where each channel is only tuned to one condition.
        %   X = pca_lunch('SIM:sparse',G,sigma);
        %   G*     : specified second moment (optional- if not specified, will use default G)
        %   sigma* : standard deviation of each measurement channel (default = 1)
        %   *optional argument
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if numel(varargin)==0
            % user did not pass through specified G, so run default G
            if numC==2
                G(:,:,1) = diag([var(1),0]);
                G(:,:,2) = diag([0,var(2)]);
            elseif numC==3
                G(:,:,1) = blkdiag(var(1),zeros(2));
                G(:,:,2) = blkdiag(0,var(2),0);
                G(:,:,3) = blkdiag(zeros(2),var(3));
            end
        elseif numel(varargin)==1
            % user passed through specified G. Assume it's size numC in 3rd
            % dimension.
            G    = varargin{1};
            numC = size(G,3);
            if numC~=size(G(:,:,1),1)
                error('G for sparse patterns needs one G per condition. Specified G should thus have one G per condition')
            end
        end
        % did user specify standard deviation of measurement channels?
        if numel(varargin)==2
            sigma = varargin{2};
        end
        X = []; % I'm not preallocating space here b/c indexing is easier without it
        for i = 1:numC
            % Here, G is positive semidefinite, but cannot be inverted.
            % This is because we are forcing patterns to be sparsely tuned 
            % (so we have one defined value in a covariance matrix full 
            % of zeros). 
            % This matrix will have a determinant equal to zero and
            % no negative eigenvalues. 
            % BUT, this G is not invertable. This could be a problem for
            % making nice random patterns for our sprase simulation.
            % However, we can be safely decompose G into A*A' with
            % cholcov(G)- see case 'SIM:mvnrndExact'.
            x = pca_lunch('SIM:mvnrndExact',G(:,:,i).*numC,N/numC,sigma);
            X = [X; x]; 
        end
        varargout = {X};
    case 'SIM:uncorrelated'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generate data where each channel's response profile is 
        % independent of its tuning for other conditions (they are uncorrelated).
        %   X = pca_lunch('SIM:uncorrelated',sigma);
        %   sigma* : standard deviation of each measurement channel (default = 1)
        %   *optional argument
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if numel(varargin)==1
            sigma = varargin{1};
        end
        G = diag(var);
        X = pca_lunch('SIM:mvnrndExact',G,N,sigma);
        varargout = {X};
    case 'SIM:correlated'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generate data where each channel can be tuned to all conditions,
        % and there is some correlation between conditions.
        % For simplicity, all condition pairs will share the same
        % covariance.
        %   X = pca_lunch('SIM:correlated',sigma);
        %   sigma* : standard deviation of each measurement channel (default = 1)
        %   *optional argument
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if numel(varargin)==1
            sigma = varargin{1};
        end
        G = diag(var);
        G = G + (ones(numC)-eye(numC)).*covar;
        X = pca_lunch('SIM:mvnrndExact',G,N,sigma);
        varargout = {X};
    case 'SIM:pcDist'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % pca_lunch('SIM:pcDist',type,numIters,addNoise,signalVar,noiseVar);
        % Case to iteratively generate data from same G and calculate PCs. 
        % Each iteration, PCs are saved.
        % Returns generated PCs.
        % Returned PCs can be used to plot a distribution cloud of PC
        % directions
        %
        % inputs:
        %       varargin{1}  : data model type ('sparse','uncorrelated','correlated')
        %       varargin{2}  : numIters - number of simulated datasets for
        %                        this model (optional)
        %       varargin{3}  : normalize - normalize condition data by its
        %                        range
        %       varargin{4}  : addNoise flag - 0 (don't add noise to data,
        %                        default) or 1 (add noise to data)
        %       varargin{5}  : signalVar (include if addNoise is 1)- best
        %                        to keep as 1
        %       varargin{6}  ; noiseVar (include if addNoise is 1)- you can
        %                       scale this value. Best not to scale both 
        %                       signalVar and noiseVar
        %
        % outputs:
        %       varargout{1} : V - [2+ x numIters] matrix of eigenvectors.
        %                       Values in row 1 are the x coord, and vals 
        %                       in row 2 are the y coord for the 1st PC. 
        %                       Vice versa for subsequent rows (row 3 = x of PC 2)
        %       varargout{2} : D - [1+ x numIters] matrix of associated eigenvalues. 
        %                           (row 1 = 1st PC, row 2 = 2nd PC, etc.)
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        type      = varargin{1}; % 'sparse', 'uncorrelated', or 'correlated'
        addNoise  = 0;           % add noise to data
        normalize = 0;
        if numel(varargin)>1
            numIters = varargin{2};
        end
        if numel(varargin)>2
            normalize = varargin{3};
        end
        if numel(varargin)>3
            addNoise = varargin{4};
        end
        % save function handle to be called on each iteration (save time)
        switch type
            case 'sparse'
                Xfcn = 'pca_lunch(''SIM:sparse'')';
            case 'uncorrelated'
                Xfcn = 'pca_lunch(''SIM:uncorrelated'')';
            case 'correlated'
                Xfcn = 'pca_lunch(''SIM:correlated'')';
        end
        if addNoise
            signalVar = 10;
            noiseVar  = 1;
            % check if user passed through different signalVar and noiseVar
            % values
            if numel(varargin)>3
                signalVar = varargin{5};
                try
                   noiseVar = varargin{6};
                catch
                end
            end
            Xfcn = ['pca_lunch(''SIM:mvnrndNoise'',' Xfcn ',signalVar,noiseVar)'];
        end
        if normalize
            Xfcn = ['pca_lunch(''MISC:normalize'',' Xfcn ')'];
        end
        V = nan(numC^2,numIters);
        D = nan(numC,numIters);
        for i = 1:numIters
            Xi     = eval(Xfcn);
            [v,d]  = pca_lunch('PCA:eig',Xi);
            V(:,i) = v(:);
            D(:,i) = d;
        end
        varargout = {V,D};     
    case 'SIM:transformA'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generates a DISCRETE transformation matrix, A.
        % A had dimensions [NxN], and is constructed as:
        %       A = W*OM*W'
        % where OM is a feature matrix, and W are measurement channel
        % weights that are applied to each feature.
        %
        % inputs:
        %       varargin{1} : N (scalar), # of measurement channels                       (generate with 'SIM:mvnrndExact')
        %       varargin{2} : dD (scalar), # of features. This is the
        %                       intrinsic dimensionality of the transform 
        %                       at each timestep.
        % output:
        %       varargout{1} : A - [NxN] discrete transformation matrix
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        N  = varargin{1}; % number of measurement channels
        dD = varargin{2}; % intrinsic dimensionality of transformation at each timestep
        A = randn(N,dD)*randn(dD,N)+10*eye(N);
        A = real(orth(A)^(1/10));       % make components orthogonal (and det==1)
        varargout = {A};
    case 'SIM:temporalDataPrep'
        % returns some matrices that we can pass through to generate 
        % temporal data with 'SIM:temporalData'
        % generate initial inputs to system
        N  = 10;                                  % no. measurement channels ("neurons")
        T  = 50;                                  % no. timepoints
        dS = 5;                                   % intrinsic dimensionality of the input signal (here this is akin to the number of conditions, but not always true)
        dD = floor(N/2);                          % intrinsic dimensionality of the dynamics
        % make input signals that vary at each timestep (this variation is not in a structured manner)
        U = nan(dS,dS,T);
        for t = 1:T
            U(:,:,t) = pca_lunch('SIM:mvnrndExact',eye(dS),dS,1); % generate exactly independent data - these are the input signals U
        end
        % define the discrete linear transform matrix A (randomly define)
        A = pca_lunch('SIM:transformA',N,dD);
        % define the input to measurement channel weighting matrix
        B = randn(N,dS);
        varargout = {U,A,B};
    case 'SIM:temporalData'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Generates temporal data for a population of measurement channels.
        % Since this example is more complex, it's more realistic to use
        % more than 2 conditions. Specifically, this case is used to
        % examine populations that either encode external variables
        % (inputs), or execute some process that is well-captured by a
        % discrete linear transformation (autonomous linear-dynamical system).
        %
        % Given initial state data X (an NxC data matrix), this case
        % returns X (NxCxT), where T is number of timepoints.
        %
        % To estimate temporal data:
        %   x(:,:,t+1) = a*A*x(:,:,t) + b*B*u(:,:,t)
        % 
        % where the next state (Xt+1) is a function of a dynamical
        % transofrmation (A*Xt) plus the tuning (B) to the initial input
        % (u). a and b are weights that can be altered by user input to
        % influence how "dynamical" (a) or "encody" (b) a system is.
        % 
        % Much of this work is inspired and explained by Seely et al.
        % (2016) PLoS Comp.Biol.
        %       https://doi.org/10.1371/journal.pcbi.1005164
        %
        % inputs:
        %       varargin{1} : U data matrix [NxC], initial input to system 
        %                       (generate with 'SIM:mvnrndExact')
        %       varargin{2} : A, discrete transformation matrix applied to
        %                       each timestep. Effect at each timestep is 
        %                       weighted by a.
        %                       Can be generated using 'SIM:transformA'
        %       varargin{3} : B, input weighting matrix [NxC]. Maps U onto X.
        %                       Effect at each timestep is weighted by b.
        %       varargin{4} : a, scalar that defines how "dynamical" the
        %                       system is (a>=0)
        %       varargin{5} : b, scalar that defines how "encody" the
        %                       system is (b>=0)
        %       varargin{6} : T, number of timepoints to generate data for
        %                       (optional) default = 100 
        % output:
        %       varargout{1} : X - [NxCxT] data matrix
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        % GET INPUTS
        T = 100;         % 100 timepoints (default)
        U = varargin{1}; % initial state
        A = varargin{2}; % discrete linear transformation matrix applied to each timestep
        B = varargin{3}; % input to neuron mapping matrix (specifies tuning of each neuron to each input)
        a = varargin{4}; % 0:1 scalar that defines how "dynamical" the system is
        b = varargin{5}; % 0:1 scalar that defines how "encody" the system is
        if numel(varargin)>5
            % did user explicitly define number of timepoints?
            T = varargin{6};
        end
        
        % HOUSEKEEPING
        [N,C] = size(B);
        t     = size(U,3);     % get number of neurons and components
        % check if input signals vary over time:
        if t==1
            % input signals are constant across time
            U = repmat(U,1,1,T);
        elseif t~=1 && t~=T
            % input signals change at each timepoint but user did not
            % specify input for all timesteps
            error('Input signals change at each timestep, but inputs for all timesteps are not defined ->  T~=(U,3)')
        end
        
        % MAKE DATA
        X        = nan(N,C,T);  % preallocate data tensor
        X(:,:,1) = B*U(:,:,1);  % assign input to first temporal state
        % * Note in the above line, weight b is not applied because system 
        % needs some sort of input to start a process (i.e. dynamic process
        % needs input to start)
        for t = 1:T-1
            % generate data for each timestep
            X(:,:,t+1) = a*A*X(:,:,t) + b*B*U(:,:,t);
            if any(isnan(X(:,:,t+1)))
                warning('nans in data: check A, B, and U')
                keyboard
            end
        end
        varargout = {X};
    case 'SIM:autoDynamics'
        A  = varargin{1};
        X1 = varargin{2};
        T  = varargin{3};
        
        [N,C] = size(X1);
        X = nan(N,C,T);
        X(:,:,1) = X1;
        for t = 1:T-1 
            X(:,:,t+1) = A*X(:,:,t) + X(:,:,t);
        end
        varargout = {X};
        
    case '0' % ------------ tensor analyses of temporal data --------------       
    case 'TENSOR:prepData'
        % Case applies firing rate normalization and removes
        % cross-condition mean from each neuron for each timepoint.
        X = varargin{1}; % [N x C x T] data matrix
        % 1. housekeeping
        [numNeurons,numConds,numTime] = size(X);
        % 2. soft normalize firing rates to ensure analysis is not
        % dominated by a few loud neurons
        softFactor  = 5;
        F           = permute(X,[3,2,1]);
        F           = reshape(F,numTime*numConds,numNeurons); % append conditions vertically [CT x N]
        ranges      = range(F);  % For each neuron, the firing rate range across all conditions and times.
        normFactors = (ranges+softFactor);
        F = bsxfun(@times, F, 1./normFactors);  % normalize
        F = reshape(F,numTime,numConds,numNeurons);
        X = permute(F,[3,2,1]); % return to original dimensions: [NxCxT]
        % 3. remove cross-condition mean [1 x T] from each neurons response
        meanX = mean(X,2);
        X     = X - repmat(meanX,1,numConds,1); % check with mean(X,2)
        varargout = {X};
    case 'TENSOR:neuronMode'
        X = varargin{1}; % [N x C x T] data matrix
        % housekeeping
        [numNeurons,numConds,numTime] = size(X);
        % reshape data tensor into condition unfolding:
        % - [N x CT] : neurons are rows - here we find neuron-mode (Nm) ***
        % - [C x NT] : conditions are rows- here we find condition-mode (Cm) 
        Nm = reshape(X,numNeurons,numConds*numTime);
        
        % do PCA:
        [U,S,V] = svd(Nm,'econ');
        [~,si]  = sort(abs(diag(S.^2)),1,'descend');   % sort eigenvalues on size
        V       = V(:,si);                             % apply same reordering to eigenvectors
        U       = U(:,si);
        
        % Now, do reconstruction with basis-conditions
        R2 = []; % preallocate R2 to describe condition reconstruction fits
        % precalculate total sums of squares across conditions for R2 calc.
        tss = permute(X,[2,1,3]); 
        tss = reshape(tss,numConds,numNeurons*numTime);
        tss = diag(tss*tss')';
        % loop through possible sizes of k (k=num latent components, max is
        % the minimum size of numNeurons & numConds)
        Xr = {};
        for k = 1:min([numConds,numNeurons])
            % neuron-mode reconstruction
            Xr{k} = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; 
            Xr{k} = reshape(Xr{k},numNeurons,numConds,numTime); % reshape output so it is [NxCxT]
            % calculate reconstruction error per condition with basis-neurons
            res = X - Xr{k};                                   % diff b/t reconstruction and original unfolded neuron data
            res = permute(res,[2,1,3]);                        % permute error matrix so it is [C x N x T]
            res = reshape(res,numConds,numNeurons*numTime);    % reshape error matrix so it is [C x NT]
            rss = diag(res*res')';                             % calculate error per condition over time
            % compute R2 per condition
            R2(k,:) = 1-rss./tss;
        end
        varargout = {R2,Xr};   
    case 'TENSOR:conditionMode'
        X = varargin{1}; % [N x C x T] data matrix
        % housekeeping
        [numNeurons,numConds,numTime] = size(X);
        % reshape data tensor into condition unfolding:
        % - [N x CT] : neurons are rows - here we find neuron-mode (Nm)
        % - [C x NT] : conditions are rows- here we find condition-mode (Cm) ***
        Cm = permute(X,[2,1,3]);
        Cm = reshape(Cm,numConds,numNeurons*numTime);
        
        % do PCA:
        [U,S,V] = svd(Cm,'econ');
        [~,si]  = sort(abs(diag(S.^2)),1,'descend');   % sort eigenvalues on size
        V      = V(:,si);                              % apply same reordering to eigenvectors
        U      = U(:,si);
        
        % Now, do reconstruction with basis-conditions
        R2 = []; % preallocate R2 to describe condition reconstruction fits
        % precalculate total sums of squares across conditions for R2 calc.
        tss = permute(X,[2,1,3]); 
        tss = reshape(tss,numConds,numNeurons*numTime);
        tss = diag(tss*tss')';
        % loop through possible sizes of k (k=num latent components, max is
        % the minimum size of numNeurons & numConds)
        Xr = {};
        for k = 1:min([numConds,numNeurons])
            % condition-mode reconstruction
            %Xproj = bsxfun(@times,U,diag(S)');
            Xr{k} = U(:,1:k)*S(1:k,1:k)*V(:,1:k)'; 
            Xr{k} = reshape(Xr{k},numConds,numNeurons,numTime);
            Xr{k} = permute(Xr{k},[2,1,3]);                  % reshape output so it is [NxCxT]
            % calculate reconstruction error per condition with
            % basis-conditions
            res = X - Xr{k};                                 % diff b/t reconstruction and original unfolded neuron data
            res = permute(res,[2,1,3]);                      % permute error matrix so it is [C x N x T]
            res = reshape(res,numConds,numNeurons*numTime);  % reshape error matrix so it is [C x NT]
            rss = diag(res*res')';                           % calculate error per condition over time
            % compute R2 per condition
            R2(k,:) = 1-rss./tss;
        end
        varargout = {R2,Xr};          
  
    case '0' % ------------ miscellaneous cases for functionality ---------
    case 'MISC:covariance'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Calculate covariance (G) across columns for data matrix X. 
        % Function 'cov' does this exactly, but it's nice 
        % to have explicit descriptions.  
        % 
        %   G = [ cov(x1,x1)  cov(x2,x1)
        %         cov(x1,x2)  cov(x2,x2) ]
        %
        %   cov(x1,x1) = var(x1) = sum((x1 - x1_mean)^2)
        %
        %   cov(x1,x2) = sum((x1 - x1_mean)*(x2 - x2_mean)) = cov(x2,x1)
        %
        % Steps:
        % 1. remove means from each condition/variable
        % 2. calculate covariance of each dimension with itself (variances)
        % 3. calculate covariances between dimensions
        % 
        % Mean-removed data (or "centred" data) reflects the euclidean
        % distance from the corresponding dimension's mean.
        % The normalized (see below) inner product of the centred data
        % yeilds the covariance matrix.
        %
        % We want an unbiased estimator of the true covariance matrix (G), 
        % which is normalized by N (number of observations/neurons/whatever).
        % Since we have sampled data from a normal distribution,
        % the best unbiased estimate of the covariance matrix is normalized 
        % by N-1:
        %
        %       var = (X-Xmean)(X-Xmean) / (n-1)
        %
        % We can appreciate that Pearson's r between variables can be
        % "read-out" from the covariance matrix:
        %       
        %       r = var(x1)*var(x2) / sqrt(cov(x1,x2)*cov(x2,x1))
        %
        % inputs:
        %       varargin{1}  : X data matrix. covariance is estimated across
        %                      columns
        % output:
        %       varargout{1} : G covariance matrix
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        X = varargin{1};
        N = size(X,1);                       % no. of rows (we integrate over these)
        X = X - (ones(N,1) .*(sum(X,1)./N)); % remove column means
        G = (X'*X) ./ (N-1);                 % calculate (co)variances across columns
        varargout = {G};
    case 'MISC:normalize'
        X = varargin{1};
        ranges  = range(X); % range of each channel per condition
        Xn      = bsxfun(@times,X,1./ranges); % scale all channels by their range
        varargout = {Xn};
    case 'MISC:enforceSign'
        % Currently not used to ensure compatability across all versions of
        % MATLAB.
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % quick case to enforce sign convention (MATLAB does this in pca):
        % Convention is that largest element in each column will be +
        % This does not conceptually change anything, since these basis
        % vectors can be scaled in either direction (pos/neg) to
        % accommodate either sign.
        %
        % !! some functionality -'linear' flag for max- not available in Matlab versions pre-2019a
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        V = varargin{1}; % eigenvectors
        mv = max(abs(V),[],1);                      % find max of absolute values in each col.
        [~,ms] = max(abs(V)==mv,[],1,'linear');     % ensure we find only one index per column (controlling for identical values in col.)
        ms = sign(V(ms));                           % get sign of the max value in each col.
        ms = ones(size(V,1)) .* ms;                 % make matrix of same max sign per col.
        % apply sign matrix to V (negative max value becomes pos, vice 
        % versa, and other vals change accordingly)
        V  = V .* ms;                   
        varargout = {V};
    case 'MISC:getDirectory'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Utility case to return the directory where pca_lunch is saved
        %
        % outputs:
        %   varargout{1} : path to folder where pca_lunch is saved
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        d = which('pca_lunch');
        d = fileparts(d);
        varargout = {d};
    case 'MISC:calcTrajectory'
        % case to calculate tangling of trajectory in PC/neural state space
        X = varargin{1};  % data matrix [NxCxT]
        % do PCA on condition-unfolding
        [N,C,T] = size(X);
        X = pca_lunch('TENSOR:prepData',X);
        X = reshape(X,N,C*T)'; % [CTxN]
        [U,S]   = svd(X,'econ');
        % find enough PCs to account for 95% of variance in data:
        varExplained = diag(S.^2);
        varExplained = varExplained./sum(varExplained).*100;
        %[~,K]        = min(abs(varExplained-0.95));
        K  = 3;
        % project data using these PCs
        Xtraj = U(:,1:K)*S(1:K,1:K);
        % reshape projected data into KxCxT
        Xtraj = reshape(Xtraj',K,C,T);
        varargout = {Xtraj,varExplained};
        
    case '0' % ------------ cases for plotting things ---------------------    
    case 'PLOT:dataScatter'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Wrapper to do consistent scatter plots of the simulated datasets.
        %
        % inputs:
        %       varargin{1} : X - data matrix. Each column is one axis in
        %                           plot, and values along rows are the 
        %                           markers in the plot.
        %
        % no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        X    = varargin{1}; % data set
        numC = size(X,2);
        % line colours can be passed as second option
        if numel(varargin) < 2
            clr = [0 0 0];
        else
            clr = varargin{2};
        end
        if ~iscell(clr)
            clr = {clr};
        end
        if numel(clr)<numC 
            % plotting everything in same colour
            for i=2:numC
                clr{i} = clr{1};
            end
        end
        % plot data as scatterplot
        if numC==2
            % plot data points
            scatter(X(:,1),X(:,2),...
                'MarkerFaceColor',data_face,...
                'MarkerEdgeColor',data_edge,...
                'MarkerFaceAlpha',alpha_face,...
                'MarkerEdgeAlpha',alpha_edge);
            axis equal
        elseif numC==3
            % plot data points
            scatter3(X(:,1),X(:,2),X(:,3),...
                'MarkerFaceColor',data_face,...
                'MarkerEdgeColor',data_edge,...
                'MarkerFaceAlpha',alpha_face,...
                'MarkerEdgeAlpha',alpha_edge);
            axis equal
        end
        xlabel('dim1');
        ylabel('dim2');
        if numC==3
           zlabel('dim3'); 
        end
    case 'PLOT:pcLines'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Plot scaled eigenvectors into a scatterplot of the original data
        % Eigenvectors (PCs) are scaled by the eigenvalues, and so the 
        % length of the plotted eigenvector lines reflects the variance
        % that is accounted for by each the corresponding component.
        % The values are plotted form the origin [0,0].
        % 
        % inputs:
        %       varargin{1} : V - eigenvectors
        %       varargin{2} : D - eigenvalues
        %       varargin{3} : clr - cell or cell array of colours for lines (optional)
        %
        % no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        V = varargin{1};
        D = varargin{2};
        % line colours can be passed as option
        if numel(varargin) < 3
            clr = pc_clr;
        else
            clr = varargin{3};
        end
        if ~iscell(clr)
            clr = {clr};
        end
        if numel(clr)<numC 
            % plotting everything in same colour
            for i=2:numC
                clr{i} = clr{1};
            end
        end
        % plot line for each PC, scaled by eigenvalue (variance along
        % dimension)
        hold on
        for i = 1:numC
            if numC==2
                line([0;V(1,i)*sqrt(D(i))],[0;V(2,i)*sqrt(D(i))],...
                    'Color',clr{i},'LineWidth',pcLineWidth);
            elseif numC==3
                line([0;V(1,i)*sqrt(D(i))],[0;V(2,i)*sqrt(D(i))],[0;V(3,i)*sqrt(D(i))],...
                    'Color',clr{i},'LineWidth',pcLineWidth);
            end
        end
    case 'PLOT:drawlines'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Draws axes lines (one per dimension), which will intersect at
        % origin.
        % 
        % inputs:
        %       varargin{1} : number of axis lines to plot
        %       varargin{2} : colour of axis lines (optional, can be cell array of colours or just one colour)     
        %
        % no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        numC = varargin{1}; 
        % line colours?
        if numel(varargin) < 2
            clr = ax_clr;
        else
            clr = varargin{2};
        end
        if ~iscell(clr)
            clr = {clr};
        end
        if numel(clr)<numC 
            % plotting everything in same colour
            for i=2:numC
                clr{i} = clr{1};
            end
        end
        % get limits (so lines span full plotting space)
        xlims = xlim;
        ylims = ylim;
        hold on
        if numC==1
            % draw line along X-axis only
            line(xlims',[0;0],'Color',clr{1},'LineWidth',ax_width);
        elseif numC==2
            % draw lines
            line(xlims',[0;0],'Color',clr{1},'LineWidth',ax_width);
            line([0;0],ylims','Color',clr{2},'LineWidth',ax_width);
        elseif numC==3
            % get limits (so lines span full plotting space)
            zlims = zlim;
            % draw lines
            line(xlims',[0;0],[0;0],'Color',clr{1},'LineWidth',ax_width);
            line([0;0],ylims',[0;0],'Color',clr{2},'LineWidth',ax_width);
            line([0;0],[0;0],zlims','Color',clr{3},'LineWidth',ax_width);
        end
        hold off  
    case 'PLOT:pcDist'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Creates a heatmap of the first estimated PCs for multiple
        % datasets.
        % Requires the eigenvectors and eigenvalues for 1+ datasets.
        % The inputs can be generated from SIM:pcDist (where data is
        % simulated using the same model each time). 
        %
        % inputs:
        %       varargout{1} : V - [2+ x numIters] matrix of eigenvectors.
        %                       Values in row 1 are the x coord, and vals 
        %                       in row 2 are the y coord for the 1st PC. 
        %                       Vice versa for subsequent rows (row 3 = x of PC 2)
        %       varargout{2} : D - [1+ x numIters] matrix of associated eigenvalues. 
        %                           (row 1 = 1st PC, row 2 = 2nd PC, etc.)
        %
        % no outputs
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        V = varargin{1};                    % first output of SIM:pcDist ([eigenvectors x simulation number])
        D = varargin{2};                    % second output of SIM:pcDist ([eigenvalues x simulation number])
        V     = V(1:2,:);                   % take x-y coords of first PCs
        D     = D(1,:);                     % take corresponding eigenvalue of first PC
        Vs    = V.*repmat(sqrt(D),2,1);     % scale eigenvectors by sqrt(eigenvalue)
        Vs    = [Vs,-Vs];                   % flip the signs to make density plot a full circle (this is fine since scaling pos/neg is arbitrary)
        % Bin the data:
        lim   = max(max(abs(Vs)));
        lim   = lim + 0.25*lim;
        pts   = linspace(-lim,lim,101);               % make bins
        N     = histcounts2(Vs(1,:),Vs(2,:),pts,pts); % count values per bin
        % Plot heatmap:
        imagesc(pts, pts, N');
        axis equal;
        set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]), 'YDir', 'normal');
        pca_lunch('PLOT:drawlines',2,ax_clr);
        xlabel('dim1');
        ylabel('dim2');
        colormap(flipud(hot));
    case 'PLOT:doAllPlots'    
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Plotting wrapper for the DO:allX cases
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        X = varargin{1};        % cell array of data matrices (1 per dataset)
        V = varargin{2};        % cell array of eigenvectors
        D = varargin{3};        % cell array of eigenvalues
        Xproj = varargin{4};    % cell array of projected data
        Xrecon = varargin{5};   % cell array of reconstructed data
        names  = varargin{6};   % cell array of dataset names
        
        % plot
        figure('Color',[1 1 1]);
        numCol = numel(X);
        numRow = 4;
        
        % FIRST ROW (image the covariance matrix)
        j = 1; % subplot counter
        for i= 1:numCol
            G = pca_lunch('MISC:covariance',X{i});
            subplot(numRow,numCol,j);
            imagesc(G);
            %c = max(max(G));
            %caxis([0 c]);
            title(sprintf('%s covariance',names{i}));
            j = j+1;
        end
        
        % SECOND ROW (scatterplot of raw data sets)
        for i= 1:numCol
            subplot(numRow,numCol,j);
            pca_lunch('PLOT:dataScatter',X{i});
            pca_lunch('PLOT:drawlines',numC);
            pca_lunch('PLOT:pcLines',V{i},D{i},pc_clr);
            title(sprintf('%s responses',names{i}));
            j = j+1;
        end
        
        % THIRD ROW (scatterplot of data sets projected to PC space)
        for i= 1:numCol
            subplot(numRow,numCol,j);
            pca_lunch('PLOT:dataScatter',Xproj{i});
            pca_lunch('PLOT:drawlines',numC,pc_clr);
            title('projected');
            xlabel('pc1');ylabel('pc2');zlabel('pc3');
            j = j+1;
        end
        
        % THIRD ROW (scatterplot of reconstructed data)
        for i= 1:numCol
            subplot(numRow,numCol,j);
            pca_lunch('PLOT:dataScatter',Xrecon{i});
            pca_lunch('PLOT:drawlines',numC);
            title('reconstructed data');
            j = j+1;
        end
    case 'PLOT:doCorrPlots'  
         % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Plotting wrapper for the DO:corrX cases
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        X = varargin{1};        % 1 data matrix 
        V = varargin{2};        % eigenvectors
        D = varargin{3};        % eigenvalues
        Xproj = varargin{4};    % projected data
        Xrecon = varargin{5};   % reconstructed data
        
        % plot
        figure('Color',[1 1 1]);
        
        % image the covariance matrix
        G = pca_lunch('MISC:covariance',X);
        subplot(2,2,1);
        imagesc(G);
        %c = max(max(G));
        %caxis([0 c]);
        title(sprintf('covariance/nmatrix'));
        
        % scatterplot of raw data
        subplot(2,2,2);
        pca_lunch('PLOT:dataScatter',X);
        pca_lunch('PLOT:drawlines',numC);
        pca_lunch('PLOT:pcLines',V,D,pc_clr);
        title(sprintf('correlated\nresponses'));
        
        % scatterplot of data sets projected to PC space
        subplot(2,2,3);
        pca_lunch('PLOT:dataScatter',Xproj);
        pca_lunch('PLOT:drawlines',numC,pc_clr);
        title('projected');
        xlabel('pc1');ylabel('pc2');zlabel('pc3');
        
        % scatterplot of reconstructed data
        subplot(2,2,4);
        pca_lunch('PLOT:dataScatter',Xrecon);
        pca_lunch('PLOT:drawlines',numC);
        title('reconstructed data');
    case 'PLOT:trajectory'
        % plots trajectories of "neural" state in lower-dimensions (if data
        % is 3+ dimensions, then plots top 3, else, plots as many as
        % possible).
        X   = varargin{1}; % data matrix [NxCxT]
        if numel(varargin)==2
            clr = varargin{2};
        else
            clr = {};
        end
        %clr = varargin{2};
        [Xtraj,varExplained] = pca_lunch('MISC:calcTrajectory',X);                  % to plot trajectory of conditions
        %[Xtraj,varExplained] = pca_lunch('MISC:calcTrajectory',permute(X,[2,1,3])); % to plot trajectory of neurons
        [K,C,T]  = size(Xtraj);
        if isempty(clr)
            clr  = pca_lunch('PLOT:getTrajectoryColours',C);
        end
        % plot each condition's trajectory through PC space
        markerSize = 8;
        hold on
        for c = 1:C 
            % data to plot
            X = squeeze(Xtraj(1,c,:));
            Y = squeeze(Xtraj(2,c,:));
            if K==3
                % mark origin
                plot3(0,0,0,'m*');   
                % plot trajectory
                Z = squeeze(Xtraj(3,c,:));
                plot3(X,Y,Z,'Color',clr{c},'LineWidth',tjLineWidth);
                plot3(X(1),Y(1),Z(1),...
                    'o','MarkerFaceColor',clr{c},'MarkerEdgeColor','k','MarkerSize',markerSize); % start state
                plot3(X(end),Y(end),Z(end),'>','Color',clr{c},'MarkerSize',markerSize);          % end state
            elseif K==2
                % mark origin
                plot(0,0,'m+','MarkerSize',markerSize);   
                % plot trajectory
                plot(X,Y,'Color',clr{c},'LineWidth',tjLineWidth);
                plot(X(1),Y(1),...
                    'o','MarkerFaceColor',clr{c},'MarkerEdgeColor','k','MarkerSize',markerSize); % start state
                plot(X(end),Y(end),'>','Color',clr{c},'MarkerSize',markerSize);                  % end state
            else
                error('case made to plot only 2 or 3 dimenions.')
            end
        end
        grid on
        axis equal
        
        xlabel(sprintf('pc1 (%2.1f%% var)',varExplained(1)));
        ylabel(sprintf('pc2 (%2.1f%% var)',varExplained(2)));
        if K==3
            zlabel(sprintf('pc3 (%2.1f%% var)',varExplained(3)));
        end
    case 'PLOT:getTrajectoryColours'
        % simple helper case to adaptively update plot colours for
        % trajectory plots
        C = varargin{1}; % number of conditions to get colours for
        mapToUse = 'jet';
        if numel(varargin)==2
            mapToUse = varargin{2};
        end
        cmap      = colormap(mapToUse);
        clrIdx    = floor(linspace(0,1,C).*size(cmap,1));
        clrIdx(1) = 1;
        clrs      = cmap(clrIdx,:);
        clrs      = mat2cell(clrs,ones(C,1),3);
        varargout = {clrs};
    case 'PLOT:temporalWeights'
        a = varargin{1};
        b = varargin{2};
        numWeights = length(a);
        imagesc([a;b]);
        title(sprintf('dynamic and input\nfeature weights'));
        % styalize
        colormap(flipud(hot));
        caxis([0 1.5]);
        xlabel('dataset');
        labels = {'dyanmic weight (a)','input weight (b)'};
        set(gca,'ytick',[1 2],'yticklabel',labels,'yticklabelrotation',45,...
            'xtick',[1:numWeights]);
        h = get(gca);
        h.YAxis.FontSize = 10;
        % label the weights in the coloured image.
        text([1:numWeights,1:numWeights],...
            [ones(1,numWeights)+0.25,ones(1,numWeights).*2.25],...
            sprintfc('%1.2f',[a,b]),...
            'Rotation',90);
        varargout = {h};
    case 'PLOT:tensorFitRatios'
        ratio = varargin{1};
        if numel(varargin)==2
            label = varargin{2};
        end
        plot(ratio,'o-k','LineWidth',1.5);
        box off
        xlim([0.5 length(ratio)+0.5]);
        %ylim([-1 1]);
        pca_lunch('PLOT:drawlines',1,[0.7 0.7 0.7]);
        xlabel('dataset');
        ylabel('log( r2 condition mode / r2 neuron mode )');
        title(sprintf('estimating the mode\nof the system'))
        if exist('label','var')
            text(1:length(ratio),ratio,label,'Rotation',-35);
        end
        text(0.6,0.1,'\uparrow stronger dynamics');
        text(0.6,-0.1,'\downarrow input-driven');
    case 'PLOT:neuronModeTrajectory'
        % plots trajectories of "neural" state in lower-dimensions (if data
        % is 3+ dimensions, then plots top 3, else, plots as many as
        % possible).
        X = varargin{1}; % data matrix [NxCxT]
        [Xtraj,varExplained] = pca_lunch('MISC:calcTrajectory',X);
        [K,C,T]  = size(Xtraj);
        clr      = pca_lunch('PLOT:getTrajectoryColours',C);
        % plot each condition's trajectory through PC space
        markerSize = 8;
        hold on
        for c = 1:C 
            % data to plot
            X = squeeze(Xtraj(1,c,:));
            Y = squeeze(Xtraj(2,c,:));
            if K==3
                % mark origin
                plot3(0,0,0,'m*');   
                % plot trajectory
                Z = squeeze(Xtraj(3,c,:));
                plot3(X,Y,Z,'Color',clr(c,:),'LineWidth',tjLineWidth);
                plot3(X(1),Y(1),Z(1),...
                    'o','MarkerFaceColor',clr(c,:),'MarkerEdgeColor','k','MarkerSize',markerSize); % start state
                plot3(X(end),Y(end),Z(end),'>','Color',clr(c,:),'MarkerSize',markerSize);          % end state
            elseif K==2
                % mark origin
                plot(0,0,'m+','MarkerSize',markerSize);   
                % plot trajectory
                plot(X,Y,'Color',clr(c,:),'LineWidth',tjLineWidth);
                plot(X(1),Y(1),...
                    'o','MarkerFaceColor',clr(c,:),'MarkerEdgeColor','k','MarkerSize',markerSize); % start state
                plot(X(end),Y(end),'>','Color',clr(c,:),'MarkerSize',markerSize);                  % end state
            else
                error('case made to plot only 2 or 3 dimenions.')
            end
        end
        grid on
        axis equal
        
        xlabel(sprintf('pc1 (%2.1f%% var)',varExplained(1)));
        ylabel(sprintf('pc2 (%2.1f%% var)',varExplained(2)));
        if K==3
            zlabel(sprintf('pc3 (%2.1f%% var)',varExplained(3)));
        end
        
    case '0' % ------------ PCA examples with image reconstruction --------
    case 'IMG:reconstruction'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Does image reconstruction from PCs of image data.
        % Plots reconstructions of the images using different numbers of
        % PCs.
        % You can adjust how many PCs are used in the reconstructions by
        % chaning the variable k below.
        %
        % inputs:
        %       varargin{1} : string of image to load 
        %                       ('baboon','cocolizo','drake')
        % output: none
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        img = varargin{1};              % image to open 
        X   = pca_lunch(['IMG:' img]);  % get image
        k   = [1 2 4 8 16 32 64 128 256 min(size(X))];% round(linspace(10,min(size(X)),4))]; % choose how many PCs to use in each reconstruction (here, we are taking equal steps from 10:max PCs)
        % get subplot indices for plotting of reconstructed images
        numSubplot = numel(k); % number of plots for reconstructed images
        numCol     = 3;
        numRow     = ceil(numSubplot/numCol) + 1;
        % plot original image
        figure('Color',[1 1 1]);
        subplot(numRow,numCol,1);
        imagesc(X);
        title('original image');
        h = get(gca);
        h.XAxis.Visible = 'off';
        h.YAxis.Visible = 'off';
        for i = 1:numel(k)
            % get reconstructed image
            [~,~,~,Xrecon] = pca_lunch('PCA:svd',X,k(i));
            %[~,~,~,Xrecon] = pca_lunch('PCA:eig',X,k(i));
            % plot reconstructed image
            subplot(numRow,numCol,numCol+i);
            imagesc(Xrecon); 
            title(sprintf('%dpc recon',k(i)));
        end
        % stylize
        for i = 1:numSubplot
            subplot(numRow,numCol,numCol+i);
            h = get(gca);
            h.XAxis.Visible = 'off';
            h.YAxis.Visible = 'off';
            %axis equal
        end
        colormap gray
        % plot varience explained by PCs
        [~,~,~,~,varExp] = pca_lunch('PCA:svd',X);
        cumVarExp = cumsum(varExp);
        subplot(numRow,numCol,2:numCol);
        plot(1:length(cumVarExp),cumVarExp,'Color','r','LineWidth',2);
        ylim([0 100]);
        box off
        grid on
        title('variance explained');
        ylabel('% variance');
        xlabel('# PCs')
    case 'IMG:baboon'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % loads baboon image
        % source: https://homepages.cae.wisc.edu/~ece533/images/
        %
        % output:
        % varargout{1} : X - 2D matrix of image that can be viewed via imagesc(X)
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        d         = pca_lunch('MISC:getDirectory');
        imgName   = fullfile(d,'test_images','baboon.png');
        X         = imread(imgName);
        X         = double(X(:,:,3)); % convert unit8 to double so we can perform operations
        varargout = {X};
    case 'IMG:cocolizo'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % loads Cocolizo image
        % source: https://commons.wikimedia.org/wiki/File:Cocolizo.jpg
        %
        % output:
        % varargout{1} : X - 2D matrix of image that can be viewed via imagesc(X)
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        d         = pca_lunch('MISC:getDirectory');
        imgName   = fullfile(d,'test_images','cocolizo.jpg');
        X         = imread(imgName);
        X         = double(X(:,:,1)); % convert unit8 to double so we can perform operations
        varargout = {X};
    case 'IMG:drake'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % loads image of Drake confusing himself for a member of the
        % Raptors courtside during the Eastern conference finals.
        % source: https://www.thebeaverton.com/2019/05/no-one-on-raptors-has-the-heart-to-tell-drake-hes-not-on-the-team/
        %
        % output:
        % varargout{1} : X - 2D matrix of image that can be viewed via imagesc(X)
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        d         = pca_lunch('MISC:getDirectory');
        imgName   = fullfile(d,'test_images','drake.jpg');
        X         = imread(imgName);
        X         = double(X(:,:,2)); % convert unit8 to double so we can perform operations
        varargout = {X};
    case 'IMG:bruce'
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % loads image of Lion's Head on the Bruce Peninsula
        % source: https://www.alltrails.com/trail/canada/ontario/lions-head-loop-via-bruce-trail
        %
        % output:
        % varargout{1} : X - 2D matrix of image that can be viewed via imagesc(X)
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        d         = pca_lunch('MISC:getDirectory');
        imgName   = fullfile(d,'test_images','bruce.jpg');
        X         = imread(imgName);
        X         = double(X(:,:,3)); % convert unit8 to double so we can perform operations
        varargout = {X};
    
        
    case '0' % ------------ cases in development --------------------------
    % TO DO:
    % - explicit estimation of A from temporal data
    % - make A rotation-only, another one scaling-only
    % - case to calculate tangling matrix (more tangling for input-driven)
    case 'MISC:calcTangling'
        X = varargin{1}; % data matrix [NxCxT]
        Xtraj = pca_lunch('MISC:calcTrajectory',X);
        dX    = Xtraj(2:end,:) - Xtraj(1:end-1,:);
        ddX   = dX(2:end,:) - dX(1:end-1,:);
        Q     = norm(ddX,2)/(norm(dX,2)+realmin);
        varargout = {Q};
    case 'DO:test'
        [U,A,B] = pca_lunch('SIM:temporalDataPrep');
        % make A rotational transform
        Ar = tril(A,-1);
        Ar = Ar-Ar';
        Ad = (A+A')./2;
        % generate data with rotational temporal dynamics
        Xr = pca_lunch('SIM:autoDynamics',Ar,U(:,:,1),size(U,3));
        Xd = pca_lunch('SIM:autoDynamics',Ad,U(:,:,1),size(U,3));
        Xa = pca_lunch('SIM:autoDynamics',Ar+(Ad./10),U(:,:,1),size(U,3));
        
        figure('Color',[1 1 1]);
        subplot(1,3,1);
        pca_lunch('PLOT:trajectory',squeeze(Xr(:,1,:)));
        subplot(1,3,2);
        pca_lunch('PLOT:trajectory',squeeze(Xd(:,1,:)));
        subplot(1,3,3);
        pca_lunch('PLOT:trajectory',squeeze(Xa(:,1,:)));   
        
        
    otherwise
        error('%s : no such case',what)
end % switch what

% ----------------------------------------------------------------------- %

%% ------------ helpful resources -----------------------------------------
%
% Lindsay Smith (Uni. of Otago, NZ) PCA tutorial:
%       https://ourarchive.otago.ac.nz/bitstream/handle/10523/7534/OUCS-2002-12.pdf?sequence=1&isAllowed=y
%
% CrossValidated thread on Making Sense of PCA (user amoeba):
%       https://stats.stackexchange.com/q/140579
%
% CrossValidated thread on Relationship between SVD & PCA (user amoeba):
%       https://stats.stackexchange.com/q/134283
%
% CrossValidated thread on reversing PCA to reconstruct data using PCs (user amoeba):
%       https://stats.stackexchange.com/q/229093       
%
% Everything you didn't know about PCA (Alex Williams, 2016):
%       http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
%
% Roweis & Gharamni (1999)- unifying model of PCA, probabilistic PCA, FA, 
% Gaussian mixture models, ICA, & Kalman filters:
%       https://www.seas.harvard.edu/courses/cs281/papers/lds.pdf
%
% Thacker & Bromiley (2001?)- proof that sqrt(poisson variables) will
% approximate a gaussian distribution even at low n:
%       http://tina.wiau.man.ac.uk/tina-knoppix/tina-memo/2001-010.pdf
%
% Grant Sanderson's 3Blue1Brown videos on Linear Algebra (presents math
% with "a visuals-first approach"):
%       https://www.3blue1brown.com/
%
% Byron Yu et al. (2009) paper on Gaussian-process factor analysis for
% single trial analyses of population activity. Different than PCA, but
% related.
%       https://web.stanford.edu/~shenoy/GroupPublications/YuEtAlNIPS2009.pdf
%
% Seely et al. (2016) paper on Tensor analysis to uncover preferred "mode"
% of a population of neurons.
%       https://doi.org/10.1371/journal.pcbi.1005164
%



end % function


