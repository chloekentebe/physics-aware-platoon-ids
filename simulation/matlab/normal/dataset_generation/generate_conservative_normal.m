model    = 'PlatooningUsingV2VTestBench';
numRuns  = 30;

% =========================================================
% REPRODUCIBLE RANDOM GENERATION
% =========================================================
rng(42, "twister");  % fixed seed — same values every run & twister so its consistent across all versions

lhs = lhs_sample(numRuns, 9);  % 9 = number of varied parameters

% Shuffle run order so CSV indices aren't sorted by value
runOrder = randperm(numRuns);

% Output folder
outPath = '/Users/chloekentebe/physics-aware-platoon-ids/simulation/matlab/normal_dataset';

% =========================================================
% MAIN RUN LOOP
% =========================================================
for loopIdx = 1:numRuns
    runIdx = runOrder(loopIdx);   % shuffled index

    currentLHS = lhs(runIdx,:);

    scenarioHandle = @(param1, param2) conservative_highway_scenario(param1, param2, currentLHS);

    % --- Setup scenario ---
    helperSLPlatooningUsingV2VSetup(ScenarioFcnName=scenarioHandle);

    % --- Modify BSM dimensions ---
    BusBSM = evalin('base', 'BusBSM');
    for i = 1:numel(BusBSM.Elements)
        if strcmp(BusBSM.Elements(i).Name, 'BSMCoreData')
            BusBSM.Elements(i).Dimensions = [5 1];
            break;
        end
    end
    assignin('base', 'BusBSM', BusBSM);

    assignin('base', 'spacing', 20);

    pose_check = evalin('base', 'initialActorPose');
    fprintf('Leader start vel: %.2f m/s\n', pose_check(1).Velocity(1));
    fprintf('Follower1 start vel: %.2f m/s\n', pose_check(2).Velocity(1));

    % --- Run simulation ---
    out = sim(model, 'StopTime', '60');

    % =========================================================
    % EXTRACT SPACING + VEHICLE SIGNALS
    % =========================================================
    spacing1  = getElement(out.logsout, 'Spacing Between Leader and Follower 1');
    spacing2  = getElement(out.logsout, 'Spacing Between Follower 1 and Follower 2');
    spacing3  = getElement(out.logsout, 'Spacing Between Follower 2 and Follower 3');
    spacing4  = getElement(out.logsout, 'Spacing Between Follower 3 and Follower 4');
    leader    = getElement(out.logsout, 'Leader V');
    follower1 = getElement(out.logsout, 'Follower 1 V');
    follower2 = getElement(out.logsout, 'Follower 2 V');
    follower3 = getElement(out.logsout, 'Follower 3 V');
    follower4 = getElement(out.logsout, 'Follower 4 V');

    time         = spacing1.Values.Time;
    N            = length(time);
    spacing1Val  = squeeze(permute(spacing1.Values.Data,  [3 2 1]));
    spacing2Val  = squeeze(permute(spacing2.Values.Data,  [3 2 1]));
    spacing3Val  = squeeze(permute(spacing3.Values.Data,  [3 2 1]));
    spacing4Val  = squeeze(permute(spacing4.Values.Data,  [3 2 1]));
    leaderVal    = squeeze(permute(leader.Values.Data,    [3 2 1]));
    follower1Val = squeeze(permute(follower1.Values.Data, [3 2 1]));
    follower2Val = squeeze(permute(follower2.Values.Data, [3 2 1]));
    follower3Val = squeeze(permute(follower3.Values.Data, [3 2 1]));
    follower4Val = squeeze(permute(follower4.Values.Data, [3 2 1]));

    % =========================================================
    % BUILD PLATOON TABLE
    % =========================================================
    T = table(time, spacing1Val, spacing2Val, spacing3Val, spacing4Val, ...
              leaderVal, follower1Val, follower2Val, follower3Val, follower4Val);

    % =====================================================
    % RUN METADATA
    % =====================================================

    T.RunID = repmat(loopIdx, N, 1);

    T.Profile = repmat( ...
        "Conservative_Highway", ...
        N, ...
        1);

    T.ScenarioType = repmat( ...
        "Normal", ...
        N, ...
        1);

    % =========================================================
    % METADATA COLUMNS — initial conditions for this run
    % =========================================================

    % Initial spacings
    T.Init_Spacing_L_F1  = repmat(initialSpacing.LeaderToFollower1,    N, 1);
    T.Init_Spacing_F1_F2 = repmat(initialSpacing.Follower1ToFollower2, N, 1);
    T.Init_Spacing_F2_F3 = repmat(initialSpacing.Follower2ToFollower3, N, 1);
    T.Init_Spacing_F3_F4 = repmat(initialSpacing.Follower3ToFollower4, N, 1);

    % Initial velocities
    T.Init_Vel_Leader    = repmat(initialVelocities.Leader,    N, 1);
    T.Init_Vel_Follower1 = repmat(initialVelocities.Follower1, N, 1);
    T.Init_Vel_Follower2 = repmat(initialVelocities.Follower2, N, 1);
    T.Init_Vel_Follower3 = repmat(initialVelocities.Follower3, N, 1);
    T.Init_Vel_Follower4 = repmat(initialVelocities.Follower4, N, 1);

    % Acceleration profile
    T.Accel_Amplitude   = repmat(ampValues(runIdx),                    N, 1);
    T.Accel_Period      = repmat(accelerationProfile.Period,           N, 1);
    T.Accel_PulseWidth1 = repmat(accelerationProfile.PulseWidth(1),   N, 1);
    T.Accel_PulseWidth2 = repmat(accelerationProfile.PulseWidth(2),   N, 1);
    T.Accel_PhaseDelay1 = repmat(accelerationProfile.PhaseDelay(1),   N, 1);
    T.Accel_PhaseDelay2 = repmat(accelerationProfile.PhaseDelay(2),   N, 1);

    T.Delta_SpeedUp1   = repmat(deltaSpeedUp1(runIdx),  N, 1);
    T.Delta_SpeedUp2   = repmat(deltaSpeedUp2(runIdx),  N, 1);
    T.Delta_SlowDown   = repmat(deltaSpeedDown(runIdx), N, 1);
    T.Timing_SpeedUp1  = repmat(timingOffset1(runIdx),  N, 1);
    T.Timing_SpeedUp2  = repmat(timingOffset2(runIdx),  N, 1);
    T.Timing_SlowDown  = repmat(timingOffset3(runIdx),  N, 1);

    % =====================================================
    % SAVE PLATOON TABLE
    % =====================================================

    platoonFile = fullfile( ...
        outPath, ...
        sprintf('platoon_run_%d.csv', loopIdx));

    writetable(T, platoonFile);

    % =====================================================
    % BUILD LONG-FORMAT BSM TABLE
    % =====================================================
    BSMTable = table();
    bsmSignalNames = { ...
        'Follower1BSM_Rx', ...
        'Follower2BSM_Rx', ...
        'Follower3BSM_Rx', ...
        'Follower4BSM_Rx'};

    senderMap = ["Leader", "Follower1", "Follower2", "Follower3", "Follower4"];

    for b = 1:length(bsmSignalNames)
        sigName      = bsmSignalNames{b};
        receiverName = erase(sigName, 'BSM_Rx');

        try
            bsmSig = getElement(out.logsout, sigName);
            ts     = bsmSig.Values;
            %fprintf('BSM time length: %d, spacing time length: %d\n', ...
                %length(ts.NumOfBSM.Time), N)
            
            for slot = 1:5
                bsmTime = ts.NumOfBSM.Time;
                M       = length(bsmTime);
            
                % Use BSM time vector — NOT spacing time
                tempTable = table();
                tempTable.Time         = bsmTime;
                tempTable.RunID        = repmat(loopIdx, M, 1);
                tempTable.Profile      = repmat("Conservative_Highway", M, 1);
                tempTable.ScenarioType = repmat("Normal", M, 1);
                tempTable.ReceiverID   = repmat(string(receiverName), M, 1);
                tempTable.SenderSlot   = repmat(slot, M, 1);
                tempTable.SenderID     = repmat(senderMap(slot), M, 1);
                tempTable.NumOfBSM     = squeeze(ts.NumOfBSM.Data(:));
                tempTable.IsValidTime  = squeeze(ts.IsValidTime.Data(:));
            
                % Simple timeseries fields
                simpleFields = {'MsgCnt','Id','SecMark','Lattitude', ...
                                'Longitude','Elevation', ...
                                'Speed','Heading','Angle'};
            
                for f = 1:length(simpleFields)
                    fname   = simpleFields{f};
                    rawData = squeeze(ts.BSMCoreData(slot).(fname).Data(:));
            
                    % Some fields may still differ — resample to M using BSM time
                    if length(rawData) ~= M
                        fieldTime = ts.BSMCoreData(slot).(fname).Time;
                        rawData   = interp1(fieldTime, double(rawData), bsmTime, 'nearest', 'extrap');
                    end
                    tempTable.(fname) = rawData;
                end
            
                % Nested struct fields
                nestedFields = {'Accuracy', 'AccelSet', 'Size'};
                for f = 1:length(nestedFields)
                    fname     = nestedFields{f};
                    subStruct = ts.BSMCoreData(slot).(fname);
                    subNames  = fieldnames(subStruct);
                    for sf = 1:length(subNames)
                        sfName  = subNames{sf};
                        subData = subStruct.(sfName);
                        if isa(subData, 'timeseries')
                            rawData = squeeze(subData.Data(:));
                            if length(rawData) ~= M
                                rawData = interp1(subData.Time, double(rawData), bsmTime, 'nearest', 'extrap');
                            end
                            colName = sprintf('%s_%s', fname, sfName);
                            tempTable.(colName) = rawData;
                        elseif isstruct(subData)
                            deepNames = fieldnames(subData);
                            for df = 1:length(deepNames)
                                dfName   = deepNames{df};
                                deepData = subData.(dfName);
                                if isa(deepData, 'timeseries')
                                    rawData = squeeze(deepData.Data(:));
                                    if length(rawData) ~= M
                                        rawData = interp1(deepData.Time, double(rawData), bsmTime, 'nearest', 'extrap');
                                    end
                                    colName = sprintf('%s_%s_%s', fname, sfName, dfName);
                                    tempTable.(colName) = rawData;
                                end
                            end
                        end
                    end
                end
            
                BSMTable = [BSMTable; tempTable]; %#ok<AGROW>
            end

            fprintf('SUCCESS: %s\n', sigName);

        catch ME
            fprintf('FAILED: %s\n', sigName);
            fprintf('Error: %s\n', ME.message);
            fprintf('Line: %d\n', ME.stack(1).line);
        end
    end

    % =====================================================
    % SAVE BSM TABLE
    % =====================================================
    bsmFile = fullfile(outPath, sprintf('normal_conservative_bsm_run_%d.csv', loopIdx));
    writetable(BSMTable, bsmFile);
    fprintf('Run %d COMPLETE\n', loopIdx);

end
disp('YAY, all simulations complete :) <3')

% Drop-in LHS replacement — no toolbox needed
function lhs = lhs_sample(n, k)
    % n = number of runs, k = number of parameters
    % Returns [n x k] matrix with values in [0, 1]
    lhs = zeros(n, k);
    for i = 1:k
        lhs(:, i) = (randperm(n)' - rand(n, 1)) / n;
    end
end