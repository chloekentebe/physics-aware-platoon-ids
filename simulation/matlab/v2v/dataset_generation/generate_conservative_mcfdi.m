% generate_mcfdi_attack.m
model    = 'PlatooningUsingV2VTestBench';
numRuns  = 10;

rng(42, 'twister');
lhs = lhs_sample(numRuns, 12);  % 9 profile params + 3 attack params

% Attack parameters
% MsgCnt — replay depth range
% 1 = replay previous message (hardest to detect)
% 5 = replay 5 messages back (more obvious)
attackMagnitudes = 1 + (5 - 1) * lhs(:, 10);   % [1, 5] steps
attackMagnitudes = round(attackMagnitudes);       % must be integer steps
attackStarts     = 10.0 + (40.0 - 10.0) * lhs(:, 11);  
attackDurations  = 2.0  + (6.0  - 2.0)  * lhs(:, 12);  

% 0=Leader, 1=Follower1, 2=Follower2, 3=Follower3, 4=Follower4
attackerVehicles = repmat((0:4)', 2, 1);        % [0;1;2;3;4;0;1;2;3;4]
attackerVehicles = attackerVehicles(randperm(numRuns));

runOrder = randperm(numRuns);

outPath = '/Users/chloekentebe/physics-aware-platoon-ids/simulation/matlab/v2v/attack_dataset';

for loopIdx = 1:numRuns
    runIdx = runOrder(loopIdx);
    currentLHS = lhs(runIdx, :);

    scenarioHandle = @(p1, p2) conservative_highway_scenario(p1, p2, currentLHS);
    helperSLPlatooningUsingV2VSetup(ScenarioFcnName=scenarioHandle);

    % BSM dimensions
    BusBSM = evalin('base', 'BusBSM');
    for i = 1:numel(BusBSM.Elements)
        if strcmp(BusBSM.Elements(i).Name, 'BSMCoreData')
            BusBSM.Elements(i).Dimensions = [5 1];
            break;
        end
    end
    assignin('base', 'BusBSM', BusBSM);
    assignin('base', 'spacing', 20);

    % Map attack type to number
    attackTypeMap = containers.Map(...
        {'None','SpeedFDI','PosFDI','AccFDI','MsgCnt'}, ...
        {0, 1, 2, 3, 4});
    attackTypeNum = attackTypeMap('MsgCnt');

    % Create timeseries for attack params so Simulink can read them
    t  = (0:0.1:60)';
    Ns = length(t);
    
    assignin('base', 'attackTypeNum',  ones(Ns,1) * attackTypeNum);
    assignin('base', 'attackerVec',    ones(Ns,1) * attackerVehicles(runIdx));
    assignin('base', 'attackMagVec',   ones(Ns,1) * attackMagnitudes(runIdx));
    assignin('base', 'attackStartVec', ones(Ns,1) * attackStarts(runIdx));
    assignin('base', 'attackDurVec',   ones(Ns,1) * attackDurations(runIdx));

    out = sim(model, 'StopTime', '60');

    % =====================================================
    % EXTRACT SIGNALS — identical to normal generation
    % =====================================================
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
    spacing1Val  = squeeze(permute(spacing1.Values.Data, [3 2 1]));
    spacing2Val  = squeeze(permute(spacing2.Values.Data, [3 2 1]));
    spacing3Val  = squeeze(permute(spacing3.Values.Data, [3 2 1]));
    spacing4Val  = squeeze(permute(spacing4.Values.Data, [3 2 1]));
    leaderVal    = squeeze(permute(leader.Values.Data,   [3 2 1]));
    follower1Val = squeeze(permute(follower1.Values.Data,[3 2 1]));
    follower2Val = squeeze(permute(follower2.Values.Data,[3 2 1]));
    follower3Val = squeeze(permute(follower3.Values.Data,[3 2 1]));
    follower4Val = squeeze(permute(follower4.Values.Data,[3 2 1]));

    % =====================================================
    % BUILD PLATOON TABLE
    % =====================================================
    T = table(time, spacing1Val, spacing2Val, spacing3Val, spacing4Val, ...
              leaderVal, follower1Val, follower2Val, follower3Val, follower4Val);

    % Run metadata
    T.RunID        = repmat(loopIdx, N, 1);
    T.Profile      = repmat("Conservative_Highway", N, 1);

    % Initial conditions from workspace
    initialSpacing    = evalin('base', 'initialSpacing');
    initialVelocities = evalin('base', 'initialVelocities');
    accelProfile      = evalin('base', 'accelerationProfile');

    T.Init_Spacing_L_F1  = repmat(initialSpacing.LeaderToFollower1,    N, 1);
    T.Init_Spacing_F1_F2 = repmat(initialSpacing.Follower1ToFollower2, N, 1);
    T.Init_Spacing_F2_F3 = repmat(initialSpacing.Follower2ToFollower3, N, 1);
    T.Init_Spacing_F3_F4 = repmat(initialSpacing.Follower3ToFollower4, N, 1);
    T.Init_Vel_Leader    = repmat(initialVelocities.Leader,    N, 1);
    T.Init_Vel_Follower1 = repmat(initialVelocities.Follower1, N, 1);
    T.Init_Vel_Follower2 = repmat(initialVelocities.Follower2, N, 1);
    T.Init_Vel_Follower3 = repmat(initialVelocities.Follower3, N, 1);
    T.Init_Vel_Follower4 = repmat(initialVelocities.Follower4, N, 1);
    T.Accel_Amplitude    = repmat(accelProfile.Amplitude(1),   N, 1);

    % =====================================================
    % ATTACK LABELS
    % =====================================================
    T.IsAttack       = repmat(1,                            N, 1);
    T.AttackVector   = repmat("V2V",                        N, 1);
    T.AttackType     = repmat("MCFDI",                     N, 1);
    T.AttackerID     = repmat(attackerVehicles(runIdx),     N, 1);
    T.AttackMag      = repmat(attackMagnitudes(runIdx),     N, 1);
    T.AttackStart    = repmat(attackStarts(runIdx),         N, 1);
    T.AttackDuration = repmat(attackDurations(runIdx),      N, 1);

    % Attack active flag per timestep — useful for ML windowing
    attackStart_i = attackStarts(runIdx);
    attackEnd_i   = attackStarts(runIdx) + attackDurations(runIdx);
    T.AttackActive = double(time >= attackStart_i & time < attackEnd_i);

    % =====================================================
    % SAVE PLATOON TABLE
    % =====================================================
    platoonFile = fullfile(outPath, sprintf('platoon_mcfdi_conservative_%d.csv', loopIdx));
    writetable(T, platoonFile);

    % =====================================================
    % BSM TABLE — identical extraction to normal
    % =====================================================
    BSMTable   = table();
    senderMap  = ["Leader","Follower1","Follower2","Follower3","Follower4"];
    bsmSignals = {'Follower1BSM_Rx','Follower2BSM_Rx', ...
                  'Follower3BSM_Rx','Follower4BSM_Rx'};

    for b = 1:length(bsmSignals)
        sigName      = bsmSignals{b};
        receiverName = erase(sigName, 'BSM_Rx');

        try
            bsmSig = getElement(out.logsout, sigName);
            ts     = bsmSig.Values;

            for slot = 1:5
                bsmTime = ts.NumOfBSM.Time;
                M       = length(bsmTime);

                tempTable = table();
                tempTable.Time         = bsmTime;
                tempTable.RunID        = repmat(loopIdx,                  M, 1);
                tempTable.Profile      = repmat("Conservative_Highway",      M, 1);
                tempTable.ReceiverID   = repmat(string(receiverName),     M, 1);
                tempTable.SenderSlot   = repmat(slot,                     M, 1);
                tempTable.SenderID     = repmat(senderMap(slot),          M, 1);
                tempTable.NumOfBSM     = squeeze(ts.NumOfBSM.Data(:));
                tempTable.IsValidTime  = squeeze(ts.IsValidTime.Data(:));

                % Attack labels at BSM level
                tempTable.IsAttack     = repmat(1,                        M, 1);
                tempTable.AttackVector = repmat("V2V",                    M, 1);
                tempTable.AttackType   = repmat("MCFDI",                 M, 1);
                tempTable.AttackerID   = repmat(attackerVehicles(runIdx), M, 1);
                tempTable.AttackMag    = repmat(attackMagnitudes(runIdx), M, 1);
                tempTable.AttackStart  = repmat(attackStarts(runIdx),     M, 1);
                tempTable.AttackDuration = repmat(attackDurations(runIdx),M, 1);

                % Per-timestep attack active flag
                tempTable.AttackActive = double( ...
                    bsmTime >= attackStart_i & bsmTime < attackEnd_i);

                % IsSenderAttacker — true when this slot's sender is the attacker
                % slot-1 maps to vehicle ID (0=Leader, 1-4=Followers)
                tempTable.IsSenderAttacker = repmat( ...
                    double(slot - 1 == attackerVehicles(runIdx)), M, 1);

                % Simple fields
                simpleFields = {'MsgCnt','Id','SecMark','Lattitude', ...
                                'Longitude','Elevation','Speed','Heading','Angle'};
                for f = 1:length(simpleFields)
                    fname   = simpleFields{f};
                    rawData = squeeze(ts.BSMCoreData(slot).(fname).Data(:));
                    if length(rawData) ~= M
                        fieldTime = ts.BSMCoreData(slot).(fname).Time;
                        rawData   = interp1(fieldTime, double(rawData), ...
                                            bsmTime, 'nearest', 'extrap');
                    end
                    tempTable.(fname) = rawData;
                end

                % True vs Reported — ground truth consistency columns
                vehicleVals = {leaderVal, follower1Val, follower2Val, ...
                               follower3Val, follower4Val};
                
                % True speed resampled to BSM time
                tempTable.TrueSpeed_ms = interp1(time, double(vehicleVals{slot}), ...
                                                 bsmTime, 'linear', 'extrap');
                
                % Reported speed in m/s
                tempTable.ReportedSpeed_ms = double(tempTable.Speed) * 0.02;
                
                % Speed deviation
                tempTable.SpeedDeviation = tempTable.ReportedSpeed_ms - ...
                                           tempTable.TrueSpeed_ms;

                % Nested fields
                nestedFields = {'Accuracy','AccelSet','Size'};
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
                                rawData = interp1(subData.Time, double(rawData), ...
                                                  bsmTime, 'nearest', 'extrap');
                            end
                            tempTable.(sprintf('%s_%s', fname, sfName)) = rawData;
                        elseif isstruct(subData)
                            deepNames = fieldnames(subData);
                            for df = 1:length(deepNames)
                                dfName   = deepNames{df};
                                deepData = subData.(dfName);
                                if isa(deepData, 'timeseries')
                                    rawData = squeeze(deepData.Data(:));
                                    if length(rawData) ~= M
                                        rawData = interp1(deepData.Time, ...
                                            double(rawData), bsmTime, ...
                                            'nearest', 'extrap');
                                    end
                                    tempTable.(sprintf('%s_%s_%s', ...
                                        fname, sfName, dfName)) = rawData;
                                end
                            end
                        end
                    end
                end

                BSMTable = [BSMTable; tempTable]; %#ok<AGROW>
            end

            fprintf('SUCCESS: %s\n', sigName);
        catch ME
            fprintf('FAILED: %s — %s (line %d)\n', sigName, ME.message, ME.stack(1).line);
        end
    end

    bsmFile = fullfile(outPath, sprintf('bsm_mcfdi_conservative_%d.csv', loopIdx));
    writetable(BSMTable, bsmFile);
    fprintf('Run %d COMPLETE — Attacker: Vehicle %d\n', loopIdx, attackerVehicles(runIdx));
end
disp('YAY, all simulations complete :) <3')

function lhs = lhs_sample(n, k)
    lhs = zeros(n, k);
    for i = 1:k
        lhs(:, i) = (randperm(n)' - rand(n, 1)) / n;
    end
end