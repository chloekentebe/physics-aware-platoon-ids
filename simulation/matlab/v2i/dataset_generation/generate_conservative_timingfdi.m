% generate_rsutiming_attack.m
model    = 'PlatooningUsingV2VTestBench';
numRuns  = 10;

rng(42, 'twister');
lhs = lhs_sample(numRuns, 12);  % 9 profile params + 3 attack params

% Attack params — columns 10-12
% For timing attack: shift when messages are sent
attackTimingShift1 = -5.0 + (10.0 - (-5.0)) * lhs(:, 10);  % [-5, +5]s shift on SpeedUp1
attackTimingShift2 = -5.0 + (10.0 - (-5.0)) * lhs(:, 11);  % [-5, +5]s shift on SpeedUp2
attackTimingShift3 = -5.0 + (10.0 - (-5.0)) * lhs(:, 12);  % [-5, +5]s shift on SlowDown


runOrder = randperm(numRuns);

outPath = '/Users/chloekentebe/physics-aware-platoon-ids/simulation/matlab/v2i/attack_dataset';

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
    attackTypeNum = attackTypeMap('None'); % for V2V

    % Attack timing — shift RSU message windows
    assignin('base', 'attackType',       'TimingFDI');
    assignin('base', 'runattackTimingShift1', attackTimingShift1(runIdx));
    assignin('base', 'runattackTimingShift2', attackTimingShift2(runIdx));
    assignin('base', 'runattackTimingShift3', attackTimingShift3(runIdx));

    % Create timeseries for attack params so Simulink can read them
    t  = (0:0.1:60)';
    Ns = length(t);

    assignin('base', 'attackTypeNum',  ones(Ns,1) * 5);   % 5=TimingFDI
    assignin('base', 'attackerVec',    ones(Ns,1) * -1);
    assignin('base', 'attackMagVec',   zeros(Ns,1));       % not used for timing
    assignin('base', 'attackStartVec', zeros(Ns,1));       % not used
    assignin('base', 'attackDurVec',   ones(Ns,1) * 60);  % whole sim

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
    % Attack labels
    T.IsAttack       = repmat(1,           N, 1);
    T.AttackVector   = repmat("V2I",       N, 1);
    T.AttackType     = repmat("TimingFDI", N, 1);
    T.AttackerID     = repmat(-1,          N, 1);  % -1 = RSU, not a vehicle
    T.AttackMag      = repmat(0,           N, 1);  % no magnitude for timing
    T.AttackStart    = repmat(0,           N, 1);  % whole sim from t=0
    T.AttackDuration = repmat(60,          N, 1);  % whole sim
    T.AttackActive   = ones(N, 1);                 % always active
    T.TimingShift1   = repmat(attackTimingShift1(runIdx), N, 1);
    T.TimingShift2   = repmat(attackTimingShift2(runIdx), N, 1);
    T.TimingShift3   = repmat(attackTimingShift3(runIdx), N, 1);
    T.AttackDelta1   = repmat(0, N, 1);
    T.AttackDelta2   = repmat(0, N, 1);
    T.AttackDelta3   = repmat(0, N, 1);

    % =====================================================
    % SAVE PLATOON TABLE
    % =====================================================
    platoonFile = fullfile(outPath, sprintf('platoon_timingfdi_conservative_%d.csv', loopIdx));
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
                tempTable.IsAttack       = repmat(1,           M, 1);
                tempTable.AttackVector   = repmat("V2I",       M, 1);
                tempTable.AttackType     = repmat("TimingFDI", M, 1);
                tempTable.AttackerID     = repmat(-1,          M, 1);  % RSU not a vehicle
                tempTable.AttackMag      = repmat(0,           M, 1);
                tempTable.AttackStart    = repmat(0,           M, 1);
                tempTable.AttackDuration = repmat(60,          M, 1);
                tempTable.AttackActive   = ones(M, 1);                 % always active
                tempTable.TimingShift1   = repmat(attackTimingShift1(runIdx), M, 1);
                tempTable.TimingShift2   = repmat(attackTimingShift2(runIdx), M, 1);
                tempTable.TimingShift3   = repmat(attackTimingShift3(runIdx), M, 1);
                tempTable.AttackDelta1   = repmat(0, M, 1);
                tempTable.AttackDelta2   = repmat(0, M, 1);
                tempTable.AttackDelta3   = repmat(0, M, 1);
                tempTable.IsSenderAttacker = zeros(M, 1);

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

    bsmFile = fullfile(outPath, sprintf('bsm_timingfdi_conservative_%d.csv', loopIdx));
    writetable(BSMTable, bsmFile);
    fprintf('Run %d COMPLETE \n', loopIdx);
end
disp('YAY, all simulations complete :) <3')

function lhs = lhs_sample(n, k)
    lhs = zeros(n, k);
    for i = 1:k
        lhs(:, i) = (randperm(n)' - rand(n, 1)) / n;
    end
end