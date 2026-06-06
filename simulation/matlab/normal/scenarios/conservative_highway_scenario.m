function [initialActorPose,accelerationProfile,stopTime] = ...
    conservative_highway_scenario( ...
    tractorTrailerParameters,vehicleDimension, currentLHS)

% =========================================================
% CONSERVATIVE HIGHWAY PROFILE
%
% Target:
% Cruise Speed      ≈ 20 m/s
% Initial Spacing   ≈ 26 m
% Steady Spacing    ≈ 12 m
% =========================================================

% =========================================================
% PROFILE VARIATION
% =========================================================

% Gentle acceleration
ampValues = 0.10 + (0.15 - 0.10)*currentLHS(1,1);

% Cruise speed
velBases = 19.5 + (20.5 - 19.5)*currentLHS(1,2);

% Initial spacing
spacingBases = 23.0 + (27.0 - 23.0)*currentLHS(1,3);

% RSU guidance values

deltaSpeedUp1  =  0.05 + (0.1 - 0.05)*currentLHS(1,4);
deltaSpeedUp2  =  0.02 + (0.07 - 0.02)*currentLHS(1,5);
deltaSpeedDown = -0.15 + (-0.05 - (-0.15))*currentLHS(1,6);

% Message timing offsets
timingOffset1 = -3.0 + 6.0*currentLHS(1,7);
timingOffset2 = -3.0 + 6.0*currentLHS(1,8);
timingOffset3 = -3.0 + 6.0*currentLHS(1,9);

% =========================================================
% VEHICLE OFFSETS
% =========================================================

velDeltas = [0.0 -0.1 -0.2 -0.3 -0.4];
spacingDeltas = [0.0 -2.0 -4.0 -6.0];

assignin('base', 'runDeltaSpeedUp1',  deltaSpeedUp1);
assignin('base', 'runDeltaSpeedUp2',  deltaSpeedUp2);
assignin('base', 'runDeltaSpeedDown', deltaSpeedDown);
assignin('base', 'runTimingOffset1',  timingOffset1);
assignin('base', 'runTimingOffset2',  timingOffset2);
assignin('base', 'runTimingOffset3',  timingOffset3);

% =========================================================
% INITIAL VELOCITIES
% =========================================================

vBase = velBases;
initialVelocities.Leader    = vBase + velDeltas(1);
initialVelocities.Follower1 = vBase + velDeltas(2);
initialVelocities.Follower2 = vBase + velDeltas(3);
initialVelocities.Follower3 = vBase + velDeltas(4);
initialVelocities.Follower4 = vBase + velDeltas(5);

% =========================================================
% INITIAL SPACINGS
% =========================================================
sBase = spacingBases;
initialSpacing.LeaderToFollower1    = sBase + spacingDeltas(1);
initialSpacing.Follower1ToFollower2 = sBase + spacingDeltas(2);
initialSpacing.Follower2ToFollower3 = sBase + spacingDeltas(3);
initialSpacing.Follower3ToFollower4 = sBase + spacingDeltas(4);

% =========================================================
% CREATE VEHICLE POSES
% =========================================================

initialActorPose = helperInitializeVehiclePose( ...
    initialSpacing, ...
    initialVelocities, ...
    tractorTrailerParameters, ...
    vehicleDimension);

% =========================================================
% LEADER ACCELERATION PROFILE
% =========================================================

accelerationProfile.Amplitude = [ampValues 0 -0.15 0];
accelerationProfile.Period = 60;
accelerationProfile.PulseWidth = [2 15 8 30];
accelerationProfile.PhaseDelay = [1 3 30 38];

% =========================================================
% SIMULATION LENGTH
% =========================================================
stopTime = 60;

% =========================================================
% EXPORT PARAMETERS TO BASE WORKSPACE
% (for logging into CSV later)
% =========================================================

assignin('base','vBase',vBase);
assignin('base','sBase',sBase);

assignin('base', 'accelerationProfile', accelerationProfile);
assignin('base','initialSpacing',initialSpacing);
assignin('base','initialVelocities',initialVelocities);

end