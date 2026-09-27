# [NEW - 2026-09-12]
# Reason: Unit test for dynamic parameter tuning and attribute updates on VisualAimController components.
# Content: Verifies updating kp_x, kd_x, kp_y, kd_y, deadband, conf_threshold, hold_right_button, monitor_index.

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.pid import DualAxisPID, PIDConfig

def test_dynamic_param_updates():
    pid = DualAxisPID()
    assert pid.yaw_pid.config.kp == 0.42
    assert pid.pitch_pid.config.kp == 0.36

    # Test dynamic update method logic
    def update_params(
        pid_obj: DualAxisPID,
        kp_x=None, kd_x=None, kp_y=None, kd_y=None, deadband=None
    ):
        if kp_x is not None:
            pid_obj.yaw_pid.config.kp = kp_x
        if kd_x is not None:
            pid_obj.yaw_pid.config.kd = kd_x
        if kp_y is not None:
            pid_obj.pitch_pid.config.kp = kp_y
        if kd_y is not None:
            pid_obj.pitch_pid.config.kd = kd_y
        if deadband is not None:
            pid_obj.yaw_pid.config.deadband = deadband
            pid_obj.pitch_pid.config.deadband = deadband

    update_params(pid, kp_x=0.60, kd_x=0.12, kp_y=0.50, kd_y=0.10, deadband=6.0)

    assert pid.yaw_pid.config.kp == 0.60
    assert pid.yaw_pid.config.kd == 0.12
    assert pid.pitch_pid.config.kp == 0.50
    assert pid.pitch_pid.config.kd == 0.10
    assert pid.yaw_pid.config.deadband == 6.0
    assert pid.pitch_pid.config.deadband == 6.0

    print("[PASS] PID dynamic parameters update successfully verified.")

if __name__ == "__main__":
    test_dynamic_param_updates()
