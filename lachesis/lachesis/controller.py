from typing import Union, Any
import pybullet as pblt

class RobotController():

    # PRIVATE METHODS
    def _determine_joint_ids(self) -> dict[str, int]:
        joint_dictionary: dict[str, int] = {}
        
        for joint_index in range(pblt.getNumJoints(self.robot_id)):
            joint_info = pblt.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode('utf-8')
            joint_dictionary[joint_name] = joint_index
        
        return joint_dictionary
    
    # CONSTRUCTOR
    def __init__(self, robot_id: int) -> None:
        self.robot_id:      int             = robot_id
        self.robot_joints:  dict[str, int]  = self._determine_joint_ids()

    # ACTION METHODS
    def update_joint_motors(
            self,
            control_mode:       int,
            target_positions:   dict[str, float] | None = None,
            target_velocities:  dict[str, float] | None = None,
            forces:             list[float] | None      = None,
            position_gains:     list[float] | None      = None,
            velocity_gains:     list[float] | None      = None,
            physics_client:     int | None              = None
    ) -> None:
        num_joints:     int         = len(self.robot_joints)

        joint_ids:  list[int]   = [-1] * num_joints
        positions:  list[float] = [-1] * num_joints
        velocities: list[float] = [-1] * num_joints

        for index, joint_name in enumerate(self.robot_joints):
            joint_ids[index]        = self.robot_joints[joint_name]
            if target_positions is not None:
                positions[index]    = target_positions[joint_name]
            if target_velocities is not None:
                velocities[index]    = target_velocities[joint_name]
        
        pblt_kwargs: dict[str, Any] = {}

        if target_positions is not None and control_mode == pblt.POSITION_CONTROL:
            pblt_kwargs["targetPositions"] = positions
        if target_velocities is not None:
            pblt_kwargs["targetVelocities"] = velocities
        if forces is not None:
            pblt_kwargs["forces"] = forces
        if position_gains is not None:
            pblt_kwargs["positionGains"] = position_gains
        if velocity_gains is not None:
            pblt_kwargs["velocityGains"] = velocity_gains
        if physics_client is not None:
            pblt_kwargs["physicsClientId"] = physics_client
        
        pblt.setJointMotorControlArray(
            bodyUniqueId = self.robot_id,
            jointIndices = joint_ids,
            control_mode = control_mode,
            **pblt_kwargs
        )
