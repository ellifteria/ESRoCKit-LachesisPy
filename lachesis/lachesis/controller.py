from typing import Union, Any
import pybullet as pblt

class RobotController():

    # PRIVATE METHODS
    def _determine_joint_link_ids(self) -> tuple[dict[str, int], dict[str, int]]:
        joint_dictionary:   dict[str, int] = {}
        link_dictionary:    dict[str, int] = {}

        base_link_name = pblt.getBodyInfo(self.robot_id)[0].decode("utf-8")
        link_dictionary[base_link_name] = -1
        
        for joint_index in range(pblt.getNumJoints(self.robot_id)):
            joint_info = pblt.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode('utf-8')
            joint_dictionary[joint_name] = joint_index
            link_name = joint_name.split("TO")[1]
            link_dictionary[link_name] = joint_index
        
        return joint_dictionary, link_dictionary
    
    # CONSTRUCTOR
    def __init__(self, robot_id: int) -> None:
        self.robot_id:      int             = robot_id
        tmp_dicts = self._determine_joint_link_ids()
        self.robot_joints:  dict[str, int]  = tmp_dicts[0]
        self.robot_links:   dict[str, int]  = tmp_dicts[1]

    # ACCESS METHODS
    def get_robot_joints(self) -> dict[str, int]:
        return self.robot_joints

    # ACTION METHODS
    def update_joint_motors(
            self,
            control_mode:       int,
            target_positions:   dict[str, float] | None     = None,
            target_velocities:  dict[str, float] | None     = None,
            forces:             list[float] | float | None  = None,
            position_gains:     list[float] | None          = None,
            velocity_gains:     list[float] | None          = None,
            physics_client:     int | None                  = None
    ) -> None:
        num_joints:     int         = len(self.robot_joints)

        joint_ids:  list[int]   = [-1] * num_joints
        positions:  list[float] = [-1.0] * num_joints
        velocities: list[float] = [-1.0] * num_joints

        for index, joint_name in enumerate(self.robot_joints):
            joint_ids[index]        = self.robot_joints[joint_name]
            if target_positions is not None:
                positions[index]    = target_positions[joint_name] if joint_name in target_positions else 0.0
            if target_velocities is not None:
                velocities[index]    = target_velocities[joint_name] if joint_name in target_velocities else 0.0
        
        pblt_kwargs: dict[str, Any] = {}

        if target_positions is not None and control_mode == pblt.POSITION_CONTROL:
            pblt_kwargs["targetPositions"] = positions
        if target_velocities is not None:
            pblt_kwargs["targetVelocities"] = velocities
        if forces is not None:
            if isinstance(forces, (float, int)):
                if target_positions is not None:
                    pblt_kwargs["forces"] = [forces for i in target_positions]
                elif target_velocities is not None:
                    pblt_kwargs["forces"] = [forces for i in target_velocities]
            else:
                pblt_kwargs["forces"] = forces
        if position_gains is not None:
            pblt_kwargs["positionGains"] = position_gains
        if velocity_gains is not None:
            pblt_kwargs["velocityGains"] = velocity_gains
        if physics_client is not None:
            pblt_kwargs["physicsClientId"] = physics_client
        
        pblt.setJointMotorControlArray(
            bodyIndex = self.robot_id,
            jointIndices = joint_ids,
            controlMode = control_mode,
            **pblt_kwargs
        )

    def is_link_colliding(self, link_name) -> bool:
        link_index = self.robot_links[link_name]
        
        contact_points = None
        while contact_points is None:
            contact_points = pblt.getContactPoints(bodyA = self.robot_id)

        for contact in contact_points:
            colliding_link_index = contact[3]

            if colliding_link_index == link_index:
                return True
        
        return False
        