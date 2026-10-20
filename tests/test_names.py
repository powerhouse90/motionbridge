"""Name evidence across common rig conventions. Run: python -m tests.test_names"""
from motionbridge.core.rigmap import parse_name

CASES = {
    # Mixamo
    'mixamorig:LeftArm': ('Left', 'Arm'), 'mixamorig:LeftForeArm': ('Left', 'ForeArm'),
    'mixamorig:LeftShoulder': ('Left', 'Shoulder'), 'mixamorig:LeftUpLeg': ('Left', 'Leg'),
    'mixamorig:LeftLeg': ('Left', 'Shin'), 'mixamorig:RightToeBase': ('Right', 'ToeBase'),
    'mixamorig:RightHandIndex2': ('Right', 'Index'), 'mixamorig:Hips': (None, 'Hips'),
    'mixamorig:Spine2': (None, 'Spine'), 'mixamorig:HeadTop_End': (None, 'Head'),
    # Unreal / Mesh2Motion
    'upperarm_l': ('Left', 'Arm'), 'lowerarm_r': ('Right', 'ForeArm'), 'clavicle_l': ('Left', 'Shoulder'),
    'thigh_r': ('Right', 'Leg'), 'calf_l': ('Left', 'Shin'), 'ball_r': ('Right', 'ToeBase'),
    'thumb_01_l': ('Left', 'Thumb'), 'pinky_03_r': ('Right', 'Pinky'), 'pelvis': (None, 'Hips'),
    'upperarm_twist_01_l': ('Left', 'twist'),
    # Rigify deform bones
    'DEF-upper_arm.L': ('Left', 'Arm'), 'DEF-upper_arm.L.001': ('Left', 'Arm'), 'DEF-forearm.R': ('Right', 'ForeArm'),
    'DEF-thigh.L': ('Left', 'Leg'), 'DEF-shin.R': ('Right', 'Shin'), 'DEF-f_index.01.L': ('Left', 'Index'),
    'DEF-spine.003': (None, 'Spine'),
    # VRM / VRoid
    'J_Bip_L_UpperArm': ('Left', 'Arm'), 'J_Bip_R_LowerLeg': ('Right', 'Shin'), 'J_Bip_C_Hips': (None, 'Hips'),
    'J_Bip_L_Little1': ('Left', 'Pinky'),
    # Character Creator
    'CC_Base_L_Upperarm': ('Left', 'Arm'), 'CC_Base_R_Calf': ('Right', 'Shin'), 'CC_Base_NeckTwist01': (None, 'Neck'),
    # Tripo / project rigs
    'L_Thigh': ('Left', 'Leg'), 'R_ForearmTwist01': ('Right', 'twist'), 'Waist': (None, 'Spine'),
    'NeckTwist02': (None, 'Neck'),
    # Must not read a side out of ordinary words
    'Leg': (None, 'Shin'), 'LowerArm': (None, 'ForeArm'), 'Root': (None, None),
}


def main():
    failures = [(n, parse_name(n), want) for n, want in CASES.items() if parse_name(n) != want]
    for name, got, want in failures:
        print(f'FAIL {name}: got {got}, want {want}')
    print(f'{len(CASES) - len(failures)}/{len(CASES)} names parsed as expected')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
