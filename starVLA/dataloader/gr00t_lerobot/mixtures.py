"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],
    "vla_arena_L0_S": [
        ("VLA_Arena_L0_S_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],
    "vla_arena_L0_M": [
        ("VLA_Arena_L0_M_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],
    "vla_arena_L0_L": [
        ("VLA_Arena_L0_L_lerobot_openpi", 1.0, "vla_arena_franka"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "SO101_pick": [
        ("pick_dataset_name", 1.0, "SO101"),
    ],

    "arx_x5": [
        ("arx_x5", 1.0, "arx_x5"),
    ],

    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"),
        ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Randomized/click_bell", 1.0, "robotwin"),
        ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Randomized/handover_block", 1.0, "robotwin"),
        ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Randomized/place_fan", 1.0, "robotwin"),
        ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Randomized/scan_object", 1.0, "robotwin"),
        ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"),
        ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],

    "robotwin_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
    ],

    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
}

DOMINO_35_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]

_DOMINO_CLEAN_DYNAMIC = [(f"Clean_Dynamic/{task}", 1.0, "robotwin32") for task in DOMINO_35_TASKS]
_DOMINO_RANDOM_DYNAMIC = [(f"Random_Dynamic/{task}", 1.0, "robotwin32") for task in DOMINO_35_TASKS]
_DOMINO_CLEAN_DYNAMIC_WRAP = [(f"Clean_Dynamic/{task}", 1.0, "robotwin_wrap32") for task in DOMINO_35_TASKS]
_DOMINO_RANDOM_DYNAMIC_WRAP = [(f"Random_Dynamic/{task}", 1.0, "robotwin_wrap32") for task in DOMINO_35_TASKS]

DATASET_NAMED_MIXTURES["domino"] = _DOMINO_CLEAN_DYNAMIC + _DOMINO_RANDOM_DYNAMIC
DATASET_NAMED_MIXTURES["domino_clean"] = _DOMINO_CLEAN_DYNAMIC
DATASET_NAMED_MIXTURES["domino_random"] = _DOMINO_RANDOM_DYNAMIC
DATASET_NAMED_MIXTURES["domino_cotrain"] = (
    _DOMINO_CLEAN_DYNAMIC
    + _DOMINO_RANDOM_DYNAMIC
    + DATASET_NAMED_MIXTURES["robotwin_all"]
)
DATASET_NAMED_MIXTURES["domino_wrap"] = _DOMINO_CLEAN_DYNAMIC_WRAP + _DOMINO_RANDOM_DYNAMIC_WRAP
DATASET_NAMED_MIXTURES["domino_clean_wrap"] = _DOMINO_CLEAN_DYNAMIC_WRAP
DATASET_NAMED_MIXTURES["domino_random_wrap"] = _DOMINO_RANDOM_DYNAMIC_WRAP

DATASET_NAMED_MIXTURES["robotwin_32"] = [
    (dataset_name, dataset_weight, "robotwin32")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin"]
]

DATASET_NAMED_MIXTURES["robotwin_all_32"] = [
    (dataset_name, dataset_weight, "robotwin32")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin_all"]
]

DATASET_NAMED_MIXTURES["robotwin_50"] = [
    (dataset_name, dataset_weight, "robotwin50")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin"]
]

DATASET_NAMED_MIXTURES["robotwin_wrap"] = [
    (dataset_name, dataset_weight, "robotwin_wrap")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin"]
]

DATASET_NAMED_MIXTURES["robotwin_wrap_32"] = [
    (dataset_name, dataset_weight, "robotwin_wrap32")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin"]
]

DATASET_NAMED_MIXTURES["robotwin_wrap_50"] = [
    (dataset_name, dataset_weight, "robotwin_wrap50")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin"]
]

DATASET_NAMED_MIXTURES["robotwin_all_wrap"] = [
    (dataset_name, dataset_weight, "robotwin_wrap")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin_all"]
]

DATASET_NAMED_MIXTURES["robotwin_all_wrap_32"] = [
    (dataset_name, dataset_weight, "robotwin_wrap32")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin_all"]
]

DATASET_NAMED_MIXTURES["robotwin_all_wrap_50"] = [
    (dataset_name, dataset_weight, "robotwin_wrap50")
    for dataset_name, dataset_weight, _ in DATASET_NAMED_MIXTURES["robotwin_all"]
]

# Dynamically load InternData-A1 split_aloha mixture from task list file
import os as _os
_split_aloha_file = _os.path.join(_os.path.dirname(__file__), "../../../playground/Datasets/InternData-A1/split_aloha_tasks.txt")
if _os.path.exists(_split_aloha_file):
    with open(_split_aloha_file) as _f:
        _tasks = [l.strip() for l in _f if l.strip()]
    DATASET_NAMED_MIXTURES["interna1_agilex_flip_wrap_50"] = [
        (t, 1.0, "split_aloha_flip_wrap50") for t in _tasks
    ]


DATASET_NAMED_MIXTURES["droid_manualvel_strict_50_half_views"] = [
    ("DROID", 0.5, "oxe_droid_exterior1_wrist_manualvel_strict_50"),
    ("DROID", 0.5, "oxe_droid_exterior2_wrist_manualvel_strict_50"),
]


def _list_lerobot_task_dirs(root_dir: str) -> list[str]:
    if not _os.path.isdir(root_dir):
        return []

    return [
        task_name
        for task_name in sorted(_os.listdir(root_dir))
        if _os.path.isdir(_os.path.join(root_dir, task_name))
        and _os.path.exists(_os.path.join(root_dir, task_name, "meta", "info.json"))
        and _os.path.exists(_os.path.join(root_dir, task_name, "meta", "modality.json"))
        and _os.path.isdir(_os.path.join(root_dir, task_name, "data"))
    ]



_robocoin_prefix = "RoboCOIN"
_robocoin_tasks = _list_lerobot_task_dirs(f"./playground/Datasets/{_robocoin_prefix}")
if _robocoin_tasks:
    DATASET_NAMED_MIXTURES["robocoin_agilex_flip_wrap_50"] = [
        (f"{_robocoin_prefix}/{task_name}", 1.0, "ROBOCOIN.AgileX_flip_wrap")
        for task_name in _robocoin_tasks
    ]

_molmoact_prefix = "MolmoAct-Dataset"
_molmoact_dataset_names = {
    "household": "molmoact_household_v21",
    "tabletop": "molmoact_tabletop_v21",
}
_molmoact_dataset_paths = {}
for _subset_name, _dataset_name in _molmoact_dataset_names.items():
    _dataset_relpath = f"{_molmoact_prefix}/{_dataset_name}"
    _dataset_abspath = f"./playground/Datasets/{_dataset_relpath}"
    if (
        _os.path.exists(_os.path.join(_dataset_abspath, "meta", "info.json"))
        and _os.path.exists(_os.path.join(_dataset_abspath, "meta", "modality.json"))
        and _os.path.isdir(_os.path.join(_dataset_abspath, "data"))
    ):
        _molmoact_dataset_paths[_subset_name] = _dataset_relpath

if _molmoact_dataset_paths.get("household"):
    DATASET_NAMED_MIXTURES["molmoact_household_2view_manualvel_strict_50_half_views"] = [
        (_molmoact_dataset_paths["household"], 0.5, "molmoact_franka_exterior1_wrist_manualvel_strict_50"),
        (_molmoact_dataset_paths["household"], 0.5, "molmoact_franka_exterior2_wrist_manualvel_strict_50"),
    ]
if _molmoact_dataset_paths.get("tabletop"):
    DATASET_NAMED_MIXTURES["molmoact_tabletop_2view_manualvel_strict_50_half_views"] = [
        (_molmoact_dataset_paths["tabletop"], 0.5, "molmoact_franka_exterior1_wrist_manualvel_strict_50"),
        (_molmoact_dataset_paths["tabletop"], 0.5, "molmoact_franka_exterior2_wrist_manualvel_strict_50"),
    ]
if _molmoact_dataset_paths:
    DATASET_NAMED_MIXTURES["droid_droid100_molmoact_all_manualvel_strict_50"] = (
        DATASET_NAMED_MIXTURES["droid_manualvel_strict_50_half_views"]
        + [("DROID_100", 1.0, "oxe_droid_exterior1_wrist_manualvel_strict_50")]
        + DATASET_NAMED_MIXTURES.get("molmoact_household_2view_manualvel_strict_50_half_views", [])
        + DATASET_NAMED_MIXTURES.get("molmoact_tabletop_2view_manualvel_strict_50_half_views", [])
    )

import json as _json


_dataset_root = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "../../../playground/Datasets")
)


def _dataset_sampling_length(dataset_name: str) -> int:
    meta_dir = _os.path.join(_dataset_root, dataset_name, "meta")
    filtered_stats = [
        _os.path.join(meta_dir, filename)
        for filename in _os.listdir(meta_dir)
        if filename.startswith("stats_gr00t_filtered_") and filename.endswith(".json")
    ] if _os.path.isdir(meta_dir) else []
    for stats_path in sorted(filtered_stats, key=_os.path.getmtime, reverse=True):
        try:
            with open(stats_path) as stats_file:
                stats = _json.load(stats_file)
            filtered_step_count = stats.get("__filtered_step_count")
            if filtered_step_count is not None:
                return int(filtered_step_count)
        except (OSError, ValueError, TypeError):
            continue

    info_path = _os.path.join(meta_dir, "info.json")
    try:
        with open(info_path) as info_file:
            return int(_json.load(info_file)["total_frames"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def _weighted_sampling_mass(entries: list[tuple[str, float, str]]) -> float:
    return sum(float(weight) * float(_dataset_sampling_length(dataset_name)) for dataset_name, weight, _ in entries)


def _balance_entries_to_mass(
    entries: list[tuple[str, float, str]],
    *,
    target_mass: float,
) -> list[tuple[str, float, str]]:
    entries_mass = _weighted_sampling_mass(entries)
    if entries_mass <= 0 or target_mass <= 0:
        multiplier = 1.0
    else:
        multiplier = target_mass / entries_mass
    return [(dataset_name, weight * multiplier, robot_type) for dataset_name, weight, robot_type in entries]


_agilex_franka_non_franka_entries_50 = (
    [
        (f"InternData-A1/{task_name}", weight, robot_type)
        for task_name, weight, robot_type in DATASET_NAMED_MIXTURES.get("interna1_agilex_flip_wrap_50", [])
    ]
    + DATASET_NAMED_MIXTURES.get("robocoin_agilex_flip_wrap_50", [])
)
_agilex_franka_franka_entries_50 = DATASET_NAMED_MIXTURES.get(
    "droid_droid100_molmoact_all_manualvel_strict_50", []
)
_agilex_franka_non_franka_mass_50 = _weighted_sampling_mass(_agilex_franka_non_franka_entries_50)
DATASET_NAMED_MIXTURES["agilex_franka_5data_manualvel_balance33_66_50"] = (
    _agilex_franka_non_franka_entries_50
    + _balance_entries_to_mass(
        _agilex_franka_franka_entries_50,
        target_mass=2.0 * _agilex_franka_non_franka_mass_50,
    )
)
