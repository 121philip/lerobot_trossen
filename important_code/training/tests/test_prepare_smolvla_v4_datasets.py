import unittest

from important_code.training.prepare_smolvla_v4_datasets import (
    MERGED_TRAIN_REPO,
    MERGED_VAL_REPO,
    build_commands,
)


class PrepareSmolVLAV4DatasetsTest(unittest.TestCase):
    def test_build_commands_include_expected_split_merge_and_info_steps(self):
        commands = build_commands(push_to_hub=True, include_info=True)

        self.assertEqual(len(commands), 7)

        split_position2 = commands[0]
        self.assertEqual(
            split_position2[:4],
            [
                "lerobot-edit-dataset",
                "--repo_id",
                "kaixiyao/widowxai_grape_grasping_V4_position2",
                "--operation.type",
            ],
        )
        self.assertEqual(
            split_position2[5:7],
            [
                "--operation.splits",
                '{"train":[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47],"val":[8,9,18,19,28,29,38,39,48,49]}',
            ],
        )
        self.assertEqual(split_position2[-2:], ["--push_to_hub", "true"])

        split_position4 = commands[2]
        self.assertEqual(
            split_position4[5:7],
            [
                "--operation.splits",
                '{"train":[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37],"val":[8,9,18,19,28,29,38,39]}',
            ],
        )

        merge_train = commands[3]
        self.assertEqual(merge_train[1:3], ["--new_repo_id", MERGED_TRAIN_REPO])
        self.assertIn("kaixiyao/widowxai_grape_grasping_V4_position2_train", merge_train[6])
        self.assertEqual(merge_train[-2:], ["--push_to_hub", "true"])

        merge_val = commands[4]
        self.assertEqual(merge_val[1:3], ["--new_repo_id", MERGED_VAL_REPO])
        self.assertIn("kaixiyao/widowxai_grape_grasping_V4_position4_val", merge_val[6])

        self.assertEqual(commands[5][1:3], ["--repo_id", MERGED_TRAIN_REPO])
        self.assertEqual(commands[6][1:3], ["--repo_id", MERGED_VAL_REPO])

    def test_build_commands_can_skip_info_and_hub_push(self):
        commands = build_commands(push_to_hub=False, include_info=False)

        self.assertEqual(len(commands), 5)
        self.assertTrue(all("--push_to_hub" not in command for command in commands))
        self.assertEqual(commands[-1][1:3], ["--new_repo_id", MERGED_VAL_REPO])


if __name__ == "__main__":
    unittest.main()
