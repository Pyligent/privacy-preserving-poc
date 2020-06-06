import unittest
import numpy as np


from data_generation import B,  inverse_B, convert_behavior_to_y

class TestUtils(unittest.TestCase):
    def test_B_1(self):
        self.assertEqual(B(3, 1, 0, []), 3)

    def test_B_2(self):
        self.assertEqual(B(3, 2, 0, []), 9)

    def test_B_3(self):
        res = np.exp(0.1) + 3 + 5 * np.exp(0.05)
        self.assertEqual(B(3, 2, 0.1, [1, 0, 0.5]), res)

    def test_B_4(self):
        self.assertEqual(B(3, 1, 0, [], True), [1, 2, 3])

    def test_inverse_B_1(self):
        B_list = [1, 2, 3]
        self.assertEqual(inverse_B(B_list, 2), 2)

    def test_inverse_B_2(self):
        B_list = [1, 2, 3]
        self.assertEqual(inverse_B(B_list, 1.8), 2)

    def test_inverse_B_3(self):
        B_list = [1, 2, 3, 4]
        self.assertEqual(inverse_B(B_list, 2.1), 2)

    def test_inverse_B_4(self):
        B_list = [1, 2, 3, 4]
        self.assertEqual(inverse_B(B_list, 4.1), 5)

    def test_convert_behavior(self):
        actions = [[(1, 3)], [(1, 2), (1, 1)]]
        M = 6
        x_1 = np.zeros(24)
        x_2 = np.zeros(24)
        x_1[0] = 1 # month 1 --> IA
        x_1[9] = 1 # month 4 --> IC
        x_2[0] = 1 # month 1 --> IA
        x_2[8] = 1 # month 3 --> IC
        x_2[15] = 1 # month 4 --> RA
        x_2[22] = 1 # month 5 --> RC
        self.assertEqual(convert_behavior_to_y(actions, M)[0].tolist(), x_1.tolist())
        self.assertEqual(convert_behavior_to_y(actions, M)[1].tolist(), x_2.tolist())

if __name__ == '__main__':
    unittest.main()
