import unittest
import os, shutil
from spirit_extras import calculation_folder


class Calculation_Folder_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.FOLDER = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_folder")
        )
        if os.path.exists(self.FOLDER):
            shutil.rmtree(self.FOLDER)
        return super().setUp()

    @unittest.expectedFailure
    def test_creation_fail(self):
        # Should fail because create=False
        folder = calculation_folder.Calculation_Folder(self.FOLDER)

    def test_creation(self):
        # Should fail because create=False
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=True)

        folder["key1"] = 100112.123234
        folder["key2"] = "string"
        folder["key3"] = [1, 2, 3]
        folder["key4"] = dict(key1="a", key2=2)

        folder["weird_key#1"] = "val"
        folder.to_json()

    def test_dictionary(self):
        self.test_creation()
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=False)

        # Check
        for k, v in folder.items():
            self.assertIn(k, folder)
            self.assertEqual(v, folder.get(k))
            self.assertEqual(v, folder[k])

        self.assertEqual(len(folder), 5)

        folder.pop("key1")
        self.assertNotIn("key1", folder)

        self.assertEqual(len(folder), 4)

        update_dict = dict(key1="updated_old_value", key5="new_value")
        folder.update(update_dict)
        self.assertEqual(folder["key1"], "updated_old_value")
        self.assertEqual(folder["key5"], "new_value")

        folder.to_json()

    def test_format(self):
        self.test_creation()
        replace_string = "{key1:.1f}_{key1:.4e}__just_a_string_123%#!!_{weird_key#1}"
        expected_result = "100112.1_1.0011e+05__just_a_string_123%#!!_val"
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=False)
        result_string = folder.format(replace_string)
        self.assertEqual(result_string, expected_result)

    def test_path_manipulations(self):
        self.test_creation()
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=False)
        print(folder.info_string())
        os.path.join(folder, "subdir")

    @unittest.expectedFailure
    def test_copy_fail(self):
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=False)

        # Should fail because create_subdirs=False
        folder.copy_file(__file__, "subfolder/script.py")

    def test_misc(self):
        self.test_creation()
        folder = calculation_folder.Calculation_Folder(self.FOLDER, create=False)

        # File copying
        folder.copy_file(__file__, "script.py")
        folder.copy_file(
            __file__, "subfolder/subsubfolder/script.py", create_subdirs=True
        )

        # Locking
        self.assertTrue(folder.lock())
        self.assertFalse(folder.lock())
        self.assertTrue(folder.locked())
        self.assertTrue(folder.unlock())
        self.assertFalse(folder.locked())

        print(folder.info_string())
