import unittest
import os, shutil
import numpy as np
from spirit_extras import dependency_gatherer

FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_folder"))
FILES = [os.path.join(FOLDER, f"file{i}.txt") for i in range(2)]


class Dependency_Gatherer_Test(unittest.TestCase):
    def generate_file(self, file):
        with open(file, "w") as _:
            pass

    def setUp(self) -> None:
        if os.path.exists(FOLDER):
            shutil.rmtree(FOLDER)
        os.makedirs(FOLDER)

        # create a bunch of files
        for file in FILES:
            self.generate_file(file)

        self.generate_file(os.path.join(FOLDER, "multiple_4.txt"))

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(FOLDER):
            shutil.rmtree(FOLDER)
        return super().tearDown()

    def test_normal(self):
        D = dependency_gatherer.Dependency_Gatherer(verbose=True)
        for f in FILES:
            D.depends(f)

        D.depends(__file__)

        D.depends(
            os.path.join(FOLDER, "does_not_exist.txt"),
            lambda: self.generate_file(os.path.join(FOLDER, "does_not_exist.txt")),
        )
        D.depends(
            os.path.join(FOLDER, "does_not_exist2.txt"),
            lambda: self.generate_file(os.path.join(FOLDER, "does_not_exist2.txt")),
        )
        D.generate(
            os.path.join(FOLDER, "generate.txt"),
            lambda: self.generate_file(os.path.join(FOLDER, "generate.txt")),
        )
        D.check()

    def test_create_multiple(self):
        D = dependency_gatherer.Dependency_Gatherer(verbose=True)
        for f in FILES:
            D.depends(f)

        D.depends(__file__)

        def create_multiple():
            # print("\n\nCB multiple\n\n")
            self.generate_file(os.path.join(FOLDER, "multiple_1.txt"))
            self.generate_file(os.path.join(FOLDER, "multiple_2.txt"))
            self.generate_file(os.path.join(FOLDER, "multiple_3.txt"))
            self.generate_file(os.path.join(FOLDER, "multiple_4.txt"))

        D.depends(
            [
                os.path.join(FOLDER, p)
                for p in [
                    "multiple_1.txt",
                    "multiple_2.txt",
                    "multiple_3.txt",
                    "multiple_4.txt",
                ]
            ],
            create_multiple,
        )
        D.check()

    @unittest.expectedFailure
    def test_fail_no_lambda(self):
        D = dependency_gatherer.Dependency_Gatherer(verbose=True)
        D.depends(os.path.join(FOLDER, "does_not_exist.txt"))
        D.check()

    @unittest.expectedFailure
    def test_fail_wrong_lambda(self):
        D = dependency_gatherer.Dependency_Gatherer(verbose=True)
        D.depends(
            os.path.join(FOLDER, "does_not_exist.txt"),
            lambda: self.generate_file(os.path.join(FOLDER, "not_the_right_file.txt")),
        )
        D.check()

    @unittest.expectedFailure
    def test_fail_lambda_throws(self):
        D = dependency_gatherer.Dependency_Gatherer(verbose=True)

        def t():
            self.generate_file(os.path.join(FOLDER, "not_the_right_file.txt"))
            raise Exception("Im an exception")

        D.depends(os.path.join(FOLDER, "does_not_exist.txt"), t)
        D.check()
