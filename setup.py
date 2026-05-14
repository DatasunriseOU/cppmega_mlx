from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        ext_modules=[
            extension.CMakeExtension(
                "cppmega_mlx.training.native_optim._ext",
                sourcedir="cppmega_mlx/training/native_optim",
            )
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
        package_data={
            "cppmega_mlx.training.native_optim": [
                "*.so",
                "*.dylib",
                "*.metallib",
            ]
        },
        zip_safe=False,
    )
