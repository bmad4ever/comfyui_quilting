import subprocess
import sys


def uninstall_package(package_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])


if __name__ == "__main__":
    # note: last item cleans dist-info, if exists due to installation via .toml
    packages_to_uninstall = ['bmq_comfyui_utils', 'bmquilting', 'comfyui_quilting']
    for package in packages_to_uninstall:
        print(f"Uninstalling {package}...")
        uninstall_package(package)
    print("Uninstallation complete.")
