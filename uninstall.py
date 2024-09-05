import subprocess
import sys

def uninstall_package(package_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])

if __name__ == "__main__":
    packages_to_uninstall = ['bmq_comfyui_utils', 'bmquilting']
    for package in packages_to_uninstall:
        print(f"Uninstalling {package}...")
        uninstall_package(package)
    print("Uninstallation complete.")
