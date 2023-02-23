import site
import os

def get_package_install_location(package_name):
    for p in site.getsitepackages():
        if os.path.isdir(os.path.join(p, package_name)):
            return os.path.join(p, package_name)
    return None

package_name = 'torch'
install_location = get_package_install_location(package_name)

if install_location:
    print(f"The installation location of {package_name} is: {install_location}")
else:
    print(f"{package_name} is not installed.")
    
    
file_path = os.path.join(install_location, "nn/modules/upsampling.py")

with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for line in lines:
        if line.strip() == "recompute_scale_factor=self.recompute_scale_factor)":
            print("Replacing")
            index = line.index("recompute_scale_factor=self.recompute_scale_factor)")
            print(f"index = {index} is having string {line[index:]}")
            print(line[:index] + ")" +
                  line[index   + len("recompute_scale_factor=self.recompute_scale_factor)"):] 
                 )
            f.write("\n")
            f.write(line[:index] + ")" +
                    line[index   + len("recompute_scale_factor=self.recompute_scale_factor)"):] )
        else:
            f.write(line)
            
