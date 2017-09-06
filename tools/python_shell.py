import subprocess

out = subprocess.check_output(["pacman -Q | grep -e python-"], stderr=subprocess.STDOUT, shell=True)
out = out.decode()
out = out.replace('\n', ' ').split()
#print(out)
out = [out[i] for i in range(len(out)) if i % 2 == 0]
#print(out)
print('sudo pacman -S ', end='')
for pkg in out:
  print(pkg, end=' ')
print()
